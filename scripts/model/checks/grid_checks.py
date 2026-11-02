"""Grid checks (D, W): the per-second state grid against the round table it was built from, plus independent recomputations of the
alive counts (from deaths.parquet) and of everything (from the raw tome events, using only events at or before the row's tick).

The grid has ~8 M rows, so every row-level test runs on slices of CHUNK rows with a one-row overlap (memory stays far below 4 GB)."""
import glob
import os

import numpy as np

from checks.core import fail, notrun, ok, skip, verdict, warn

CHUNK = 1_500_000
GRID_DTYPES = {"match_key": "String", "round": "Int64", "tick": "Int64", "second": "Int32", "alive_ct": "Int8", "alive_t": "Int8",
               "ct_alive_minus_t": "Int8", "bomb_planted": "Int8", "seconds_since_round_live": "Float32", "seconds_since_plant": "Float32",
               "label_ct_win": "Int8", "w_round": "Float32"}
FLOAT_COLS = ("seconds_since_round_live", "seconds_since_plant", "w_round")
GRID_COLS = ["match_key", "round", "tick", "second", "alive_ct", "alive_t", "ct_alive_minus_t", "bomb_planted", "seconds_since_round_live",
             "seconds_since_plant", "round_number", "is_pistol_round", "is_overtime", "label_ct_win", "w_round", "era", "map_name"]


def round_table(ctx):
    """Used rounds with the anchors the grid must follow, as a small DataFrame."""
    pl, P = ctx.pl, ctx.P
    cap = int(P.clock_cap_s * P.tick_rate)
    return (ctx.used.select("match_key", "round", "freeze", "plant_tick", "end_eff", "final", "max_rounds_final")
            .with_columns(pl.when(pl.col("plant_tick").is_null()).then(pl.min_horizontal(pl.col("end_eff"), pl.col("freeze") + cap))
                          .otherwise(pl.col("end_eff")).alias("cap_end")))


def grid_chunks(ctx, cols, size=CHUNK):
    pl = ctx.pl
    lf = ctx.grid_scan(cols)
    n = int(lf.select(pl.len()).collect().item())
    off = 0
    while off < n:
        lo = max(off - 1, 0)
        yield off, lf.slice(lo, min(size, n - off) + (off - lo)).collect(), (off - lo)
        off += size


def read_grid_rows(path, keys, cols, size=CHUNK):
    """Grid rows (selected columns) of the given (match_key, round) pairs, read slice by slice so that memory stays bounded however large the grid is."""
    import polars as pl
    lf = pl.scan_parquet(path).select(cols)
    n = int(lf.select(pl.len()).collect().item())
    out = [lf.slice(off, size).collect().join(keys, on=["match_key", "round"], how="semi") for off in range(0, n, size)]
    return pl.concat(out) if out else pl.DataFrame(schema={c: pl.Null for c in cols})


def scan_grid(ctx):
    """One pass over the grid: row-level violation counts, per-round aggregates, orientation tables. Cached on the context."""
    if "grid_scan" in ctx._c:
        return ctx._c["grid_scan"]
    pl, P = ctx.pl, ctx.P
    tr, T, CT = P.tick_rate, P.side_t, P.side_ct
    Rs = round_table(ctx)
    tot, per_round, orient = {}, [], []

    def add(k, v):
        tot[k] = tot.get(k, 0) + int(v)

    for off, df, skip in grid_chunks(ctx, GRID_COLS):
        same = ((pl.col("match_key") == pl.col("match_key").shift(1)) & (pl.col("round") == pl.col("round").shift(1))).fill_null(False)
        df = df.with_columns(same.alias("_same"), (pl.col("tick") - pl.col("tick").shift(1)).alias("_dt"))
        if skip:
            df = df.slice(1)
        J = df.join(Rs, on=["match_key", "round"], how="left")
        dt = (pl.col("tick") - pl.col("freeze"))
        planted = (pl.col("plant_tick").is_not_null() & (pl.col("tick") >= pl.col("plant_tick"))).fill_null(False)
        ssp_exp = pl.when(planted).then((pl.col("tick") - pl.col("plant_tick")) / tr).otherwise(-1.0)
        ac, at = pl.col("alive_ct").cast(pl.Int16), pl.col("alive_t").cast(pl.Int16)
        bad = lambda e: e.fill_null(True).sum()  # noqa: E731 - a NULL in a checked column is a violation
        r = J.select(
            pl.len().alias("n"),
            pl.col("freeze").is_null().sum().alias("no_round"),
            bad((pl.col("seconds_since_round_live").cast(pl.Float64) - dt / tr).abs() > P.clock_tol).alias("v_clock"),
            bad(dt % tr != 0).alias("v_phase"),
            bad(pl.col("second").cast(pl.Int64) != dt // tr).alias("v_second"),
            bad(pl.col("bomb_planted").cast(pl.Int64) != planted.cast(pl.Int64)).alias("v_planted"),
            bad((pl.col("seconds_since_plant").cast(pl.Float64) - ssp_exp).abs() > P.clock_tol).alias("v_ssp"),
            bad(pl.col("tick") >= pl.col("cap_end")).alias("v_end"),
            bad(pl.col("label_ct_win").cast(pl.Int64) != (pl.col("final") == CT).cast(pl.Int64)).alias("v_label"),
            bad((ac < 0) | (ac > P.players_per_side) | (at < 0) | (at > P.players_per_side)).alias("v_alive_range"),
            bad(pl.col("ct_alive_minus_t").cast(pl.Int16) != ac - at).alias("v_alive_diff"),
            bad((pl.col("bomb_planted") == 1) != (pl.col("seconds_since_plant") >= 0)).alias("v_bomb_fields"),
            bad(pl.col("is_pistol_round").cast(pl.Int64) != ((pl.col("round") == 1) | (pl.col("round") == pl.col("max_rounds_final") // 2 + 1)).cast(pl.Int64)).alias("v_pistol"),
            bad(pl.col("is_overtime").cast(pl.Int64) != (pl.col("round") > pl.col("max_rounds_final")).cast(pl.Int64)).alias("v_overtime"),
            (pl.col("alive_ct") == 0).sum().alias("ct_wiped_rows"),
            (pl.col("alive_t") == 0).sum().alias("t_wiped_rows"),
            ((pl.col("alive_t") == 0) & (pl.col("bomb_planted") == 0)).sum().alias("t_wiped_unplanted_rows"),
            (pl.col("_same") & (pl.col("_dt") != tr)).sum().alias("v_step"),
            (~pl.col("_same")).sum().alias("blocks"),
            sum([pl.col(c).is_null().cast(pl.Int64).sum() for c in df.columns if not c.startswith("_")]).alias("nulls"),
            sum([(pl.col(c).is_nan() | pl.col(c).is_infinite()).fill_null(False).cast(pl.Int64).sum() for c in FLOAT_COLS]).alias("nonfinite"),
        ).row(0, named=True)
        for k, v in r.items():
            add(k, v)
        per_round.append(df.group_by(["match_key", "round"]).agg(
            pl.col("tick").min().alias("tmin"), pl.col("tick").max().alias("tmax"), pl.len().alias("rows"), pl.col("second").min().alias("smin"),
            pl.col("label_ct_win").min().alias("lmin"), pl.col("label_ct_win").max().alias("lmax"),
            pl.col("w_round").min().alias("wmin"), pl.col("w_round").max().alias("wmax")))
        orient.append(df.group_by(["era", "map_name"]).agg(
            pl.len().alias("n"), (pl.col("alive_ct") > pl.col("alive_t")).sum().alias("n_ct_ahead"),
            ((pl.col("alive_ct") > pl.col("alive_t")) & (pl.col("label_ct_win") == 1)).sum().alias("y_ct_ahead"),
            (pl.col("alive_ct") < pl.col("alive_t")).sum().alias("n_t_ahead"),
            ((pl.col("alive_ct") < pl.col("alive_t")) & (pl.col("label_ct_win") == 1)).sum().alias("y_t_ahead")))
    pr = pl.concat(per_round).group_by(["match_key", "round"]).agg(
        pl.col("tmin").min(), pl.col("tmax").max(), pl.col("rows").sum(), pl.col("smin").min(), pl.col("lmin").min(), pl.col("lmax").max(),
        pl.col("wmin").min(), pl.col("wmax").max())
    ori = pl.concat(orient).group_by(["era", "map_name"]).agg(pl.all().sum())
    ctx._c["grid_scan"] = (tot, pr, ori, Rs)
    return ctx._c["grid_scan"]


def grid_checks(ctx):
    R, P = ctx.R, ctx.P
    pl = ctx.pl
    if not ctx.want("D") and not ctx.want("W"):
        return
    miss = ctx.missing("grid.parquet", "rounds.parquet")
    if miss:
        R.add("D*", "grid checks", notrun(f"missing {miss}"))
        return
    tr, T, CT = P.tick_rate, P.side_t, P.side_ct

    def d0():
        sch = {k: str(v) for k, v in ctx.grid_scan().collect_schema().items()}
        wrong = {k: (sch.get(k), want) for k, want in GRID_DTYPES.items() if sch.get(k) != want}
        return verdict(not wrong, f"{len(sch)} columns; wrong or missing: {wrong}", n=len(sch))
    R.run("D0", "grid schema: required columns with the pinned dtypes (alive counts are Int8, features Float32)", d0)
    if int(ctx.grid_scan().select(pl.len()).collect().item()) == 0:
        R.add("D1", "grid has rows", fail("grid.parquet has 0 rows: every grid check would pass vacuously, which is a failure", n=0))
        return
    try:
        tot, pr, ori, Rs = scan_grid(ctx)
    except Exception as exc:  # noqa: BLE001
        import traceback
        R.add("D1", "grid scan", fail(f"grid scan raised {type(exc).__name__}: {str(exc)[:300]}"), tb=traceback.format_exc())
        return
    n = tot["n"]
    ctx.record("grid", "rows", n)
    ex = pr.join(Rs, on=["match_key", "round"], how="full", coalesce=True)

    R.run("D1", "no duplicate (match, round, tick) rows: every round is one contiguous block with strictly increasing ticks",
          lambda: verdict(tot["blocks"] == pr.height and tot["v_step"] == 0, f"{n:,} rows, {pr.height:,} rounds, {tot['blocks']:,} blocks", n=n), min_n=P.min_grid_rows)
    R.run("D2", f"ticks rise in exact {tr}-tick steps within a round",
          lambda: verdict(tot["v_step"] == 0, f"{tot['v_step']} steps != {tr}", n=n), min_n=P.min_grid_rows)

    def d3():
        bad = pr.filter((pl.col("smin") != 0) | (pl.col("lmin") != pl.col("lmax"))).height
        return verdict(bad == 0, f"{pr.height:,} rounds; {bad} do not start at second 0 or have more than one label", n=pr.height)
    R.run("D3", "every round starts at second 0 and has one constant label", d3, min_n=P.min_used_rounds)
    R.run("D4", "alive counts within 0..5 and ct_alive_minus_t consistent",
          lambda: verdict(tot["v_alive_range"] + tot["v_alive_diff"] == 0, f"range violations {tot['v_alive_range']}, difference violations {tot['v_alive_diff']}", n=n), min_n=P.min_grid_rows)
    R.run("D5", "bomb fields consistent (seconds_since_plant >= 0 iff planted)",
          lambda: verdict(tot["v_bomb_fields"] == 0, f"{tot['v_bomb_fields']} violations", n=n), min_n=P.min_grid_rows)
    R.run("D6", "undecided states only: no grid row with the CT side wiped",
          lambda: verdict(tot["ct_wiped_rows"] == 0, f"{tot['ct_wiped_rows']} rows with alive_ct == 0", n=n), min_n=P.min_grid_rows)
    R.run("D7", "grid label equals the round table's final label, every used round has grid rows and no other round does", lambda: verdict(
        tot["v_label"] == 0 and tot["no_round"] == 0 and ex.filter(pl.col("rows").is_null() | pl.col("freeze").is_null()).height == 0,
        f"{pr.height:,} grid rounds vs {Rs.height:,} used rounds; label mismatches {tot['v_label']}; grid rows of unused rounds {tot['no_round']}; "
        f"used rounds without rows {ex.filter(pl.col('rows').is_null()).height}", n=Rs.height), min_n=P.min_used_rounds)

    def d8():
        parts = {k: tot[k] for k in ("v_clock", "v_phase", "v_second", "v_planted", "v_ssp", "v_pistol", "v_overtime")}
        return verdict(sum(parts.values()) == 0, f"{n:,} rows vs the round table (freeze tick, plant tick); violations {parts}", n=n)
    R.run("D8", "grid anchored to the round table: seconds_since_round_live == (tick - freeze)/64, bomb_planted == (tick >= plant_tick), "
                "seconds_since_plant == (tick - plant)/64, second and phase, pistol/overtime flags", d8, min_n=P.min_grid_rows)

    def d9():
        g = ex.filter(pl.col("rows").is_not_null() & pl.col("freeze").is_not_null())
        exp_rows = ((pl.col("cap_end") - pl.col("freeze") + (tr - 1)) // tr).clip(lower_bound=0)
        v_first = g.filter(pl.col("tmin") != pl.col("freeze")).height
        v_hi = g.filter(pl.col("tmax") >= pl.col("cap_end")).height
        v_lo = g.filter(pl.col("tmax") < pl.col("cap_end") - tr).height
        v_cnt = g.filter(pl.col("rows") != exp_rows).height
        v_w = g.filter(((pl.col("wmin") - 1.0 / pl.col("rows")).abs() > 1e-6) | ((pl.col("wmax") - 1.0 / pl.col("rows")).abs() > 1e-6)).height
        extra = tot["v_end"]
        msg = (f"{g.height:,} rounds; first tick != freeze tick {v_first}; last tick at/after the effective end {v_hi}; last tick more than {tr} before it {v_lo}; "
               f"points != ceil((end - freeze)/{tr}) {v_cnt}; w_round != 1/points {v_w}; rows at/after the effective end {extra}")
        return verdict(v_first + v_hi + v_lo + v_cnt + v_w + extra == 0, msg, n=g.height)
    R.run("D9", "grid span: first tick == freeze tick, every row strictly before the effective end (116 s cap when unplanted), points per round == ceil formula", d9, min_n=P.min_used_rounds)

    R.run("D11", "no null, NaN or infinite value in the grid",
          lambda: verdict(tot["nulls"] == 0 and tot["nonfinite"] == 0, f"{tot['nulls']} nulls, {tot['nonfinite']} NaN/inf in {n:,} rows", n=n), min_n=P.min_grid_rows)

    # ---- W: wipe symmetry and orientation
    def w1():
        nt = tot["t_wiped_rows"]
        if nt == 0:
            return notrun("no grid row with the T side wiped (planted rounds that run to the defuse) in this store")
        return verdict(tot["t_wiped_unplanted_rows"] == 0, f"{nt:,} rows with alive_t == 0, {tot['t_wiped_unplanted_rows']} of them with no plant (such a state is decided and must not be in the grid)", n=nt)
    R.run("W1", "T-side mirror of D6: alive_t == 0 only after the plant", w1)

    def w2():
        by_era = ori.group_by("era").agg(pl.all().exclude("era", "map_name").sum())
        bad, lines, nn = [], [], 0
        for scope, tab, keys in (("era", by_era, ["era"]), ("era|map", ori, ["era", "map_name"])):
            for r in tab.iter_rows(named=True):
                if r["n_ct_ahead"] < 1000 or r["n_t_ahead"] < 1000:
                    continue
                a, b = r["y_ct_ahead"] / r["n_ct_ahead"], r["y_t_ahead"] / r["n_t_ahead"]
                nn += 1
                name = "|".join(str(r[k]) for k in keys)
                if scope == "era":
                    lines.append(f"{name}: P(CT win | CT ahead) {a:.3f} vs {b:.3f}")
                if not a > b:
                    bad.append(f"{name} {a:.3f} <= {b:.3f}")
        return verdict(not bad, "; ".join(lines) + (f"; WRONG WAY: {bad}" if bad else f"; {nn} cells"), n=nn)
    R.run("W2", "orientation: CT wins more often when ahead in alive players than when behind, per era and per map", w2)

    # ---- D10: alive counts recomputed from deaths.parquet with an as-of join (an algorithm independent of make_grid's searchsorted)
    def d10():
        if ctx.missing("deaths.parquet"):
            return notrun("deaths.parquet missing")
        U = ctx.used.select("match_key", "round")
        k = min(P.grid_recompute_rounds, U.height)
        samp = U.sample(n=k, seed=0) if k < U.height else U
        g = read_grid_rows(ctx.p("grid.parquet"), samp, ["match_key", "round", "tick", "alive_ct", "alive_t"])
        dth = pl.read_parquet(ctx.p("deaths.parquet")).filter(pl.col("side_ok")).join(samp, on=["match_key", "round"], how="semi")
        g = g.sort("tick")
        import warnings
        for side, col in ((CT, "alive_ct"), (T, "alive_t")):
            d = dth.filter(pl.col("side") == side).sort(["match_key", "round", "tick"])
            d = d.with_columns((pl.int_range(pl.len()).over(["match_key", "round"]) + 1).alias("cum"))
            d = d.group_by(["match_key", "round", "tick"]).agg(pl.col("cum").max()).sort("tick")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # "sortedness cannot be checked with by groups": both frames are sorted by tick above
                g = g.join_asof(d, on="tick", by=["match_key", "round"], strategy="backward", suffix="_d").with_columns(
                    (P.players_per_side - pl.col("cum").fill_null(0)).alias("exp_" + col)).drop("cum")
        bad_ct = g.filter(pl.col("alive_ct").cast(pl.Int64) != pl.col("exp_alive_ct")).height
        bad_t = g.filter(pl.col("alive_t").cast(pl.Int64) != pl.col("exp_alive_t")).height
        return verdict(bad_ct + bad_t == 0, f"{g.height:,} rows of {samp.height:,} rounds; alive_ct differs {bad_ct}, alive_t differs {bad_t} from 5 minus the deaths at or before the tick",
                       n=g.height)
    R.run("D10", "alive counts equal 5 minus the deaths at or before the row tick, recomputed from deaths.parquet (as-of join, sample of rounds)", d10, min_n=1)

    # ---- D12: causal recompute from the raw tome events
    if ctx.fast:
        R.add("D12", "causal recompute canary from raw tome events", skip("--fast: raw-tome canary skipped"))
    else:
        R.run("D12", "causal recompute canary: alive counts, bomb state and clock recomputed from raw tome events with tick <= the row tick", lambda: causal_canary(ctx), min_n=1)


def _channel_pages(ctx, ch):
    d = ctx.channel_dir(ch)
    return sorted(glob.glob(os.path.join(d, "dataframe_*"))) if d else []


def causal_canary(ctx):
    """For a sample of used rounds, recompute alive_ct, alive_t, bomb_planted, seconds_since_round_live and seconds_since_plant for every
    grid tick from the RAW tome events, looking only at events with tick <= the row tick (so a feature that peeks at the future, or a
    tick-boundary off-by-one, shows up), and compare with the grid."""
    pl, P = ctx.pl, ctx.P
    tr, T, CT = P.tick_rate, P.side_t, P.side_ct
    need = {ch: _channel_pages(ctx, ch) for ch in ("player_death", "round_state", "bomb_state", "player_info", "round_end")}
    miss = [c for c, p in need.items() if not p]
    if miss:
        return notrun(f"raw tome channels missing: {miss} (tomes dir {ctx.tomes_dir})")
    U = ctx.used.filter(pl.col("missing_side").is_null() & (pl.col("deaths_unresolved") == 0)).select("match_key", "round", "freeze", "end")
    if U.height == 0:
        return notrun("no used round with a fully resolved roster")
    samp = U.sample(n=min(P.causal_rounds, U.height), seed=1)
    keys = samp.select("match_key", "round")
    mks = samp["match_key"].unique().to_list()

    def scan(ch, cols):
        return pl.scan_parquet(need[ch]).filter(pl.col("match_key").is_in(mks)).select(cols).join(keys.lazy(), on=["match_key", "round"], how="semi").collect()
    deaths = scan("player_death", ["match_key", "round", "tick", "player_id", "player_team_code"])
    roster = scan("player_info", ["match_key", "round", "player_id", "team_code"]).unique(["match_key", "round", "player_id"], keep="last")
    rs = scan("round_state", ["match_key", "round", "tick", "event_type"]).filter(pl.col("event_type") == "round_freeze_end")
    bomb = scan("bomb_state", ["match_key", "round", "tick", "event_type"]).filter(pl.col("event_type") == "bomb_planted")
    ends = scan("round_end", ["match_key", "round", "tick"]).sort("tick").unique(["match_key", "round"], keep="first")
    freeze = rs.sort("tick").unique(["match_key", "round"], keep="first").select("match_key", "round", pl.col("tick").alias("fz"))
    end = ends.select("match_key", "round", pl.col("tick").alias("en"))
    deaths = (deaths.join(freeze, on=["match_key", "round"]).join(end, on=["match_key", "round"])
              .filter((pl.col("tick") >= pl.col("fz")) & (pl.col("tick") <= pl.col("en"))).sort(["match_key", "round", "tick"])
              .unique(["match_key", "round", "player_id"], keep="first", maintain_order=True)
              .join(roster.rename({"team_code": "rt"}), on=["match_key", "round", "player_id"], how="left")
              .with_columns(pl.coalesce([pl.col("player_team_code"), pl.col("rt")]).alias("side")))
    bomb = bomb.join(freeze, on=["match_key", "round"]).join(end, on=["match_key", "round"]).filter(
        (pl.col("tick") >= pl.col("fz")) & (pl.col("tick") <= pl.col("en") + tr))
    plant = bomb.group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("pt"))
    grid = read_grid_rows(ctx.p("grid.parquet"), keys, ["match_key", "round", "tick", "alive_ct", "alive_t", "bomb_planted", "seconds_since_round_live",
                                                         "seconds_since_plant"]).sort(["match_key", "round", "tick"])
    d_by = {}
    for part in deaths.select("match_key", "round", "tick", "side").partition_by(["match_key", "round"], maintain_order=True):
        d_by[(part["match_key"][0], part["round"][0])] = (part["tick"].to_numpy(), part["side"].fill_null(-1).to_numpy())
    fz_d = {(a, b): c for a, b, c in freeze.iter_rows()}
    pt_d = {(a, b): c for a, b, c in plant.iter_rows()}
    n_rows, bad = 0, {"alive_ct": 0, "alive_t": 0, "bomb": 0, "clock": 0, "since_plant": 0}
    first_bad = None
    for (mk, rd), gr in _iter_rounds(grid):
        fz = fz_d.get((mk, rd))
        if fz is None:
            bad["clock"] += gr.height
            n_rows += gr.height
            continue
        pt = pt_d.get((mk, rd))
        d_t, d_s = d_by.get((mk, rd), (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)))
        for row in gr.iter_rows(named=True):
            t = row["tick"]
            seen = d_t <= t  # only events at or before the row tick are visible
            ect = P.players_per_side - int(((d_s == CT) & seen).sum())
            et = P.players_per_side - int(((d_s == T) & seen).sum())
            planted = pt is not None and pt <= t
            e_clock = (t - fz) / tr
            e_ssp = (t - pt) / tr if planted else -1.0
            b = (row["alive_ct"] != ect, row["alive_t"] != et, bool(row["bomb_planted"]) != planted,
                 abs(row["seconds_since_round_live"] - e_clock) > P.clock_tol, abs(row["seconds_since_plant"] - e_ssp) > P.clock_tol)
            for k, v in zip(bad, b):
                bad[k] += int(v)
            if any(b) and first_bad is None:
                first_bad = (mk[-12:], rd, t, row, (ect, et, planted, e_clock, e_ssp))
            n_rows += 1
    tot_bad = sum(bad.values())
    msg = f"{n_rows:,} grid rows of {samp.height} rounds recomputed from raw events; mismatches {bad}"
    if first_bad:
        msg += f"; first: round {first_bad[1]} tick {first_bad[2]} grid {first_bad[3]} vs recomputed {first_bad[4]}"
    return verdict(tot_bad == 0, msg, n=n_rows)


def _iter_rounds(grid):
    """(match_key, round) -> sub-frame, for a grid slice sorted by match_key, round, tick."""
    for part in grid.partition_by(["match_key", "round"], maintain_order=True):
        yield (part["match_key"][0], part["round"][0]), part
