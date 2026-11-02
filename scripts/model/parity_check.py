"""Train/serve parity on real rows: the exported model, fed only what a live service would have (the table of living players, the bomb
state and the clock), must reproduce the training pipeline's inputs and predictions for the same grid rows.

  python -I -X utf8 scripts\\model\\parity_check.py --model models\\winprob_v1.json --fallback models\\winprob_v1_basic.json \\
         --sample data\\combined\\export_winprob_v1_test_sample.parquet [--n 5000] [--out PATH] [--allow-missing]

Model- and store-agnostic: the store, the maps and the side codes come from the sample's location and the model's meta; the tolerance comes
from the export report (max(1e-6, 100 x the measured GPU-vs-CPU difference), else 1e-4). Checks (PASS / WARN / FAIL like e2e_checks.py):
  P1 inputs         model, sample, 1 Hz tables exist; the sample holds every model feature; at least --min-rows rows (never a vacuous pass)
  P2 coverage       every in-scope map, bomb state and alive pattern has rows in the compared set
  P3 exclusions     rounds the exporter must have dropped (roster-defect: wp_common.drop_roster_defect_rounds; table != grid at some tick) are
                    removed from the comparison and REPORTED SEPARATELY with their own difference statistics; a bounded share is a WARN (stale
                    sample), a larger share a FAIL
  P4 table-vs-grid  on the compared rounds the table has exactly alive_ct / alive_t rows at EVERY tick of the round (not only the sampled ticks)
  P5 features       the model input vector serving builds equals the stored training features exactly (all features, all rows)
  P6 predictions    serving p == p_pipeline within the tolerance, status "ok", no "model input(s) not given" warning, finite
  P7 batch==single  predict() equals predict_many() element by element on a varied, shuffled mixed batch
  P8 fallback       the alive-count model reproduces its own sample; without a player table the full model answers "fallback" with that number
The JSON result goes to --out (default data/<store>/parity_<model>.json under the project root of this checkout) atomically, with hashes of
the model, meta and sample, the row counts, the exclusions and the thresholds; a "running" file with passed=false is written first.
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from checks import core  # noqa: E402

core.setup_env(threads=2)
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from checks.core import Reporter, fail, notrun, ok, verdict, warn  # noqa: E402
from snapshot_features import _side  # noqa: E402
from winprob import WinProb  # noqa: E402
import wp_items as WI  # noqa: E402

PLAYER_COLS = {"side": "side", "health": "health", "armor": "armor", "has_helmet": "has_helmet", "has_defuser": "has_defuser", "is_defusing": "is_defusing",
               "primary": "inv_primary", "flashbang": "inv_flashbang", "smokegrenade": "inv_smokegrenade", "hegrenade": "inv_hegrenade",
               "molotov": "inv_molotov", "incgrenade": "inv_incgrenade", "decoy": "inv_decoy", "equipment_value": "equipment_value_calc"}
STATUS_KEYS = ["match_key", "round", "tick"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Train/serve parity on real rows (see the module docstring).")
    ap.add_argument("--model", required=True)
    ap.add_argument("--fallback", default=None)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--fallback-sample", default=None, help="sample of the fallback model (default: export_<fallback>_test_sample.parquet beside --sample)")
    ap.add_argument("--n", type=int, default=5000, help="rows of the sample to compare (0 = all)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--store", default=None, help="store name (default: the folder name of the sample)")
    ap.add_argument("--root", default=None, help="project root that holds data/status_1hz (default: the root of this checkout)")
    ap.add_argument("--status-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tol", type=float, default=None, help="probability tolerance (default max(1e-6, 100 x the export's measured difference), else 1e-4)")
    ap.add_argument("--feature-tol", type=float, default=1e-6)
    ap.add_argument("--min-rows", type=int, default=100)
    ap.add_argument("--min-rows-per-map", type=int, default=20)
    ap.add_argument("--excluded-share-max", type=float, default=0.01, help="largest share of sampled rows in rounds the exporter should have dropped before it is a FAIL")
    ap.add_argument("--batch-size", type=int, default=300)
    ap.add_argument("--min-age", type=float, default=120.0, help="ignore status day files written in the last N seconds")
    ap.add_argument("--allow-missing", action="store_true")
    return ap.parse_args(argv)


def sha(path):
    return core.sha256_file(path) if path and os.path.exists(path) else None


class Parity:
    def __init__(self, a):
        self.a = a
        self.root = os.path.abspath(a.root or core.project_root())
        self.sample_path = os.path.abspath(a.sample)
        self.data_dir = os.path.dirname(self.sample_path)
        self.store = a.store or os.path.basename(self.data_dir)
        up = os.path.dirname(self.data_dir)
        self.data_root = os.path.dirname(up) if os.path.basename(up) == "data" else None  # the project root that owns the sample (may differ from self.root)
        self.status_dir = a.status_dir or os.path.join(self.root, "data", "status_1hz")
        self.model_path = os.path.abspath(a.model)
        self.name = os.path.splitext(os.path.basename(a.model))[0]
        self.meta_path = os.path.splitext(self.model_path)[0] + ".meta.json"
        self.meta = json.load(open(self.meta_path, encoding="utf-8")) if os.path.exists(self.meta_path) else None
        self.R = Reporter(allow_missing=a.allow_missing)
        self.info = {"excluded": {}}
        self.files = core.list_parquets(self.status_dir, a.min_age)  # not a file the extractor is still writing

    # ---- tolerance from the export report
    def tolerance(self):
        if self.a.tol is not None:
            return self.a.tol, "--tol"
        rep = os.path.join(self.data_dir, f"export_{self.name}.json")
        try:
            d = json.load(open(rep, encoding="utf-8"))["cpu_inference_of_gpu_model_max_abs_diff"]
            return max(1e-6, 100.0 * float(d)), f"100 x the export's measured {d:.2e}"
        except Exception:  # noqa: BLE001
            return 1e-4, "default (no export report)"

    # ---- reading
    def table_counts(self, rounds, mks):
        """Rows of the living-player table per (match, round, tick, side count) for every tick of the given rounds."""
        parts = []
        for f in self.files:  # one day file at a time: the whole 1 Hz history never sits in memory
            lf = (pl.scan_parquet(f).select("match_key", pl.col("round").cast(pl.Int64), "tick", "side").filter(pl.col("match_key").is_in(mks))
                  .join(rounds.lazy(), on=["match_key", "round"], how="semi"))
            parts.append(lf.group_by(STATUS_KEYS).agg((pl.col("side") == WI.SIDE_CT).sum().alias("n_ct"), (pl.col("side") == WI.SIDE_T).sum().alias("n_t")).collect())
        return pl.concat(parts).group_by(STATUS_KEYS).agg(pl.col("n_ct").sum(), pl.col("n_t").sum())

    def player_rows(self, keys, mks):
        cols = ["match_key", pl.col("round").cast(pl.Int64), "tick"] + [c for c in PLAYER_COLS.values()]
        parts = [pl.scan_parquet(f).select(*cols).filter(pl.col("match_key").is_in(mks)).join(keys.lazy(), on=STATUS_KEYS, how="semi").collect() for f in self.files]
        return pl.concat(parts)

    @staticmethod
    def states_from(samp, rows):
        by = {}
        names = list(PLAYER_COLS)
        src = [PLAYER_COLS[n] for n in names]
        for tup in rows.select(STATUS_KEYS + src).iter_rows():
            by.setdefault(tup[:3], []).append(dict(zip(names, tup[3:])))
        states, no_table = [], 0
        for r in samp.iter_rows(named=True):
            pls = by.get((r["match_key"], r["round"], r["tick"]))
            if not pls:
                no_table += 1
            states.append({"seconds_since_round_live": r["seconds_since_round_live"], "bomb_planted": r["bomb_planted"], "seconds_since_plant": r["seconds_since_plant"],
                           "map_name": r["map_name"], "mode": "competitive", "players": pls or []})
        return states, no_table


def run(argv=None):
    a = parse_args(argv)
    t0 = time.time()
    P = Parity(a)
    R = P.R
    out = a.out or os.path.join(P.root, "data", P.store, f"parity_{P.name}.json")
    info = P.info

    def write(status):
        rep = {"status": status, "passed": status == "complete" and R.all_passed(), "model": P.name, "store": P.store, "counts": R.counts(), "checks": R.results,
               "info": info, "elapsed_s": round(time.time() - t0, 1), "peak_rss_mb": core.peak_rss_mb(), "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "argv": sys.argv[1:], "script_sha256": sha(os.path.abspath(__file__))}
        core.dump_json_atomic(rep, out)
    write("running")
    status = "crashed"
    try:
        status = _run(P, a, info)
    except BaseException as exc:  # noqa: BLE001
        R.add("RUN", "run aborted", fail(f"{type(exc).__name__}: {str(exc)[:300]}"), tb=traceback.format_exc())
        status = "aborted"
        raise
    finally:
        if not R.results:
            R.add("RUN", "at least one check ran", fail("no check ran"))
        write(status)
        print("\n" + R.summary_line(), flush=True)
        print("PASS: serving reproduces the training pipeline" if R.all_passed() else "REVIEW: differences found (see the lines above)", flush=True)
        print(f"result -> {out}  ({time.time() - t0:.0f}s, peak {core.peak_rss_mb()} MiB)", flush=True)
    sys.exit(0 if R.all_passed() else 1)


def _run(P, a, info):
    R = P.R
    tol, tol_how = P.tolerance()
    info.update({"tolerance": tol, "tolerance_basis": tol_how, "feature_tol": a.feature_tol, "model_sha256": sha(P.model_path), "meta_sha256": sha(P.meta_path),
                 "sample_sha256": sha(P.sample_path), "fallback_sha256": sha(a.fallback), "status_files": len(P.files),
                 "status_first_last": [os.path.basename(P.files[0]), os.path.basename(P.files[-1])] if P.files else None, "seed": a.seed})
    missing = [p for p in (P.model_path, P.meta_path, P.sample_path) if not os.path.exists(p)]
    if missing or not P.files:
        R.add("P1", "inputs exist", notrun(f"missing {missing}; status files in {P.status_dir}: {len(P.files)}"))
        return "complete"
    wp = WinProb.load(P.model_path, a.fallback)
    feats = list(wp.features)
    samp_all = pl.read_parquet(P.sample_path)
    miss_cols = [f for f in feats if f not in samp_all.columns] + [c for c in ("match_key", "round", "tick", "seconds_since_round_live", "bomb_planted", "seconds_since_plant",
                                                                                  "map_name", "alive_ct", "alive_t", "p_pipeline") if c not in samp_all.columns]
    n_take = samp_all.height if a.n <= 0 else min(a.n, samp_all.height)
    samp = samp_all.sample(n=n_take, seed=a.seed) if n_take < samp_all.height else samp_all
    R.add("P1", "inputs: model, meta, sample and 1 Hz tables exist; the sample holds every model feature; enough rows",
          verdict(not miss_cols and samp.height >= a.min_rows, f"{samp.height:,} of {samp_all.height:,} sample rows compared (floor {a.min_rows}); {len(feats)} model features; "
                  f"missing columns {miss_cols}; {len(P.files)} status files; tolerance {tol:.2e} ({tol_how})", n=samp.height))
    if miss_cols or samp.height < a.min_rows:
        return "complete"
    maps = sorted(set((P.meta.get("scope") or {}).get("maps", [])) or {"de_" + f[len("map_is_"):] for f in feats if f.startswith("map_is_")})

    # ---- exclusions: rounds the exporter must have dropped
    rounds = samp.select("match_key", "round").unique()
    mks = rounds["match_key"].unique().to_list()
    excl_roster = pl.DataFrame({"match_key": [], "round": []}, schema={"match_key": pl.String, "round": rounds.schema["round"]})
    rosterless = None
    if P.data_root is not None and os.path.exists(os.path.join(P.data_root, "data", P.store, "rounds.parquet")):
        import wp_common as W
        old = W.ROOT
        W.ROOT = P.data_root  # the project that owns the sample (read-only access to its rounds.parquet)
        try:
            kept = W.drop_roster_defect_rounds(rounds, P.store)
        finally:
            W.ROOT = old
        excl_roster = rounds.join(kept, on=["match_key", "round"], how="anti")
    else:
        rosterless = f"no rounds.parquet beside the sample ({P.data_dir}): the roster-defect exclusion cannot be applied"
    grid_path = os.path.join(P.data_dir, "grid.parquet")
    tc = P.table_counts(rounds, mks)
    if os.path.exists(grid_path):
        from checks import grid_checks
        g = grid_checks.read_grid_rows(grid_path, rounds, ["match_key", "round", "tick", "alive_ct", "alive_t"])
    else:
        g = samp.select("match_key", "round", "tick", "alive_ct", "alive_t")
    j = g.join(tc, on=STATUS_KEYS, how="left").with_columns(pl.col("n_ct").fill_null(0), pl.col("n_t").fill_null(0))
    bad_ticks = j.filter((pl.col("n_ct") != pl.col("alive_ct")) | (pl.col("n_t") != pl.col("alive_t")))
    excl_incons = bad_ticks.select("match_key", "round").unique()
    excl_any = pl.concat([excl_roster.select("match_key", "round"), excl_incons.select("match_key", "round")]).unique()
    in_excl = samp.join(excl_any.with_columns(pl.lit(True).alias("_x")), on=["match_key", "round"], how="left").with_columns(pl.col("_x").fill_null(False))
    n_excl_rows = int(in_excl["_x"].sum())
    n_roster_rows = samp.join(excl_roster.select("match_key", "round"), on=["match_key", "round"], how="semi").height
    n_incons_rows = samp.join(excl_incons, on=["match_key", "round"], how="semi").height
    info["excluded"] = {"roster_defect_rounds": excl_roster.height, "roster_defect_rows": n_roster_rows, "table_vs_grid_inconsistent_rounds": excl_incons.height,
                        "table_vs_grid_inconsistent_rows": n_incons_rows, "rows_total_excluded": n_excl_rows, "rounds_total": rounds.height,
                        "inconsistent_tick_rows": bad_ticks.height, "first_inconsistent": [list(x) for x in excl_incons.head(10).iter_rows()]}
    share = n_excl_rows / max(samp.height, 1)

    def p3():
        msg = (f"{rounds.height:,} sampled rounds: roster-defect {excl_roster.height} rounds / {n_roster_rows} rows; table != grid at some tick {excl_incons.height} rounds / "
               f"{n_incons_rows} rows ({bad_ticks.height} ticks); excluded from the comparison {n_excl_rows} of {samp.height:,} rows ({share:.2%}, bound {a.excluded_share_max:.2%}); "
               f"a fresh export contains none of them" + (f"; {rosterless}" if rosterless else ""))
        if rosterless:
            return notrun(rosterless)
        if share > a.excluded_share_max:
            return fail(msg, n=samp.height)
        if n_excl_rows:
            return warn(msg + " (the sample predates the exporter's exclusion: re-export)", n=samp.height)
        return ok(msg, n=samp.height)
    R.run("P3", "exclusions reported separately: roster-defect rounds and rounds where the table disagrees with the grid are not in a fresh sample", p3)
    comp = in_excl.filter(~pl.col("_x")).drop("_x")
    if comp.height < a.min_rows:
        R.add("P4", "compared set is large enough", fail(f"only {comp.height} rows left after the exclusions (floor {a.min_rows})", n=comp.height))
        return "complete"

    # ---- coverage of the compared set
    def p2():
        per_map = {m: int((comp["map_name"] == m).sum()) for m in maps}
        short = {m: c for m, c in per_map.items() if c < a.min_rows_per_map}
        planted, unplanted = int((comp["bomb_planted"] == 1).sum()), int((comp["bomb_planted"] == 0).sum())
        patt = int(comp.select((pl.col("alive_ct").cast(pl.String) + "v" + pl.col("alive_t").cast(pl.String)).n_unique()).item())
        return verdict(not short and planted > 0 and unplanted > 0, f"rows per map {per_map}; maps below {a.min_rows_per_map}: {short}; planted {planted} / unplanted {unplanted}; {patt} alive patterns", n=comp.height)
    R.run("P2", "coverage: every in-scope map, both bomb states and a spread of alive patterns have rows in the compared set", p2)

    # ---- build states
    rows = P.player_rows(samp.select(STATUS_KEYS), mks)
    states, no_table = P.states_from(comp, rows)
    ex_rows = in_excl.filter(pl.col("_x"))
    ex_states, _ = P.states_from(ex_rows, rows) if ex_rows.height else ([], 0)
    info["rows_compared"], info["rows_without_player_table"] = comp.height, no_table

    def p4():
        """Every compared round has the grid's alive counts in the table at every tick (all ticks were checked in P3); here the sampled rows are recounted per side
        from the state dicts serving receives, with the side codes read through snapshot_features._side."""
        bad_ct = bad_t = 0
        for s_, ac, at in zip(states, comp["alive_ct"].to_list(), comp["alive_t"].to_list()):
            sides = [_side(p_["side"]) for p_ in s_["players"]]
            bad_ct += sides.count(WI.SIDE_CT) != ac
            bad_t += sides.count(WI.SIDE_T) != at
        return verdict(bad_ct + bad_t == 0 and no_table == 0, f"{len(states):,} compared rows: CT table count != alive_ct on {bad_ct}, T table count != alive_t on {bad_t}; "
                       f"rows without any player table {no_table}; {excl_incons.height} rounds with a disagreement at SOME tick were set aside in P3", n=len(states))
    R.run("P4", "on every compared row the player table has alive_ct CT and alive_t T rows (per side, side codes through snapshot_features._side)", p4)

    # ---- P5/P6: the model input vector and the prediction
    preps = [wp._prepare(s) for s in states]
    ok_prep = np.array(["model" in p for p in preps])
    X = np.full((comp.height, len(feats)), np.nan)
    for i, p_ in enumerate(preps):
        if "model" in p_ and p_["model"] is wp.primary:
            X[i] = p_["x"][0]
    stored = comp.select(feats).to_numpy().astype(np.float64)
    d = np.abs(X - stored)
    bad_cell = ~(d <= a.feature_tol)  # NaN on either side is a mismatch
    per_feat = {f: int(bad_cell[:, k].sum()) for k, f in enumerate(feats) if bad_cell[:, k].any()}
    n_bad_rows = int(bad_cell.any(axis=1).sum())
    info["feature_mismatch_by_feature"] = per_feat
    no_snap = int(sum(1 for p_ in preps if p_.get("model") is not wp.primary))
    R.run("P5", "the model input vector serving builds from the player table equals the stored training features (all features, all compared rows)", lambda: verdict(
        n_bad_rows == 0, f"{len(feats)} features x {comp.height:,} rows; rows with any feature off by more than {a.feature_tol:g}: {n_bad_rows}; by feature {dict(list(per_feat.items())[:8])}; "
                         f"rows where serving built no snapshot input {no_snap}", n=comp.height))

    out_ = wp.predict_many(states)
    p = np.array([o["p_ct_win"] for o in out_])
    ref = comp["p_pipeline"].to_numpy().astype(np.float64)
    st = [o["status"] for o in out_]
    counts = {s: st.count(s) for s in sorted(set(st))}
    warn_missing = sum(any("not given" in w for w in o["warnings"]) for o in out_)
    bad_p = ~(np.abs(p - ref) <= tol)
    non_ok = sum(1 for s in st if s != "ok")
    info["status_counts"] = counts
    info["max_abs_diff"] = float(np.max(np.abs(p - ref))) if len(p) else None
    R.run("P6", "serving reproduces p_pipeline within the tolerance, answers 'ok' on every row, builds every model input, returns finite numbers", lambda: verdict(
        not bad_p.any() and non_ok == 0 and warn_missing == 0 and np.isfinite(p).all(),
        f"{len(p):,} rows; statuses {counts}; max|serving - pipeline| {float(np.max(np.abs(p - ref))):.2e} (tolerance {tol:.1e}); rows over tolerance {int(bad_p.sum())}; "
        f"rows with a 'model input(s) not given' warning {warn_missing}; non-ok rows {non_ok}", n=len(p)))
    if ex_states:
        pe = np.array([o["p_ct_win"] for o in wp.predict_many(ex_states)])
        de = np.abs(pe - ex_rows["p_pipeline"].to_numpy().astype(np.float64))
        info["excluded"]["max_abs_diff"] = float(np.nanmax(de))
        info["excluded"]["rows_over_tolerance"] = int((~(de <= tol)).sum())
        print(f"INFO    excluded rows (reported separately, not in the verdict): {len(pe)} rows, max|serving - pipeline| {float(np.nanmax(de)):.2e}, over tolerance {int((~(de <= tol)).sum())}", flush=True)

    # ---- P7: predict() vs predict_many() on a varied, shuffled mixed batch
    def p7():
        rng = np.random.default_rng(a.seed)
        n_full = min(len(states), a.batch_size // 2)
        idx = rng.choice(len(states), size=n_full, replace=False)
        batch = [dict(states[i]) for i in idx]
        base = dict(states[idx[0]])
        no_table = [dict({k: v for k, v in base.items() if k != "players"}, alive_ct=int(rng.integers(0, 6)), alive_t=int(rng.integers(0, 6)),
                         seconds_since_round_live=float(rng.uniform(0, 140)), bomb_planted=int(rng.integers(0, 2)), seconds_since_plant=float(rng.uniform(0, 45)))
                    for _ in range(a.batch_size // 4)]
        odd = [dict(base, map_name=str(rng.choice(["de_vertigo", "de_overpass", "cs_office", ""])), seconds_since_round_live=float(rng.uniform(0, 100)))
               for _ in range(a.batch_size // 8)] + [dict(base, players="garbage"), dict(base, seconds_since_round_live="x"), "not a dict", {}]
        batch = batch + no_table + odd
        order = rng.permutation(len(batch))
        batch = [batch[i] for i in order]
        many = wp.predict_many(batch)
        diff, stat = 0.0, 0
        for s_, m_ in zip(batch, many):
            one = wp.predict(s_)
            diff = max(diff, abs(one["p_ct_win"] - m_["p_ct_win"]))
            stat += one["status"] != m_["status"]
        kinds = sorted({m_["status"] for m_ in many})
        return verdict(diff <= 1e-12 and stat == 0, f"{len(batch)} mixed states (statuses {kinds}); max|predict - predict_many| {diff:.1e}; status differences {stat}", n=len(batch))
    R.run("P7", "predict() equals predict_many() element by element on a shuffled mixed batch", p7)

    # ---- P8: the alive-count fallback
    fb_sample = a.fallback_sample
    if fb_sample is None and a.fallback:
        fb_sample = os.path.join(P.data_dir, f"export_{os.path.splitext(os.path.basename(a.fallback))[0]}_test_sample.parquet")
    if not a.fallback:
        R.add("P8", "fallback model reproduces its sample; no player table -> 'fallback' with that number", notrun("no --fallback model given"))
    elif not (os.path.exists(a.fallback) and fb_sample and os.path.exists(fb_sample)):
        R.add("P8", "fallback model reproduces its sample; no player table -> 'fallback' with that number", notrun(f"missing {a.fallback} or its sample {fb_sample}"))
    else:
        def p8():
            fs = pl.read_parquet(fb_sample)
            fs = fs.sample(n=min(len(fs), max(a.n, 200) if a.n > 0 else len(fs)), seed=a.seed)
            fwp = WinProb.load(a.fallback)
            full = WinProb.load(P.model_path, a.fallback)
            sts = [{"seconds_since_round_live": r["seconds_since_round_live"], "bomb_planted": r["bomb_planted"], "seconds_since_plant": r["seconds_since_plant"], "alive_ct": r["alive_ct"],
                    "alive_t": r["alive_t"], "map_name": r["map_name"], "mode": "competitive"} for r in fs.iter_rows(named=True)]
            ref_ = fs["p_pipeline"].to_numpy().astype(np.float64)
            o1, o2 = fwp.predict_many(sts), full.predict_many(sts)
            p1, p2_ = np.array([o["p_ct_win"] for o in o1]), np.array([o["p_ct_win"] for o in o2])
            # a row serving answers by rule ("terminal": the round is already decided) is a state the exporter should not contain; like the exclusions of P3 it is
            # reported separately and bounded by a share, never compared with the model's number
            term = np.array([o["status"] == "terminal" for o in o1])
            term2 = np.array([o["status"] == "terminal" for o in o2])
            keep = ~term & ~term2
            bad1, bad2 = int((~(np.abs(p1 - ref_) <= tol))[keep].sum()), int((~(np.abs(p2_ - ref_) <= tol))[keep].sum())
            s1 = {o["status"] for o, k in zip(o1, keep) if k}
            s2 = {o["status"] for o, k in zip(o2, keep) if k}
            share_t = float((term | term2).mean())
            ex = [(round(r["seconds_since_round_live"], 1), r["bomb_planted"], round(r["seconds_since_plant"], 1), r["alive_ct"], r["alive_t"], r["era"])
                  for r, t_ in zip(fs.iter_rows(named=True), term | term2) if t_][:3]
            msg = (f"{len(sts):,} rows of {os.path.basename(fb_sample)}; alive-count model alone: {bad1} rows over tolerance, statuses {sorted(s1)}; full model without a table: "
                   f"{bad2} rows over tolerance, statuses {sorted(s2)} (expected 'fallback'); rows serving answers by rule (terminal, outside the training domain) "
                   f"{int((term | term2).sum())} = {share_t:.2%} (bound {a.excluded_share_max:.2%}) e.g. (secs, planted, secs_since_plant, alive_ct, alive_t, era) {ex}")
            if bad1 or bad2 or s1 != {"ok"} or s2 != {"fallback"} or share_t > a.excluded_share_max:
                return fail(msg, n=int(keep.sum()))
            if share_t > 0:
                return warn(msg + " (the sample predates the exporter's domain rule: re-export)", n=int(keep.sum()))
            return ok(msg, n=int(keep.sum()))
        R.run("P8", "fallback model reproduces its sample; without a player table the full model answers 'fallback' with that number", p8)
    return "complete"


if __name__ == "__main__":
    run()
