"""T8: smoke-grenade charts from the 2026-10-01..08 tome.

(a) 2D scatter of smokegrenade_detonate positions for the most common map (clean 5v5 rounds).
(b) Mean smokes detonated per clean 5v5 round by round number 1-24, match-cluster bootstrap CI.
Also reports smoke lifetime: smokegrenade_expired minus smokegrenade_detonate, paired by entity_id.

Outputs: data/viz/T8.json and data/viz/T8.svg (two panels in one 800x450 SVG).
Pure polars + stdlib; SVG drawn with string templates. Read-only on the tome.
Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\viz\\T8.py
"""
import glob
import json
import math
import os
import random
import statistics
import warnings

import polars as pl

warnings.filterwarnings("ignore")

ROOT = r"G:\pureskill\csds-cs2"
TOME = os.path.join(ROOT, "tomes", "2026-10-01_to_08", "tome", "csds")
OUT_JSON = os.path.join(ROOT, "data", "viz", "T8.json")
OUT_SVG = os.path.join(ROOT, "data", "viz", "T8.svg")

N_BOOT = 1000
SEED = 0
CLEAN_PLAYERS = 10
BAR_ROUNDS = list(range(1, 25))
PISTOL_ROUNDS = (1, 13)
TICK_RATE = 64
CT_C, T_C, NEU, GRID, ACC, INK = "#2563eb", "#d97706", "#475569", "#e2e8f0", "#059669", "#0f172a"


def files(sub):
    found = sorted(glob.glob(os.path.join(TOME, sub, "dataframe_*")))
    if not found:
        raise FileNotFoundError(sub)
    return found


def read(sub, cols):
    return pl.scan_parquet(files(sub)).select(cols).collect()


def pct(vals, q):
    v = sorted(vals)
    pos = (len(v) - 1) * q
    lo_i, hi_i = int(math.floor(pos)), int(math.ceil(pos))
    return v[lo_i] + (v[hi_i] - v[lo_i]) * (pos - lo_i)


def boot_sums(wide, n_units):
    """Match-cluster bootstrap: resample match rows with replacement, return column sums per resample."""
    rng = random.Random(SEED)
    cols = wide.columns
    out = []
    for _ in range(N_BOOT):
        idx = pl.Series("idx", [rng.randrange(n_units) for _ in range(n_units)], dtype=pl.Int64)
        row = wide.select([pl.col(c).gather(idx).sum().alias(c) for c in cols]).row(0)
        out.append(dict(zip(cols, row)))
    return out


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=12, anchor="start", fill="#334155", rot=None, weight=None):
    attrs = f'x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" fill="{fill}"'
    if weight:
        attrs += f' font-weight="{weight}"'
    if rot is not None:
        attrs += f' transform="rotate({rot} {x:.1f} {y:.1f})"'
    return f"<text {attrs}>{esc(s)}</text>"


def line(x1, y1, x2, y2, stroke, w=1.0):
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{w}"/>'


def main():
    # ---------- header: match keys, maps, round caps
    hdr = read("header", ["key", "map_name", "max_rounds", "tick_rate"])
    assert hdr["key"].n_unique() == hdr.height
    assert set(hdr["tick_rate"].to_list()) == {TICK_RATE}
    keys = hdr["key"].to_list()
    kidx = {k: i for i, k in enumerate(keys)}
    M = len(keys)
    max_rounds = hdr["max_rounds"].to_list()

    # ---------- round_end: clean 5v5 rounds
    rend = read("basic_round_end*", ["match_key", "round", "tick", "player_count"]).join(
        hdr.select([pl.col("key").alias("match_key"), "map_name"]), on="match_key", how="left"
    )
    assert rend.height == rend.select(["match_key", "round"]).unique().height
    clean = rend.filter(pl.col("player_count") == CLEAN_PLAYERS)

    # ---------- grenade_state: smoke detonations and expiries (projected, filtered lazily)
    gs = (
        pl.scan_parquet(files("basic_grenade_state*"))
        .filter(pl.col("event_type").is_in(["smokegrenade_detonate", "smokegrenade_expired"]))
        .select(
            ["match_key", "round", "tick", "event_type", "entity_id", "x_pos", "y_pos", "player_team_code", "tick_throw"]
        )
        .collect()
    )
    det = (
        gs.filter(pl.col("event_type") == "smokegrenade_detonate")
        .drop("event_type")
        .rename({"tick": "det_tick", "x_pos": "det_x", "y_pos": "det_y"})
    )
    exp = (
        gs.filter(pl.col("event_type") == "smokegrenade_expired")
        .select(["match_key", "round", "tick", "entity_id"])
        .rename({"tick": "exp_tick"})
    )
    n_det_all = det.height
    n_exact_dup = n_det_all - det.unique(subset=["match_key", "round", "det_tick", "det_x", "det_y"]).height
    det = det.join(
        rend.select(["match_key", "round", "tick", "player_count", "map_name"]).rename({"tick": "re_tick"}),
        on=["match_key", "round"],
        how="left",
    )
    assert det["re_tick"].null_count() == 0
    n_det_clean = det.filter(pl.col("player_count") == CLEAN_PLAYERS).height

    # ---------- smokes per clean round (zero-filled so empty rounds count in the denominator)
    per_round = det.group_by(["match_key", "round"]).agg(pl.len().alias("n_smokes"))
    clean_r = clean.join(per_round, on=["match_key", "round"], how="left").with_columns(
        pl.col("n_smokes").fill_null(0)
    )

    # ---------- (b) bar chart: round 1-24 means with match-cluster bootstrap
    bar = clean_r.filter(pl.col("round").is_between(1, 24))
    S = {r: [0] * M for r in BAR_ROUNDS}
    C = {r: [0] * M for r in BAR_ROUNDS}
    for mk, r, n in bar.select(["match_key", "round", "n_smokes"]).iter_rows():
        S[r][kidx[mk]] = int(n)
        C[r][kidx[mk]] = 1
    wide = pl.DataFrame({**{f"s{r}": S[r] for r in BAR_ROUNDS}, **{f"c{r}": C[r] for r in BAR_ROUNDS}})
    bs = boot_sums(wide, M)
    bar_out = {"round": BAR_ROUNDS, "clean_rounds": [], "smokes": [], "mean": [], "ci95_lo": [], "ci95_hi": []}
    for r in BAR_ROUNDS:
        sS, sC = sum(S[r]), sum(C[r])
        vals = [b[f"s{r}"] / b[f"c{r}"] for b in bs if b[f"c{r}"] > 0]
        bar_out["clean_rounds"].append(sC)
        bar_out["smokes"].append(sS)
        bar_out["mean"].append(sS / sC)
        bar_out["ci95_lo"].append(pct(vals, 0.025))
        bar_out["ci95_hi"].append(pct(vals, 0.975))
    for k in range(24):
        assert bar_out["clean_rounds"][k] > 0
        assert bar_out["smokes"][k] >= 0
        assert bar_out["ci95_lo"][k] <= bar_out["mean"][k] <= bar_out["ci95_hi"][k], BAR_ROUNDS[k]
    tot_rounds = sum(bar_out["clean_rounds"])
    tot_smokes = sum(bar_out["smokes"])
    n_bar_matches = sum(1 for i in range(M) if any(C[r][i] for r in BAR_ROUNDS))

    # sensitivity: drop the header max_rounds==16 matches (short or MR8-style games)
    keep = [i for i in range(M) if max_rounds[i] != 16]
    sens_mean = []
    for r in BAR_ROUNDS:
        sS = sum(S[r][i] for i in keep)
        sC = sum(C[r][i] for i in keep)
        sens_mean.append(sS / sC)
    sens_max_abs = max(abs(a - b) for a, b in zip(sens_mean, bar_out["mean"]))

    # ---------- (a) scatter: most common map by matches
    top = hdr.group_by("map_name").len().sort(["len", "map_name"], descending=[True, False])
    top_map = top["map_name"][0]
    top_map_matches = int(top["len"][0])
    sc = det.filter((pl.col("player_count") == CLEAN_PLAYERS) & (pl.col("map_name") == top_map))
    n_map_null_xy = sc.filter(pl.col("det_x").is_null() | pl.col("det_y").is_null()).height
    pts = []
    for x, y, t in sc.select(["det_x", "det_y", "player_team_code"]).iter_rows():
        if x is None or y is None:
            continue
        pts.append((round(float(x), 1), round(float(y), 1), {3: "CT", 2: "T"}.get(t, "unk")))
    n_map = len(pts)
    n_map_rounds = clean.filter(pl.col("map_name") == top_map).height
    n_map_matches_clean = clean.filter(pl.col("map_name") == top_map)["match_key"].n_unique()
    team_n = {"CT": 0, "T": 0, "unk": 0}
    for _, _, t in pts:
        team_n[t] += 1
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    lo_v = min(min(xs), min(ys))
    hi_v = max(max(xs), max(ys))
    pad = 500
    x0 = int(math.floor(lo_v / pad) * pad)
    x1 = int(math.ceil(hi_v / pad) * pad)
    dom = x1 - x0
    tstep = next(s for s in (500, 1000, 2000) if dom / s <= 7)
    assert all(x0 <= v <= x1 for v in xs + ys)

    # ---------- lifetime: expiry paired to detonation by (match, round, entity_id), backward as-of
    e = exp.sort("exp_tick")
    d = det.select(["match_key", "round", "det_tick", "entity_id"]).sort("det_tick")
    pair = e.join_asof(
        d, left_on="exp_tick", right_on="det_tick", by=["match_key", "round", "entity_id"], strategy="backward"
    )
    paired = pair.filter(pl.col("det_tick").is_not_null()).with_columns(
        (pl.col("exp_tick") - pl.col("det_tick")).alias("life_ticks")
    )
    dup_pairs = paired.height - paired.select(["match_key", "round", "det_tick", "entity_id"]).unique().height
    lt = paired["life_ticks"].to_list()
    life_stats = {
        "n_expired_events": exp.height,
        "n_paired": paired.height,
        "n_detonations_all": n_det_all,
        "n_detonations_unpaired": n_det_all - (paired.height - dup_pairs),
        "duplicate_pairings": dup_pairs,
        "lifetime_ticks_unique_values": sorted(set(lt)),
        "mean_s": statistics.mean(lt) / TICK_RATE,
        "median_s": pct(lt, 0.5) / TICK_RATE,
        "p25_s": pct(lt, 0.25) / TICK_RATE,
        "p75_s": pct(lt, 0.75) / TICK_RATE,
        "min_s": min(lt) / TICK_RATE,
        "max_s": max(lt) / TICK_RATE,
        "sd_ticks": statistics.pstdev(lt),
    }
    lsum = [0] * M
    lcnt = [0] * M
    for mk, l_ in paired.select(["match_key", "life_ticks"]).iter_rows():
        lsum[kidx[mk]] += int(l_)
        lcnt[kidx[mk]] += 1
    lb = boot_sums(pl.DataFrame({"ls": lsum, "lc": lcnt}), M)
    lmeans = [b["ls"] / b["lc"] / TICK_RATE for b in lb if b["lc"] > 0]
    life_stats["mean_ci95_match_bootstrap_s"] = [pct(lmeans, 0.025), pct(lmeans, 0.975)]
    life_max = max(lt)
    paired_keys = paired.select(["match_key", "round", "det_tick", "entity_id"]).unique()
    unp = det.join(paired_keys, on=["match_key", "round", "det_tick", "entity_id"], how="anti")
    life_stats["unpaired_detonations_cut_by_round_end"] = unp.filter(
        pl.col("det_tick") + life_max > pl.col("re_tick")
    ).height
    life_stats["unpaired_detonations_total"] = unp.height

    # ---------- data checks on detonation timing and throw ticks
    late = det.filter(pl.col("det_tick") > pl.col("re_tick"))
    flight = det.with_columns((pl.col("det_tick") - pl.col("tick_throw")).alias("flight"))
    checks = {
        "detonation_rows_all_rounds": n_det_all,
        "detonation_rows_clean_5v5": n_det_clean,
        "exact_duplicate_detonations_removed": 0 if n_exact_dup == 0 else n_exact_dup,
        "detonations_after_round_end": late.height,
        "detonations_after_round_end_in_clean_rounds": late.filter(pl.col("player_count") == CLEAN_PLAYERS).height,
        "detonations_after_round_end_thrown_before_end": late.filter(pl.col("tick_throw") <= pl.col("re_tick")).height,
        "median_detonation_minus_tick_throw_ticks": flight["flight"].median(),
        "detonation_minus_tick_throw_over_60s": flight.filter(pl.col("flight") > 60 * TICK_RATE).height,
        "detonations_null_xy": n_map_null_xy,
        "detonations_with_null_team": det.filter(pl.col("player_team_code").is_null()).height,
        "matches_max_rounds_16": M - len(keep),
        "matches_total": M,
    }

    # ---------- SVG: panel (a) scatter, panel (b) bars
    W, H = 800, 450
    parts = [
        f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
    ]

    # panel a: square scatter, shared axis domain so the aspect is 1:1
    L1, T1, S1 = 72, 40, 270
    B1 = T1 + S1

    def sx(x):
        return L1 + (x - x0) / dom * S1

    def sy(y):
        return B1 - (y - x0) / dom * S1

    ticks = list(range(int(math.ceil(x0 / tstep) * tstep), x1 + 1, tstep))
    for t in ticks:
        parts.append(line(sx(t), T1, sx(t), B1, GRID, 1))
        parts.append(line(L1, sy(t), L1 + S1, sy(t), GRID, 1))
    parts.append(line(L1, T1, L1, B1, NEU, 1))
    parts.append(line(L1, B1, L1 + S1, B1, NEU, 1))
    for t in ticks:
        parts.append(text(sx(t), B1 + 15, f"{t}", 12, "middle"))
        parts.append(text(L1 - 6, sy(t) + 4, f"{t}", 12, "end"))
    for team, col in (("CT", CT_C), ("T", T_C), ("unk", NEU)):
        seg = "".join(f"M{sx(x):.1f} {sy(y):.1f}h0.01" for x, y, t in pts if t == team)
        if seg:
            parts.append(
                f'<path d="{seg}" fill="none" stroke="{col}" stroke-width="2.2" '
                f'stroke-linecap="round" stroke-opacity="0.22"/>'
            )
    parts.append(text(L1 + S1 / 2, B1 + 36, "x (game units)", 13, "middle", INK))
    parts.append(text(20, T1 + S1 / 2, "y (game units)", 13, "middle", INK, rot=-90))
    parts.append(text(L1, 16, f"(a) {top_map}, n = {n_map:,} smoke detonations", 12, "start", INK))
    parts.append(
        text(L1, 30, f"{n_map_rounds:,} clean 5v5 rounds, {n_map_matches_clean} matches", 12, "start", "#475569")
    )
    legend_a = [("CT thrower", CT_C, team_n["CT"]), ("T thrower", T_C, team_n["T"]), ("Side unknown", NEU, team_n["unk"])]
    for k, (lab, col, n) in enumerate(legend_a):
        yy = B1 + 56 + k * 16
        parts.append(f'<circle cx="{L1 + 5}" cy="{yy - 4}" r="4" fill="{col}" fill-opacity="0.8"/>')
        parts.append(text(L1 + 16, yy, f"{lab}: n = {n:,}", 12, "start", "#334155"))

    # panel b: bars with match-cluster bootstrap 95% CI
    L2, T2, W2, H2 = 444, 40, 336, 270
    B2 = T2 + H2
    ymax = math.ceil(max(bar_out["ci95_hi"]) / 0.5) * 0.5
    ystep = 0.5 if ymax <= 3 else 1.0
    yticks = [round(k * ystep, 2) for k in range(int(round(ymax / ystep)) + 1)]

    def by(v):
        return B2 - v / ymax * H2

    band = W2 / 24.0
    bw = band * 0.72
    for t in yticks:
        parts.append(line(L2, by(t), L2 + W2, by(t), GRID, 1))
        parts.append(text(L2 - 6, by(t) + 4, f"{t:.1f}", 12, "end"))
    parts.append(line(L2, T2, L2, B2, NEU, 1))
    parts.append(line(L2, B2, L2 + W2, B2, NEU, 1))
    for k, r in enumerate(BAR_ROUNDS):
        cx = L2 + (k + 0.5) * band
        m = bar_out["mean"][k]
        lo = bar_out["ci95_lo"][k]
        hi = bar_out["ci95_hi"][k]
        col = ACC if r in PISTOL_ROUNDS else NEU
        parts.append(
            f'<rect x="{cx - bw / 2:.1f}" y="{by(m):.1f}" width="{bw:.1f}" height="{B2 - by(m):.1f}" '
            f'fill="{col}" fill-opacity="0.9"/>'
        )
        parts.append(line(cx, by(lo), cx, by(hi), INK, 1.2))
        parts.append(line(cx - 3, by(lo), cx + 3, by(lo), INK, 1.2))
        parts.append(line(cx - 3, by(hi), cx + 3, by(hi), INK, 1.2))
        parts.append(text(cx, B2 + 15, str(r), 12, "middle"))
    parts.append(text(L2 + W2 / 2, B2 + 36, "Round number", 13, "middle", INK))
    parts.append(text(398, T2 + H2 / 2, "Smokes per clean 5v5 round", 13, "middle", INK, rot=-90))
    parts.append(text(L2, 16, f"(b) n = {tot_rounds:,} clean 5v5 rounds, rounds 1-24", 12, "start", INK))
    parts.append(
        text(L2, 30, f"{n_bar_matches:,} matches, {tot_smokes:,} smoke detonations", 12, "start", "#475569")
    )
    parts.append(f'<rect x="{L2}" y="{B2 + 52}" width="10" height="10" fill="{NEU}"/>')
    parts.append(text(L2 + 16, B2 + 61, "Mean per round, 95% CI (match bootstrap)", 12, "start", "#334155"))
    parts.append(f'<rect x="{L2}" y="{B2 + 68}" width="10" height="10" fill="{ACC}"/>')
    parts.append(text(L2 + 16, B2 + 77, "Pistol round (rounds 1 and 13)", 12, "start", "#334155"))
    parts.append("</svg>")
    svg = "\n".join(parts) + "\n"
    with open(OUT_SVG, "w", encoding="utf-8", newline="\n") as f:
        f.write(svg)

    # ---------- JSON: plotted series and key numbers
    out = {
        "id": "T8",
        "title": "Smokes: detonation positions and smokes per round",
        "window": "2026-10-01 to 2026-10-08 tome, 1,752 matches",
        "definitions": {
            "smoke_thrown_proxy": "count of grenade_state smokegrenade_detonate rows (one per smoke; no throw event used)",
            "clean_5v5_round": "round_end.player_count == 10",
            "side_codes": "player_team_code 2 = T, 3 = CT (thrower side at detonation)",
            "lifetime": "smokegrenade_expired tick minus smokegrenade_detonate tick, paired by (match_key, round, entity_id) with a backward as-of join",
            "tick_rate": TICK_RATE,
        },
        "bootstrap": {"unit": "match", "resamples": N_BOOT, "seed": SEED, "interval": "percentile 2.5 to 97.5"},
        "data_checks": checks,
        "bar": {
            **bar_out,
            "pistol_rounds": list(PISTOL_ROUNDS),
            "matches_with_clean_rounds_1_24": n_bar_matches,
            "total_clean_rounds_1_24": tot_rounds,
            "total_smokes_1_24": tot_smokes,
            "overall_mean_per_round": tot_smokes / tot_rounds,
            "sensitivity_excl_max_rounds_16": {
                "matches_excluded": M - len(keep),
                "mean": sens_mean,
                "max_abs_diff_vs_main": sens_max_abs,
            },
        },
        "scatter": {
            "map_name": top_map,
            "map_matches_header": top_map_matches,
            "map_matches_with_clean_rounds": n_map_matches_clean,
            "map_clean_rounds": n_map_rounds,
            "n_detonations_plotted": n_map,
            "n_by_side": team_n,
            "domain": [x0, x1],
            "x_min": min(xs),
            "x_max": max(xs),
            "y_min": min(ys),
            "y_max": max(ys),
            "points_xy_side": pts,
        },
        "lifetime": life_stats,
        "caveats": [
            "lifetime is identical (1412 ticks = 22.06 s) for every paired expiry, which points to a fixed expiry timer in the parser output rather than a measured dissipation",
            "smokes thrown is proxied by detonation rows; flights that never detonate are not counted",
            "detonations after round_end are kept (assigned to the round by grenade_state)",
            "tick_throw is unreliable (some detonation minus throw gaps exceed 60 s) and is not used for counting",
            "header max_rounds is 16 for 149 matches; their rounds have only 2-4 players (no clean 5v5 rounds), so they add nothing and the sensitivity difference is 0.0 by construction",
            "rounds 19-24 have far fewer clean rounds (1,072 down to 284) because only longer matches reach them; those means may be selected and carry wider CIs",
            "round-number means are pooled over both sides and both halves, so side-swap effects are not separated",
            "entity_id is reused within a round for 21 detonation pairs; these are distinct smokes at different ticks and positions, so every detonation row is counted",
            "scatter is in CS2 world coordinates with no radar overlay; axes are not oriented to the radar image",
        ],
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    # ---------- console summary for verification
    print("matches", M, "map", top_map, top_map_matches)
    print("bar tot_rounds", tot_rounds, "tot_smokes", tot_smokes, "matches_with_clean", n_bar_matches)
    for k, r in enumerate(BAR_ROUNDS):
        print(
            f"r{r:>2} rounds={bar_out['clean_rounds'][k]:>5} smokes={bar_out['smokes'][k]:>5} "
            f"mean={bar_out['mean'][k]:.4f} ci=[{bar_out['ci95_lo'][k]:.4f},{bar_out['ci95_hi'][k]:.4f}]"
        )
    print("sens max abs diff", round(sens_max_abs, 4), "excluded", M - len(keep))
    print("scatter n", n_map, "sides", team_n, "domain", x0, x1, "xrange", min(xs), max(xs), "yrange", min(ys), max(ys))
    print("scatter clean rounds on map", n_map_rounds, "matches clean", n_map_matches_clean)
    print("life", {k: v for k, v in life_stats.items() if k != "lifetime_ticks_unique_values"})
    print("checks", checks)
    print("svg bytes", os.path.getsize(OUT_SVG), "json bytes", os.path.getsize(OUT_JSON))


if __name__ == "__main__":
    main()
