"""T6: first kill. Histogram of first-kill time and round win rate of the first-kill side.

Source: 2026-10 tomes (1,752 matches), clean 5v5 rounds (player_count == 10).
Live start = round_freeze_end tick (validated against rounds.parquet live_start_tick
for all 2,673 rounds of the 1 Hz sample). Writes data/viz/T6.json and data/viz/T6.svg.
Uses polars, numpy and pyarrow (schema checks) only. No network.
"""
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
OUT_DIR = ROOT / "data" / "viz"
OUT_JSON = OUT_DIR / "T6.json"
OUT_SVG = OUT_DIR / "T6.svg"

N_BOOT = 1000
SEED = 0
BIN_W = 3.0
BIN_MAX = 120.0

CT_COL = "#2563eb"
T_COL = "#d97706"
NEUTRAL = "#475569"
GRID = "#e2e8f0"
KEYS = ["match_key", "round"]


def channel_file(prefix):
    folders = sorted(p for p in TOME.glob(prefix + ".*") if p.is_dir())
    if len(folders) != 1:
        raise RuntimeError(f"expected exactly one folder for {prefix}, found {len(folders)}")
    f = folders[0] / "dataframe_00000"
    if not f.exists():
        raise RuntimeError(f"missing {f}")
    return f


def load(f, cols):
    names = set(pq.read_schema(str(f)).names)
    missing = [c for c in cols if c not in names]
    if missing:
        raise RuntimeError(f"{f} is missing columns {missing}")
    return pl.scan_parquet(str(f)).select(cols).collect()


def nice_step(span, target=5):
    raw = span / target
    mag = 10.0 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if m * mag >= raw - 1e-12:
            return m * mag
    return 10 * mag


def pct(x):
    return f"{round(float(x) * 100, 4):g}%"


def compute():
    checks = {}

    # Round table: all rounds, then clean 5v5 (player_count == 10).
    rnd = load(channel_file("basic_round_end"),
               ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"])
    rnd = rnd.rename({"tick": "end_tick"})
    checks["round_end_rows_all"] = rnd.height
    checks["round_end_duplicate_keys"] = rnd.height - rnd.select(KEYS).unique().height
    checks["matches_in_tome"] = rnd["match_key"].n_unique()

    hdr = load(TOME / "header" / "dataframe_00000", ["key", "tick_rate", "max_rounds", "is_wingman"])
    hdr = hdr.rename({"key": "match_key"})
    checks["header_wingman_matches"] = int(hdr["is_wingman"].sum())
    checks["header_tick_rate_counts"] = {str(k): v for k, v in Counter(hdr["tick_rate"].to_list()).items()}
    checks["header_max_rounds_counts"] = {str(k): v for k, v in Counter(hdr["max_rounds"].to_list()).items()}

    clean = rnd.filter(pl.col("player_count") == 10).join(hdr, on="match_key", how="left")
    checks["clean_rounds"] = clean.height
    checks["clean_rounds_missing_header"] = clean.filter(pl.col("tick_rate").is_null()).height
    checks["clean_matches"] = clean["match_key"].n_unique()

    # Live start = round_freeze_end.
    rs = load(channel_file("basic_round_state"), ["match_key", "round", "tick", "event_type"])
    fe = (rs.filter(pl.col("event_type") == "round_freeze_end")
            .select(KEYS + [pl.col("tick").alias("live_tick")]))
    checks["freeze_end_rows"] = fe.height
    checks["freeze_end_duplicate_keys"] = fe.height - fe.select(KEYS).unique().height
    clean = clean.join(fe, on=KEYS, how="left")
    checks["clean_rounds_without_live_start"] = clean.filter(pl.col("live_tick").is_null()).height
    clean_live = clean.filter(pl.col("live_tick").is_not_null())
    checks["clean_rounds_with_live_start"] = clean_live.height

    base = clean_live.with_columns([
        (pl.col("winner_team_code") == 3).alias("ct_win"),
        (pl.col("round") > pl.col("max_rounds")).alias("is_overtime"),
    ]).select(["match_key", "round", "live_tick", "end_tick", "tick_rate", "max_rounds",
               "ct_win", "is_overtime", "winner_team_code", "win_reason_code"])

    # Deaths within clean live rounds.
    dt = load(channel_file("basic_player_death"),
              ["match_key", "round", "tick", "player_id", "attacker_id",
               "player_team_code", "attacker_team_code"])
    checks["death_rows_all"] = dt.height
    dj = dt.join(base.select(["match_key", "round", "live_tick", "end_tick", "tick_rate"]),
                 on=KEYS, how="inner")
    checks["death_rows_clean_live_rounds"] = dj.height
    checks["deaths_before_live_start"] = dj.filter(pl.col("tick") < pl.col("live_tick")).height
    checks["deaths_after_round_end"] = dj.filter(pl.col("tick") > pl.col("end_tick")).height
    dj = dj.filter((pl.col("tick") >= pl.col("live_tick")) & (pl.col("tick") <= pl.col("end_tick")))

    # Team codes: 2 = T, 3 = CT. Anything else is treated as unknown.
    dj = dj.with_columns([
        pl.when(pl.col("player_team_code").is_in([2, 3])).then(pl.col("player_team_code"))
          .otherwise(None).alias("vic_team"),
        pl.when(pl.col("attacker_team_code").is_in([2, 3])).then(pl.col("attacker_team_code"))
          .otherwise(None).alias("atk_team"),
    ])
    dj = dj.with_columns(
        pl.when(pl.col("vic_team").is_null()).then(pl.lit("victim_unknown"))
          .when(pl.col("atk_team").is_null()).then(pl.lit("attacker_unknown"))
          .when(pl.col("atk_team") == pl.col("vic_team")).then(pl.lit("same_team"))
          .otherwise(pl.lit("enemy")).alias("kind")
    )
    checks["death_kind_counts"] = {str(k): int(v) for k, v in dj.group_by("kind").len().rows()}

    # First enemy kill per round.
    enemy = dj.filter(pl.col("kind") == "enemy")
    first_tick = enemy.group_by(KEYS).agg(pl.col("tick").min().alias("first_tick"))
    first_side = (enemy.join(first_tick, on=KEYS)
                  .filter(pl.col("tick") == pl.col("first_tick"))
                  .group_by(KEYS)
                  .agg([pl.col("atk_team").n_unique().alias("n_sides"),
                        pl.col("atk_team").min().alias("first_side")]))
    amb = (dj.filter(pl.col("kind").is_in(["victim_unknown", "attacker_unknown"]))
           .group_by(KEYS).agg(pl.col("tick").min().alias("amb_tick")))
    rounds = (base.join(first_tick, on=KEYS, how="left")
                  .join(first_side, on=KEYS, how="left")
                  .join(amb, on=KEYS, how="left"))
    rounds = rounds.with_columns([
        ((pl.col("first_tick") - pl.col("live_tick")) / pl.col("tick_rate")).alias("first_kill_s"),
        pl.when(pl.col("first_tick").is_null()).then(pl.lit("no_enemy_kill"))
          .when(pl.col("amb_tick").is_not_null() & (pl.col("amb_tick") < pl.col("first_tick")))
          .then(pl.lit("ambiguous_before_first_kill"))
          .when(pl.col("n_sides") > 1).then(pl.lit("tie_opposite_sides"))
          .when(pl.col("first_side") == 3).then(pl.lit("ct_first"))
          .when(pl.col("first_side") == 2).then(pl.lit("t_first"))
          .otherwise(pl.lit("unclassified")).alias("status"),
    ])
    status_counts = {str(k): int(v) for k, v in rounds.group_by("status").len().rows()}
    checks["status_counts"] = status_counts
    checks["status_sum_equals_clean_live"] = sum(status_counts.values()) == clean_live.height
    checks["unclassified_rounds"] = status_counts.get("unclassified", 0)

    an = rounds.filter(pl.col("status").is_in(["ct_first", "t_first"]))
    ct_an = an.filter(pl.col("status") == "ct_first")
    t_an = an.filter(pl.col("status") == "t_first")

    # Match-level aggregates for the cluster bootstrap. Clusters = clean 5v5 matches.
    universe = sorted(clean["match_key"].unique().to_list())
    M = len(universe)
    midx = {k: i for i, k in enumerate(universe)}
    agg = np.zeros((M, 4), dtype=np.float64)  # n_ct, w_ct, n_t, w_t
    g_ct = ct_an.group_by("match_key").agg([pl.len().alias("n"),
                                            pl.col("ct_win").cast(pl.Int64).sum().alias("w")])
    g_t = t_an.group_by("match_key").agg([pl.len().alias("n"),
                                          (~pl.col("ct_win")).cast(pl.Int64).sum().alias("w")])
    for mk, n, w in g_ct.rows():
        agg[midx[mk], 0] = n
        agg[midx[mk], 1] = w
    for mk, n, w in g_t.rows():
        agg[midx[mk], 2] = n
        agg[midx[mk], 3] = w

    n_ct = float(agg[:, 0].sum())
    w_ct = float(agg[:, 1].sum())
    n_t = float(agg[:, 2].sum())
    w_t = float(agg[:, 3].sum())
    assert n_ct == ct_an.height and n_t == t_an.height, "match aggregates disagree with round counts"
    ct_rate = w_ct / n_ct
    t_rate = w_t / n_t

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, M, size=(N_BOOT, M))
    sums = agg[idx].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ct_b = sums[:, 1] / sums[:, 0]
        t_b = sums[:, 3] / sums[:, 2]
    ct_ci = [float(v) for v in np.nanpercentile(ct_b, [2.5, 97.5])]
    t_ci = [float(v) for v in np.nanpercentile(t_b, [2.5, 97.5])]
    boot_nan = {"ct": int(np.isnan(ct_b).sum()), "t": int(np.isnan(t_b).sum())}

    # Histogram of first-kill time (seconds from live start), by side.
    secs_ct = ct_an["first_kill_s"].to_numpy().astype(float)
    secs_t = t_an["first_kill_s"].to_numpy().astype(float)
    secs_all = np.concatenate([secs_ct, secs_t])
    edges = np.arange(0.0, BIN_MAX + BIN_W / 2, BIN_W)
    ct_hist, _ = np.histogram(secs_ct, bins=edges)
    t_hist, _ = np.histogram(secs_t, bins=edges)
    overflow = int((secs_all > BIN_MAX).sum())
    below = int((secs_all < 0).sum())
    assert int(ct_hist.sum()) + overflow * 0 == len(secs_ct) - int((secs_ct > BIN_MAX).sum())

    # Sensitivity: MR12 matches only (max_rounds == 24).
    mr12 = an.filter(pl.col("max_rounds") == 24)
    mr12_ct = mr12.filter(pl.col("status") == "ct_first")
    mr12_t = mr12.filter(pl.col("status") == "t_first")
    sens = {
        "n_ct_first": mr12_ct.height,
        "n_t_first": mr12_t.height,
        "ct_first_ct_win_rate": float(mr12_ct["ct_win"].cast(pl.Float64).mean()),
        "t_first_t_win_rate": float(1.0 - mr12_t["ct_win"].cast(pl.Float64).mean()),
    }

    def med_q(v):
        if v.size == 0:
            return None
        return {"median": float(np.median(v)), "p25": float(np.percentile(v, 25)),
                "p75": float(np.percentile(v, 75))}

    denominators = {
        "matches_in_tome": checks["matches_in_tome"],
        "round_end_rows_all": checks["round_end_rows_all"],
        "clean_5v5_rounds": checks["clean_rounds"],
        "clean_5v5_matches": checks["clean_matches"],
        "clean_rounds_without_live_start_event": checks["clean_rounds_without_live_start"],
        "clean_rounds_with_live_start": clean_live.height,
        "no_enemy_kill_after_live_start": status_counts.get("no_enemy_kill", 0),
        "excluded_ambiguous_team_data_before_first_kill": status_counts.get("ambiguous_before_first_kill", 0),
        "excluded_tie_opposite_sides_same_tick": status_counts.get("tie_opposite_sides", 0),
        "analysis_rounds_first_enemy_kill": an.height,
        "analysis_ct_first_kill_rounds": ct_an.height,
        "analysis_t_first_kill_rounds": t_an.height,
        "analysis_matches_with_first_kill": an["match_key"].n_unique(),
        "analysis_ct_first_matches": ct_an["match_key"].n_unique(),
        "analysis_t_first_matches": t_an["match_key"].n_unique(),
        "bootstrap_cluster_universe_matches": M,
        "analysis_overtime_rounds": int(an.filter(pl.col("is_overtime")).height),
        "analysis_rounds_in_max_rounds_16_matches": int(an.filter(pl.col("max_rounds") != 24).height),
        "death_rows_clean_live_rounds": checks["death_rows_clean_live_rounds"],
        "death_kind_counts": checks["death_kind_counts"],
    }

    result = {
        "id": "T6",
        "title": "First kill: timing and win rate of the first-kill side",
        "source": {
            "tomes": "tomes/2026-10-01_to_08/tome/csds (basic_round_end, basic_round_state, basic_player_death, header)",
            "unit": "round (clean 5v5 = player_count == 10)",
            "tick_rate": "64 (header; checked, all matches)",
        },
        "definitions": {
            "live_start": "tick of round_freeze_end (equals rounds.parquet live_start_tick for all 2,673 sample rounds)",
            "first_kill": "earliest death after live start, up to round_end tick, with victim team and attacker team both known (2 or 3) and attacker team != victim team",
            "first_kill_seconds": "(first kill tick - live start tick) / tick_rate",
            "first_kill_side": "attacker team of the first kill (2 = T, 3 = CT)",
            "win": "round winner_team_code == first-kill side (winner 3 = CT, 2 = T)",
            "excluded": "same-team kills and suicides, world deaths, rounds with unknown victim or attacker team before the first enemy kill, same-tick kills by opposite sides",
        },
        "denominators": denominators,
        "first_kill_time": {
            "bin_width_s": BIN_W,
            "bin_edges_s": [float(e) for e in edges],
            "ct_first_counts": [int(c) for c in ct_hist],
            "t_first_counts": [int(t) for t in t_hist],
            "n_total": int(len(secs_all)),
            "n_ct_first": int(len(secs_ct)),
            "n_t_first": int(len(secs_t)),
            "overflow_above_120s": overflow,
            "negative_times": below,
            "min_s": float(secs_all.min()) if secs_all.size else None,
            "max_s": float(secs_all.max()) if secs_all.size else None,
            "all": med_q(secs_all),
            "ct_first": med_q(secs_ct),
            "t_first": med_q(secs_t),
        },
        "win_rate": {
            "ct_first_kill": {
                "n_rounds": int(n_ct), "n_matches": int(ct_an["match_key"].n_unique()),
                "win_rate": ct_rate, "ci95_match_cluster": ct_ci,
                "metric": "CT round win rate given CT got the first kill",
            },
            "t_first_kill": {
                "n_rounds": int(n_t), "n_matches": int(t_an["match_key"].n_unique()),
                "win_rate": t_rate, "ci95_match_cluster": t_ci,
                "metric": "T round win rate given T got the first kill",
            },
        },
        "bootstrap": {
            "method": "match-cluster bootstrap: resample clean 5v5 matches with replacement",
            "n_resamples": N_BOOT,
            "seed": SEED,
            "rng": "numpy.random.default_rng(seed).integers",
            "interval": "percentile 2.5 and 97.5 of the resampled pooled rates",
            "resamples_with_empty_group": boot_nan,
        },
        "sensitivity_mr12_only_point_estimates": sens,
        "checks": checks,
    }
    result["checks"]["ct_win_rate_ci_ordered"] = ct_ci[0] <= ct_rate <= ct_ci[1]
    result["checks"]["t_win_rate_ci_ordered"] = t_ci[0] <= t_rate <= t_ci[1]
    result["checks"]["histogram_sums_match_counts"] = (
        int(ct_hist.sum()) == int((secs_ct <= BIN_MAX).sum()) and
        int(t_hist.sum()) == int((secs_t <= BIN_MAX).sum()))
    result["checks"]["no_negative_counts"] = bool((ct_hist >= 0).all() and (t_hist >= 0).all())
    result["caveats"] = [
        "Source is the 2026-10 tomes (1,752 matches), not the 150-match 1 Hz sample. The first-kill search uses death events. The sample was used only to validate live start.",
        "Live start = round_freeze_end tick. It equals rounds.parquet live_start_tick for all 2,673 sample rounds. freezetime_ended_inferred is one tick later in most rounds and is not used.",
        f"The {checks['header_wingman_matches']} max_rounds = 16 matches are wingman (2v2) matches with no clean 5v5 rounds, so they drop out. 11 other matches (max player count 2, 9 or 11) have no player_count = 10 round.",
        f"Clean 5v5 includes overtime ({denominators['analysis_overtime_rounds']} analysis rounds) and pistol rounds (1 and 13). No stratification was done.",
        "First kill excludes same-team kills, suicides and world deaths. Every world or bomb death inside the live window carries equal victim and attacker team codes, so none is counted as an enemy kill.",
        f"Excluded: {status_counts.get('ambiguous_before_first_kill', 0)} rounds with unknown victim or attacker team before the first enemy kill, and {status_counts.get('tie_opposite_sides', 0)} same-tick opposite-side ties.",
        f"Not used: {checks['deaths_before_live_start']} deaths before live start (all weapon world) and {checks['deaths_after_round_end']} deaths after round end. Every analysis round has an enemy kill inside the window.",
        "Team codes are read from each death row (2 = T, 3 = CT, per the brief). They were not checked against round-start rosters or side switches.",
        f"Intervals are match-cluster percentile bootstrap (1,000 resamples, seed 0) over {M:,} clusters. The median first-kill time has no interval. The histogram is counts only.",
        f"Sensitivity: {denominators['analysis_rounds_in_max_rounds_16_matches']} analysis rounds come from max_rounds = 16 matches, so the MR12-only point estimates equal the full-sample ones.",
    ]
    return result, ct_hist, t_hist, edges, secs_all


def build_svg(res, ct_hist, t_hist, edges):
    W, H = 800, 450
    ax0, ax1, ay0, ay1 = 84, 456, 60, 330
    bx0, bx1 = 566, 764
    den = res["denominators"]
    fk = res["first_kill_time"]
    wr_ct = res["win_rate"]["ct_first_kill"]
    wr_t = res["win_rate"]["t_first_kill"]
    n_fk = den["analysis_rounds_first_enemy_kill"]
    med_all = fk["all"]["median"]
    out = []
    a = out.append
    a(f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">')
    a(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')

    # Panel A: stacked histogram, CT at the base and T on top.
    totals = ct_hist + t_hist
    ymax_raw = int(totals.max())
    step = nice_step(ymax_raw, 5)
    ymax = math.ceil(ymax_raw / step) * step

    def yA(v):
        return ay1 - (v / ymax) * (ay1 - ay0)

    def xA(t):
        return ax0 + (t / BIN_MAX) * (ax1 - ax0)

    k = 0
    while k * step <= ymax + 1e-9:
        v = k * step
        y = yA(v)
        a(f'<line x1="{ax0}" x2="{ax1}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        a(f'<text x="{ax0 - 8}" y="{y + 4:.1f}" font-size="12" text-anchor="end" fill="{NEUTRAL}">{int(round(v)):,}</text>')
        k += 1
    bw = (ax1 - ax0) / len(ct_hist)
    for i in range(len(ct_hist)):
        x = ax0 + i * bw + 1
        w = bw - 2
        c = int(ct_hist[i])
        t = int(t_hist[i])
        if c > 0:
            a(f'<rect x="{x:.2f}" y="{yA(c):.1f}" width="{w:.2f}" height="{yA(0) - yA(c):.1f}" fill="{CT_COL}"/>')
        if t > 0:
            a(f'<rect x="{x:.2f}" y="{yA(c + t):.1f}" width="{w:.2f}" height="{yA(c) - yA(c + t):.1f}" fill="{T_COL}"/>')
    a(f'<line x1="{ax0}" x2="{ax0}" y1="{ay0}" y2="{ay1}" stroke="{NEUTRAL}" stroke-width="1"/>')
    a(f'<line x1="{ax0}" x2="{ax1}" y1="{ay1}" y2="{ay1}" stroke="{NEUTRAL}" stroke-width="1"/>')
    for t in range(0, int(BIN_MAX) + 1, 15):
        x = xA(t)
        a(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{ay1}" y2="{ay1 + 4}" stroke="{NEUTRAL}" stroke-width="1"/>')
        a(f'<text x="{x:.1f}" y="{ay1 + 18}" font-size="12" text-anchor="middle" fill="{NEUTRAL}">{t}</text>')
    a(f'<text x="{(ax0 + ax1) / 2:.1f}" y="{ay1 + 38}" font-size="13" text-anchor="middle" fill="{NEUTRAL}">Seconds from live start (3 s bins)</text>')
    a(f'<text transform="translate(24 {(ay0 + ay1) / 2:.1f}) rotate(-90)" font-size="13" text-anchor="middle" fill="{NEUTRAL}">Rounds with a first kill (count)</text>')

    # Legend and panel A annotation.
    a(f'<rect x="{ax0}" y="20" width="11" height="11" fill="{CT_COL}"/>')
    a(f'<text x="{ax0 + 16}" y="30" font-size="12" fill="{NEUTRAL}">CT first kill</text>')
    a(f'<rect x="{ax0 + 120}" y="20" width="11" height="11" fill="{T_COL}"/>')
    a(f'<text x="{ax0 + 136}" y="30" font-size="12" fill="{NEUTRAL}">T first kill</text>')
    a(f'<text x="{ax0}" y="46" font-size="12" fill="{NEUTRAL}">n = {n_fk:,} rounds; median first kill {med_all:.1f} s</text>')

    # Panel B: first-kill-side win rate with match-cluster 95% CI.
    lo_all = min(wr_ct["ci95_match_cluster"][0], wr_t["ci95_match_cluster"][0], 0.5)
    hi_all = max(wr_ct["ci95_match_cluster"][1], wr_t["ci95_match_cluster"][1], 0.5)
    step_b = nice_step((hi_all - lo_all) + 0.04, 4)
    y_lo = math.floor((lo_all - 0.02) / step_b) * step_b
    y_hi = math.ceil((hi_all + 0.02) / step_b) * step_b

    def yB(v):
        return ay1 - (v - y_lo) / (y_hi - y_lo) * (ay1 - ay0)

    k0 = round(y_lo / step_b)
    k1 = round(y_hi / step_b)
    for kk in range(k0, k1 + 1):
        v = kk * step_b
        y = yB(v)
        a(f'<line x1="{bx0}" x2="{bx1}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        a(f'<text x="{bx0 - 8}" y="{y + 4:.1f}" font-size="12" text-anchor="end" fill="{NEUTRAL}">{pct(v)}</text>')
    a(f'<line x1="{bx0}" x2="{bx0}" y1="{ay0}" y2="{ay1}" stroke="{NEUTRAL}" stroke-width="1"/>')
    a(f'<line x1="{bx0}" x2="{bx1}" y1="{ay1}" y2="{ay1}" stroke="{NEUTRAL}" stroke-width="1"/>')
    y50 = yB(0.5)
    a(f'<line x1="{bx0}" x2="{bx1}" y1="{y50:.1f}" y2="{y50:.1f}" stroke="{NEUTRAL}" stroke-width="1.2" stroke-dasharray="4 4"/>')
    a(f'<text transform="translate(512 {(ay0 + ay1) / 2:.1f}) rotate(-90)" font-size="13" text-anchor="middle" fill="{NEUTRAL}">Win rate of first-kill side</text>')

    points = [
        ("CT first kill", bx0 + 0.27 * (bx1 - bx0), wr_ct, CT_COL),
        ("T first kill", bx0 + 0.73 * (bx1 - bx0), wr_t, T_COL),
    ]
    for label, xc, row, col in points:
        lo, hi = row["ci95_match_cluster"]
        rate = row["win_rate"]
        a(f'<line x1="{xc:.1f}" x2="{xc:.1f}" y1="{yB(lo):.1f}" y2="{yB(hi):.1f}" stroke="{col}" stroke-width="2"/>')
        a(f'<line x1="{xc - 7:.1f}" x2="{xc + 7:.1f}" y1="{yB(lo):.1f}" y2="{yB(lo):.1f}" stroke="{col}" stroke-width="2"/>')
        a(f'<line x1="{xc - 7:.1f}" x2="{xc + 7:.1f}" y1="{yB(hi):.1f}" y2="{yB(hi):.1f}" stroke="{col}" stroke-width="2"/>')
        a(f'<circle cx="{xc:.1f}" cy="{yB(rate):.1f}" r="5.5" fill="{col}"/>')
        a(f'<text x="{xc + 12:.1f}" y="{yB(rate) + 4:.1f}" font-size="12" fill="{NEUTRAL}">{rate * 100:.1f}%</text>')
        a(f'<text x="{xc:.1f}" y="{ay1 + 18}" font-size="12" text-anchor="middle" fill="{NEUTRAL}">{label}</text>')
        a(f'<text x="{xc:.1f}" y="{ay1 + 34}" font-size="12" text-anchor="middle" fill="{NEUTRAL}">n = {row["n_rounds"]:,} rounds</text>')
        a(f'<text x="{xc:.1f}" y="{ay1 + 48}" font-size="12" text-anchor="middle" fill="{NEUTRAL}">{row["n_matches"]:,} matches</text>')
    a(f'<text x="{bx1}" y="46" font-size="12" text-anchor="end" fill="{NEUTRAL}">whiskers: 95% CI, match bootstrap</text>')

    # Footnotes: denominators and definitions.
    exc_amb = den["excluded_ambiguous_team_data_before_first_kill"]
    exc_tie = den["excluded_tie_opposite_sides_same_tick"]
    exc_none = den["no_enemy_kill_after_live_start"]
    foot = [
        f"Clean 5v5 rounds (player_count = 10) from {den['clean_5v5_matches']:,} matches: {den['clean_5v5_rounds']:,} rounds, "
        f"{den['clean_rounds_with_live_start']:,} with a live-start event.",
        f"First kill = first enemy death after freeze end. Excluded: {exc_none:,} no enemy kill, "
        f"{exc_amb:,} unknown team data, {exc_tie:,} ties.",
        "Win rate = share of rounds won by the first-kill side. CI: match-cluster bootstrap, 1,000 resamples, seed 0.",
    ]
    for i, line in enumerate(foot):
        a(f'<text x="{ax0}" y="{402 + i * 16}" font-size="11" fill="{NEUTRAL}">{line}</text>')
    a('</svg>')
    return "\n".join(out) + "\n"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res, ct_hist, t_hist, edges, secs_all = compute()
    svg = build_svg(res, ct_hist, t_hist, edges)
    OUT_SVG.write_text(svg, encoding="utf-8")
    res["svg_path"] = str(OUT_SVG)
    res["json_path"] = str(OUT_JSON)
    OUT_JSON.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({"denominators": res["denominators"],
                      "win_rate": res["win_rate"],
                      "first_kill_time": {k: res["first_kill_time"][k] for k in
                                          ("n_total", "n_ct_first", "n_t_first", "overflow_above_120s",
                                           "negative_times", "min_s", "max_s", "all", "ct_first", "t_first")},
                      "checks": res["checks"]}, indent=2))


if __name__ == "__main__":
    main()
