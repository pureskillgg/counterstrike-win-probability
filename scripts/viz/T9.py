# -*- coding: utf-8 -*-
"""T9: utility per round -- mean flashbang, smoke and HE detonations per round,
by round number (1-24), from the 2026 tome grenade_state channel.

Population: clean 5v5 rounds (round_end.player_count == 10), round numbers 1-24.
Detonations are counted from grenade_state event_type in
{flashbang_detonate, smokegrenade_detonate, hegrenade_detonate}, joined to the
round on (match_key, round). Rounds with zero detonations stay in the denominator.

Intervals: match-cluster bootstrap (resample matches with replacement, 1000
resamples, numpy default_rng seed 0), ratio of sums per resample, 2.5/97.5
percentiles.

Outputs: data/viz/T9.json and data/viz/T9.svg (no external libraries for drawing).
"""
import html
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path("G:/pureskill/csds-cs2")
CHAN = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
GRN = CHAN / "basic_grenade_state.2026-09-18,2026-10-08" / "dataframe_00000"
REND = CHAN / "basic_round_end.2026-09-18,2026-10-08" / "dataframe_00000"
HDR = CHAN / "header" / "dataframe_00000"
OUT_JSON = ROOT / "data" / "viz" / "T9.json"
OUT_SVG = ROOT / "data" / "viz" / "T9.svg"

MAX_ROUND = 24
PISTOL_ROUNDS = (1, 13)
N_BOOT = 1000
SEED = 0
CI_LO, CI_HI = 2.5, 97.5

# key, legend label, grenade_state event_type, colour, marker shape
SERIES = [
    ("flash", "Flashbang", "flashbang_detonate", "#2563eb", "circle"),
    ("smoke", "Smoke", "smokegrenade_detonate", "#475569", "square"),
    ("he", "HE grenade", "hegrenade_detonate", "#059669", "diamond"),
]
NEUTRAL = "#475569"
GRID = "#e2e8f0"
TEXT = "#334155"


def load_population():
    rend = pl.scan_parquet(REND).select(["match_key", "round", "tick", "player_count"]).collect()
    assert rend.unique(subset=["match_key", "round"]).height == rend.height, "duplicate round_end keys"
    clean = rend.filter((pl.col("player_count") == 10) & pl.col("round").is_between(1, MAX_ROUND))
    return rend, clean


def load_detonations():
    ev_types = [s[2] for s in SERIES]
    grn = (
        pl.scan_parquet(GRN)
        .select(["match_key", "round", "tick", "event_type", "entity_id"])
        .filter(pl.col("event_type").is_in(ev_types) & pl.col("round").is_between(1, MAX_ROUND))
        .collect()
    )
    return grn


def main():
    rend, clean = load_population()
    grn = load_detonations()

    # Context counts before the 5v5 filter
    all_rounds_1_24 = {}
    for key, label, ev, _, _ in SERIES:
        all_rounds_1_24[key] = int(grn.filter(pl.col("event_type") == ev).height)

    # Duplicate check on the detonation events (same grenade logged twice would show here)
    dup_check = grn.group_by(["match_key", "round", "event_type", "entity_id", "tick"]).len()
    n_dup_detonations = int(dup_check.filter(pl.col("len") > 1).height)

    # Keep only clean 5v5 rounds (1-24) and attach round end tick
    g = grn.join(
        clean.select(["match_key", "round", pl.col("tick").alias("end_tick")]),
        on=["match_key", "round"],
        how="inner",
    ).with_columns((pl.col("tick") > pl.col("end_tick")).alias("post_end"))

    aggs = []
    for key, _, ev, _, _ in SERIES:
        aggs.append((pl.col("event_type") == ev).sum().alias(key))
        aggs.append(((pl.col("event_type") == ev) & pl.col("post_end")).sum().alias(key + "_post"))
    cnt = g.group_by(["match_key", "round"]).agg(aggs)

    base = clean.select(["match_key", "round"]).join(cnt, on=["match_key", "round"], how="left")
    fill_cols = [c for c in base.columns if c not in ("match_key", "round")]
    base = base.with_columns([pl.col(c).fill_null(0) for c in fill_cols])

    matches = sorted(base["match_key"].unique().to_list())
    midx = {m: i for i, m in enumerate(matches)}
    M = len(matches)

    # A[match, round-1, 0] = clean round indicator; [1..3] = flash, smoke, HE counts
    A = np.zeros((M, MAX_ROUND, 4))
    P = np.zeros((M, MAX_ROUND, 3))  # detonations after round_end tick (sensitivity only)
    for row in base.iter_rows(named=True):
        i = midx[row["match_key"]]
        j = int(row["round"]) - 1
        A[i, j, 0] = 1.0
        for k, s in enumerate(SERIES):
            A[i, j, k + 1] = float(row[s[0]])
            P[i, j, k] = float(row[s[0] + "_post"])

    n_r = A[:, :, 0].sum(axis=0)
    assert (n_r > 0).all(), "a round number has no clean rounds"
    tot_r = A[:, :, 1:].sum(axis=0)
    mean_r = tot_r / n_r[:, None]
    n_rounds = int(n_r.sum())

    # Match-cluster bootstrap
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, M, size=(N_BOOT, M))
    W = np.zeros((N_BOOT, M))
    for b in range(N_BOOT):
        W[b] = np.bincount(idx[b], minlength=M)
    S = (W @ A.reshape(M, -1)).reshape(N_BOOT, MAX_ROUND, 4)
    bn = S[:, :, 0]
    bmean = S[:, :, 1:] / bn[:, :, None]
    lo = np.percentile(bmean, CI_LO, axis=0)
    hi = np.percentile(bmean, CI_HI, axis=0)

    # Pooled over rounds 1-24, pistol rounds, non-pistol rounds
    pooled_pt = A[:, :, 1:].sum(axis=(0, 1)) / n_r.sum()
    Tot = S.sum(axis=1)
    pooled_b = Tot[:, 1:] / Tot[:, 0:1]
    pooled_lo, pooled_hi = np.percentile(pooled_b, [CI_LO, CI_HI], axis=0)

    pi = [r - 1 for r in PISTOL_ROUNDS]
    npi = [r for r in range(MAX_ROUND) if r not in pi]
    pist_pt = A[:, pi, 1:].sum(axis=(0, 1)) / n_r[pi].sum()
    pist_b = S[:, pi, 1:].sum(axis=1) / S[:, pi, 0].sum(axis=1)[:, None]
    nonp_pt = A[:, npi, 1:].sum(axis=(0, 1)) / n_r[npi].sum()
    nonp_b = S[:, npi, 1:].sum(axis=1) / S[:, npi, 0].sum(axis=1)[:, None]

    totals = [int(round(x)) for x in tot_r.sum(axis=0)]
    post_totals = [int(round(x)) for x in P.sum(axis=(0, 1))]
    excl_pt = (tot_r.sum(axis=0) - P.sum(axis=(0, 1))) / n_r.sum()

    # Header check: max_rounds and tick rate for analysis matches
    hdr = pl.scan_parquet(HDR).select(["key", "max_rounds", "tick_rate"]).collect()
    hm = hdr.filter(pl.col("key").is_in(matches))
    header_info = {
        "header_rows": int(hdr.height),
        "analysis_matches": M,
        "analysis_matches_matched_in_header": int(hm.height),
        "max_rounds_counts": {str(r["max_rounds"]): int(r["len"]) for r in hm.group_by("max_rounds").len().iter_rows(named=True)},
        "tick_rate_counts": {str(r["tick_rate"]): int(r["len"]) for r in hm.group_by("tick_rate").len().iter_rows(named=True)},
    }

    # Sanity checks on outputs
    assert (mean_r >= 0).all() and (lo >= 0).all()
    assert (lo <= hi + 1e-12).all()
    assert all(int(x) >= 0 for x in totals)

    rounds = list(range(1, MAX_ROUND + 1))
    series_out = []
    for k, (key, label, ev, colour, marker) in enumerate(SERIES):
        per = []
        for r in rounds:
            j = r - 1
            per.append({
                "round": r,
                "pistol": r in PISTOL_ROUNDS,
                "n_rounds": int(n_r[j]),
                "detonations": int(round(tot_r[j, k])),
                "mean": round(float(mean_r[j, k]), 4),
                "ci95_lo": round(float(lo[j, k]), 4),
                "ci95_hi": round(float(hi[j, k]), 4),
            })
        series_out.append({
            "key": key,
            "label": label,
            "event_type": ev,
            "colour": colour,
            "marker": marker,
            "total_detonations_analysis": totals[k],
            "total_detonations_all_rounds_1_24_any_player_count": all_rounds_1_24[key],
            "per_round": per,
        })

    peak = {}
    for k, s in enumerate(SERIES):
        j = int(np.argmax(mean_r[:, k]))
        peak[s[0]] = {"round": j + 1, "mean": round(float(mean_r[j, k]), 4)}
    low = {}
    for k, s in enumerate(SERIES):
        j = int(np.argmin(mean_r[:, k]))
        low[s[0]] = {"round": j + 1, "mean": round(float(mean_r[j, k]), 4)}

    def triple(pt, blo, bhi):
        out = {}
        for k, s in enumerate(SERIES):
            out[s[0]] = {
                "mean": round(float(pt[k]), 4),
                "ci95_lo": round(float(blo[k]), 4),
                "ci95_hi": round(float(bhi[k]), 4),
            }
        return out

    pist_lo, pist_hi = np.percentile(pist_b, [CI_LO, CI_HI], axis=0)
    nonp_lo, nonp_hi = np.percentile(nonp_b, [CI_LO, CI_HI], axis=0)

    result = {
        "id": "T9",
        "chart": "Utility per round: mean flashbang, smoke and HE detonations per round, by round number (1-24)",
        "source": {
            "grenade_state": str(GRN),
            "round_end": str(REND),
            "header": str(HDR),
            "detonation_event_types": [s[2] for s in SERIES],
        },
        "population": {
            "rule": "round_end.player_count == 10 and round in 1..24",
            "matches": M,
            "rounds": n_rounds,
            "round_end_rows_all_rounds_all_player_counts": int(rend.height),
            "round_end_rows_player_count_10_rounds_1_24": int(clean.height),
            "note": "clean 5v5 filter applied to round_end rows with round 1-24; overtime rounds 25-48 excluded",
            "rounds_per_round_number": {str(r): int(n_r[r - 1]) for r in rounds},
        },
        "bootstrap": {
            "method": "match-cluster (resample matches with replacement)",
            "resamples": N_BOOT,
            "seed": SEED,
            "rng": "numpy.random.default_rng(0).integers(0, M, size=(1000, M))",
            "interval": "percentile 2.5 / 97.5, ratio of sums within each resample",
        },
        "series": series_out,
        "pooled_mean_per_round_rounds_1_24": triple(pooled_pt, pooled_lo, pooled_hi),
        "pistol_rounds_1_and_13": triple(pist_pt, pist_lo, pist_hi),
        "non_pistol_rounds": triple(nonp_pt, nonp_lo, nonp_hi),
        "peak_round_by_mean": peak,
        "lowest_round_by_mean": low,
        "sensitivity_excluding_detonations_after_round_end_tick": {
            "post_round_end_detonations": {s[0]: post_totals[k] for k, s in enumerate(SERIES)},
            "pooled_mean_per_round": {s[0]: round(float(excl_pt[k]), 4) for k, s in enumerate(SERIES)},
        },
        "header_check": header_info,
        "duplicate_detonation_rows_same_match_round_type_entity_tick": n_dup_detonations,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")

    svg = build_svg(series_out, mean_r, lo, hi, totals, M, n_rounds, n_r)
    ET.fromstring(svg)  # well-formedness check
    OUT_SVG.write_text(svg, encoding="utf-8")

    # Console summary for the run log
    print("matches", M, "rounds", n_rounds)
    print("rounds per round number", [int(x) for x in n_r])
    print("totals (analysis)", dict(zip([s[0] for s in SERIES], totals)))
    print("all rounds 1-24 any player count", all_rounds_1_24)
    print("post-round-end detonations", post_totals)
    print("duplicate detonation rows", n_dup_detonations)
    print("header", header_info)
    for k, s in enumerate(SERIES):
        print(s[0], "pooled", round(float(pooled_pt[k]), 4), "CI", round(float(pooled_lo[k]), 4), round(float(pooled_hi[k]), 4))
        print(s[0], "peak", peak[s[0]], "low", low[s[0]])
    print("pistol", triple(pist_pt, pist_lo, pist_hi))
    print("non-pistol", triple(nonp_pt, nonp_lo, nonp_hi))
    print("excl post-end pooled", result["sensitivity_excluding_detonations_after_round_end_tick"]["pooled_mean_per_round"])
    print("wrote", OUT_JSON, OUT_SVG)


def nice_step(raw):
    exp = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if m * exp >= raw:
            return m * exp
    return 10 * exp


def build_svg(series_out, mean_r, lo, hi, totals, M, n_rounds, n_r):
    W_, H_ = 800, 450
    X0, X1, Y0, Y1 = 72.0, 784.0, 74.0, 340.0
    ymax_raw = float(hi.max()) * 1.08
    step = nice_step(ymax_raw / 7.0)
    ymax = math.ceil(ymax_raw / step) * step
    ticks = [round(step * i, 6) for i in range(int(round(ymax / step)) + 1)]

    def xp(r):
        return X0 + (r - 0.5) / MAX_ROUND * (X1 - X0)

    def yp(v):
        return Y1 - v / ymax * (Y1 - Y0)

    half = (X1 - X0) / MAX_ROUND / 2.0
    parts = []
    parts.append(f'<svg viewBox="0 0 {W_} {H_}" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">')
    parts.append(f'<rect x="0" y="0" width="{W_}" height="{H_}" fill="#ffffff"/>')

    # Legend (top row)
    for i, s in enumerate(series_out):
        lx = 72 + i * 240
        ly = 24
        col = s["colour"]
        parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 22}" y2="{ly}" stroke="{col}" stroke-width="2"/>')
        parts.append(marker_svg(lx + 11, ly, col, s["marker"]))
        label = f'{s["label"]} (total {totals[i]:,})'
        parts.append(f'<text x="{lx + 30}" y="{ly + 4.5}" font-size="13" fill="{TEXT}">{html.escape(label)}</text>')

    # Pistol round bands and labels
    for r in PISTOL_ROUNDS:
        parts.append(f'<rect x="{xp(r) - half:.2f}" y="{Y0}" width="{2 * half:.2f}" height="{Y1 - Y0}" fill="{GRID}"/>')
        parts.append(f'<text x="{xp(r):.2f}" y="{Y0 - 8}" font-size="12" text-anchor="middle" fill="{NEUTRAL}">pistol</text>')

    # Gridlines and y tick labels
    for v in ticks:
        y = yp(v)
        if v > 0:
            parts.append(f'<line x1="{X0}" y1="{y:.2f}" x2="{X1}" y2="{y:.2f}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(f'<text x="64" y="{y + 4:.2f}" font-size="12" text-anchor="end" fill="{TEXT}">{v:g}</text>')

    # CI bands
    for k, s in enumerate(series_out):
        col = s["colour"]
        top = [(xp(r), yp(hi[r - 1, k])) for r in range(1, MAX_ROUND + 1)]
        bot = [(xp(r), yp(lo[r - 1, k])) for r in range(MAX_ROUND, 0, -1)]
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in top + bot)
        parts.append(f'<polygon points="{pts}" fill="{col}" fill-opacity="0.15" stroke="none"/>')

    # Mean lines and markers
    for k, s in enumerate(series_out):
        col = s["colour"]
        pts = " ".join(f"{xp(r):.2f},{yp(mean_r[r - 1, k]):.2f}" for r in range(1, MAX_ROUND + 1))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2" stroke-linejoin="round"/>')
        for r in range(1, MAX_ROUND + 1):
            parts.append(marker_svg(xp(r), yp(mean_r[r - 1, k]), col, s["marker"]))

    # Axes
    parts.append(f'<line x1="{X0}" y1="{Y1}" x2="{X1}" y2="{Y1}" stroke="{NEUTRAL}" stroke-width="1"/>')
    parts.append(f'<line x1="{X0}" y1="{Y0}" x2="{X0}" y2="{Y1}" stroke="{NEUTRAL}" stroke-width="1"/>')
    for r in range(1, MAX_ROUND + 1):
        x = xp(r)
        parts.append(f'<line x1="{x:.2f}" y1="{Y1}" x2="{x:.2f}" y2="{Y1 + 4}" stroke="{NEUTRAL}" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="{Y1 + 17}" font-size="12" text-anchor="middle" fill="{TEXT}">{r}</text>')

    # Axis titles
    parts.append(f'<text x="{(X0 + X1) / 2:.1f}" y="376" font-size="13" text-anchor="middle" fill="{TEXT}">Round number (MR12; pistol rounds are 1 and 13)</text>')
    parts.append(f'<text x="18" y="{(Y0 + Y1) / 2:.1f}" font-size="13" text-anchor="middle" fill="{TEXT}" transform="rotate(-90 18 {(Y0 + Y1) / 2:.1f})">Mean detonations per round</text>')

    # Sample size footer (n per round number falls because later rounds only occur in longer matches)
    n_first, n_last = int(n_r[0]), int(n_r[-1])
    parts.append(f'<text x="72" y="396" font-size="12" fill="{TEXT}">n = {M:,} matches, {n_rounds:,} rounds (clean 5v5, player_count = 10; round numbers 1-24)</text>')
    parts.append(f'<text x="72" y="413" font-size="12" fill="{TEXT}">Shaded bands: 95% match-cluster bootstrap CI (1,000 resamples, seed 0). Points: mean per round.</text>')
    parts.append(f'<text x="72" y="430" font-size="12" fill="{TEXT}">Higher round numbers come only from longer matches: n falls from {n_first:,} (round 1) to {n_last:,} (round 24).</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def marker_svg(x, y, col, shape):
    if shape == "square":
        return f'<rect x="{x - 3:.2f}" y="{y - 3:.2f}" width="6" height="6" fill="{col}"/>'
    if shape == "diamond":
        return f'<polygon points="{x:.2f},{y - 4:.2f} {x + 4:.2f},{y:.2f} {x:.2f},{y + 4:.2f} {x - 4:.2f},{y:.2f}" fill="{col}"/>'
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{col}"/>'


if __name__ == "__main__":
    main()
