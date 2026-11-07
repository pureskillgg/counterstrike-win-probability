"""T4: live round length (round_freeze_end -> round_end) by outcome, clean 5v5 rounds.

Reads (read-only):
  tomes/2026-10-01_to_08/tome/csds/basic_round_end.*      (round_end rows)
  tomes/2026-10-01_to_08/tome/csds/basic_round_state.*    (round_freeze_end events)
  tomes/2026-10-01_to_08/tome/csds/header/dataframe_*     (tick_rate per match)
  data/state_sample/rounds.parquet                        (only for a sensitivity cut on the 150 sample matches)

Writes:
  data/viz/T4.json  (plotted series, key numbers, accounting, caveats)
  data/viz/T4.svg   (800x450 chart, string-template SVG, no external libraries)

Method:
  clean 5v5 = round_end.player_count == 10, win_reason_code in {8, 9, 7, 1, 12}.
  length_s = (round_end.tick - min tick of round_freeze_end for same match_key, round) / tick_rate.
  Medians with 95% CIs from a match-cluster bootstrap: resample matches with replacement,
  1000 resamples, numpy default_rng(0), percentile interval.
"""

import glob
import json
import math
import os
import xml.etree.ElementTree as ET

import numpy as np
import polars as pl

REPO = r"G:\pureskill\csds-cs2"
TOME = os.path.join(REPO, "tomes", "2026-10-01_to_08", "tome", "csds")
SAMPLE_ROUNDS = os.path.join(REPO, "data", "state_sample", "rounds.parquet")
DATA_DIR = os.path.join(REPO, "data", "viz")
OUT_JSON = os.path.join(DATA_DIR, "T4.json")
OUT_SVG = os.path.join(DATA_DIR, "T4.svg")

TARGET_CODES = [8, 9, 7, 1, 12]
ORDER = [8, 9, 7, 1, 12]
OUTCOMES = {
    8: {"label": "CT elimination", "short": "CT elimination (8)", "side": "CT", "color": "#2563eb"},
    9: {"label": "T elimination", "short": "T elimination (9)", "side": "T", "color": "#d97706"},
    7: {"label": "Bomb defused", "short": "Bomb defused (7)", "side": "CT", "color": "#059669"},
    1: {"label": "Bomb exploded", "short": "Bomb exploded (1)", "side": "T", "color": "#92400e"},
    12: {"label": "Time-out (target saved)", "short": "Time-out (12)", "side": "CT", "color": "#475569"},
}
SEED = 0
N_BOOT = 1000
BIN_W = 5.0
X_MAX = 160.0
NEUTRAL = "#475569"
GRID = "#e2e8f0"

# SVG geometry (viewBox 0 0 800 450)
W_SVG, H_SVG = 800, 450
PL, PR, PT, PB = 72.0, 536.0, 30.0, 380.0


def parts(prefix):
    out = []
    for d in sorted(glob.glob(os.path.join(TOME, prefix + "*"))):
        out.extend(sorted(glob.glob(os.path.join(d, "dataframe_*"))))
    return out


def load_round_table():
    re_parts = parts("basic_round_end")
    rs_parts = parts("basic_round_state")
    hd_parts = glob.glob(os.path.join(TOME, "header", "dataframe_*"))
    if not (re_parts and rs_parts and hd_parts):
        raise SystemExit("T4: tome parts not found")

    round_end = pl.scan_parquet(re_parts).select(
        ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"]
    )
    fe = (
        pl.scan_parquet(rs_parts)
        .select(["match_key", "round", "tick", "event_type"])
        .filter(pl.col("event_type") == "round_freeze_end")
        .group_by(["match_key", "round"])
        .agg([pl.col("tick").min().alias("freeze_end_tick"), pl.len().alias("n_freeze_end")])
    )
    header = pl.scan_parquet(hd_parts).select(
        [pl.col("key").alias("match_key"), "max_rounds", "tick_rate", "is_wingman"]
    )
    allre = (
        round_end.join(fe, on=["match_key", "round"], how="left")
        .join(header, on="match_key", how="left")
        .collect()
    )
    hdr = header.collect()
    return allre, hdr


def main():
    allre, hdr = load_round_table()
    acc = {}
    acc["header_rows"] = hdr.height
    acc["header_unique_match_keys"] = hdr["match_key"].n_unique()
    acc["header_tick_rate_values"] = sorted(hdr["tick_rate"].unique().to_list())
    acc["header_max_rounds_counts"] = {
        str(k): v for k, v in sorted(hdr.group_by("max_rounds").len().rows())
    }
    acc["header_is_wingman_counts"] = {
        str(k): v for k, v in sorted(hdr.group_by("is_wingman").len().rows())
    }
    acc["header_max_rounds16_is_wingman_counts"] = {
        str(k): v
        for k, v in sorted(hdr.filter(pl.col("max_rounds") == 16).group_by("is_wingman").len().rows())
    }
    acc["round_end_rows_all"] = allre.height
    acc["round_end_matches_all"] = allre["match_key"].n_unique()
    acc["round_end_rows_missing_freeze_end_all"] = allre.filter(pl.col("freeze_end_tick").is_null()).height
    acc["freeze_end_max_events_per_round"] = int(allre["n_freeze_end"].max())
    if acc["freeze_end_max_events_per_round"] != 1:
        raise SystemExit("T4: expected one freeze_end per round")
    if acc["header_unique_match_keys"] != acc["header_rows"]:
        raise SystemExit("T4: header match_key not unique")

    five = allre.filter(pl.col("player_count") == 10)
    acc["round_end_rows_player_count_10"] = five.height
    other = five.filter(~pl.col("win_reason_code").is_in(TARGET_CODES))
    acc["player_count_10_other_codes"] = {
        str(k): v for k, v in sorted(other.group_by("win_reason_code").len().rows())
    }
    tgt = five.filter(pl.col("win_reason_code").is_in(TARGET_CODES))
    acc["target_code_rows_player_count_10"] = tgt.height
    acc["target_code_rows_missing_freeze_end"] = tgt.filter(pl.col("freeze_end_tick").is_null()).height

    ana = tgt.filter(pl.col("freeze_end_tick").is_not_null() & pl.col("tick_rate").is_not_null())
    ana = ana.with_columns(
        ((pl.col("tick") - pl.col("freeze_end_tick")) / pl.col("tick_rate")).alias("len_s")
    )
    if ana.filter(pl.col("len_s") <= 0).height > 0:
        raise SystemExit("T4: non-positive round length found")
    ana = ana.sort(["match_key", "round"])

    N = ana.height
    match_list = sorted(ana["match_key"].unique().to_list())
    M = len(match_list)
    midx = {m: i for i, m in enumerate(match_list)}
    acc["analysed_rounds"] = N
    acc["analysed_matches"] = M
    acc["matches_total"] = acc["round_end_matches_all"]
    acc["matches_without_analysed_rounds"] = acc["round_end_matches_all"] - M
    excluded_hdr = hdr.filter(~pl.col("match_key").is_in(match_list))
    acc["excluded_matches_by_is_wingman_and_max_rounds"] = [
        {"is_wingman": bool(w), "max_rounds": int(mr), "matches": int(c)}
        for w, mr, c in sorted(excluded_hdr.group_by(["is_wingman", "max_rounds"]).len().rows())
    ]
    acc["overtime_rounds_included"] = ana.filter(pl.col("round") > pl.col("max_rounds")).height

    keys = ana["match_key"].to_list()
    MI = np.array([midx[k] for k in keys], dtype=np.int64)
    L = ana["len_s"].to_numpy().astype(np.float64)
    C = ana["win_reason_code"].to_numpy().astype(np.int64)
    ROUND = ana["round"].to_numpy()
    assert L.shape[0] == N == MI.shape[0] == C.shape[0]

    # ----- per-outcome point estimates and histogram -----
    edges = [i * BIN_W for i in range(int(X_MAX / BIN_W) + 1)]
    max_len_seen = float(L.max())
    if max_len_seen >= X_MAX:
        raise SystemExit("T4: round length beyond x-axis range")

    per = {}
    for code in TARGET_CODES:
        sel = np.flatnonzero(C == code)
        ordr = sel[np.argsort(L[sel], kind="stable")]
        vals = L[ordr]
        mi = MI[ordr]
        n_code = vals.shape[0]
        if n_code == 0:
            raise SystemExit(f"T4: no rounds for code {code}")
        ones = np.ones(M, dtype=np.int64)
        wm = weighted_median(vals, mi, ones)
        if abs(wm - float(np.median(vals))) > 1e-9:
            raise SystemExit("T4: weighted median self-check failed")
        counts, _ = np.histogram(vals, bins=edges)
        if int(counts.sum()) != n_code:
            raise SystemExit("T4: histogram does not cover all rounds")
        per[code] = {
            "vals": vals,
            "mi": mi,
            "n": n_code,
            "n_matches": int(np.unique(mi).shape[0]),
            "median": float(np.median(vals)),
            "p05": float(np.percentile(vals, 5)),
            "p25": float(np.percentile(vals, 25)),
            "p75": float(np.percentile(vals, 75)),
            "p95": float(np.percentile(vals, 95)),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "mean": float(vals.mean()),
            "counts": counts.astype(int).tolist(),
            "share_pct": (counts / n_code * 100.0).tolist(),
        }

    # ----- match-cluster bootstrap for medians -----
    rng = np.random.default_rng(SEED)
    boot = {code: np.empty(N_BOOT, dtype=np.float64) for code in TARGET_CODES}
    for b in range(N_BOOT):
        draws = rng.integers(0, M, size=M)
        mult = np.bincount(draws, minlength=M).astype(np.int64)
        for code in TARGET_CODES:
            boot[code][b] = weighted_median(per[code]["vals"], per[code]["mi"], mult)
    for code in TARGET_CODES:
        lo, hi = np.percentile(boot[code], [2.5, 97.5])
        per[code]["ci_lo"] = float(lo)
        per[code]["ci_hi"] = float(hi)
        per[code]["boot_sd"] = float(boot[code].std(ddof=1))

    # ----- sensitivity: regulation rounds only (round <= header max_rounds) -----
    MAXR = ana["max_rounds"].to_numpy().astype(np.int64)
    reg_mask = ROUND.astype(np.int64) <= MAXR
    reg_sens = {}
    for code in TARGET_CODES:
        m = reg_mask & (C == code)
        reg_sens[str(code)] = {"n": int(m.sum()), "median": float(np.median(L[m]))}
    reg_sens["all_regulation_rounds"] = int(reg_mask.sum())

    # ----- sensitivity: 150 matches of the 1 Hz sample -----
    samp_keys = (
        pl.scan_parquet(SAMPLE_ROUNDS).select("match_key").unique().collect()["match_key"].to_list()
    )
    samp_set = set(samp_keys)
    sub_mask = np.array([k in samp_set for k in keys], dtype=bool)
    sens = {"sample_matches": len(samp_set), "matches_with_analysed_rounds": int(len(set(np.array(keys, dtype=object)[sub_mask].tolist())))}
    sens["by_outcome"] = {}
    for code in TARGET_CODES:
        m = sub_mask & (C == code)
        v = L[m]
        sens["by_outcome"][str(code)] = {
            "n": int(v.shape[0]),
            "median": float(np.median(v)) if v.shape[0] else None,
        }
    sens["all_clean_target_rounds_in_sample_matches"] = int(sub_mask.sum())
    tome_all_in_sample = allre.filter(pl.col("match_key").is_in(samp_keys))
    sens["tome_rounds_in_sample_matches_all_codes"] = tome_all_in_sample.height
    sens["tome_rounds_in_sample_matches_player_count_10"] = tome_all_in_sample.filter(
        pl.col("player_count") == 10
    ).height

    # ----- SVG -----
    ymax_raw = max(per[c]["share_pct"][i] for c in TARGET_CODES if c != 12 for i in range(len(edges) - 1))
    ymax = max(2, int(math.ceil(ymax_raw * 1.15 / 2.0) * 2))
    svg = build_svg(per, edges, ymax, N, M)
    ET.fromstring(svg)  # well-formedness check
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUT_SVG, "w", encoding="utf-8") as fh:
        fh.write(svg)

    # ----- JSON -----
    outcomes = []
    for code in ORDER:
        p = per[code]
        outcomes.append({
            "win_reason_code": code,
            "label": OUTCOMES[code]["label"],
            "legend_label": OUTCOMES[code]["short"],
            "winning_side": OUTCOMES[code]["side"],
            "color": OUTCOMES[code]["color"],
            "n_rounds": p["n"],
            "n_matches": p["n_matches"],
            "share_of_analysed_rounds_pct": round(p["n"] / N * 100.0, 3),
            "median_s": round(p["median"], 4),
            "median_ci95_s": [round(p["ci_lo"], 4), round(p["ci_hi"], 4)],
            "bootstrap_sd_of_median_s": round(p["boot_sd"], 4),
            "mean_s": round(p["mean"], 4),
            "p05_s": round(p["p05"], 4),
            "p25_s": round(p["p25"], 4),
            "p75_s": round(p["p75"], 4),
            "p95_s": round(p["p95"], 4),
            "min_s": round(p["min"], 4),
            "max_s": round(p["max"], 4),
            "hist_counts": p["counts"],
            "hist_share_pct": [round(x, 4) for x in p["share_pct"]],
        })

    result = {
        "id": "T4",
        "chart": "Live round length by outcome, overlaid histograms (clean 5v5)",
        "data_window": "2026-10-01 to 2026-10-08 tomes (1,752 matches)",
        "source_files": {
            "round_end": os.path.join(TOME, "basic_round_end.*", "dataframe_*"),
            "round_state": os.path.join(TOME, "basic_round_state.*", "dataframe_*"),
            "header": os.path.join(TOME, "header", "dataframe_*"),
            "sensitivity_sample": SAMPLE_ROUNDS,
        },
        "filters": {
            "player_count": 10,
            "win_reason_code_in": TARGET_CODES,
            "start_event": "round_freeze_end (earliest tick per match_key and round; exactly one per round)",
            "end_event": "basic_round_end tick",
            "length_formula": "(round_end.tick - round_freeze_end.tick) / tick_rate (tick_rate=64 for every match)",
            "overtime": "rounds numbered above header max_rounds are included",
        },
        "n_rounds": N,
        "n_matches": M,
        "bootstrap": {
            "unit": "match (cluster)",
            "resample_population": "analysed matches (those with at least one analysed round)",
            "resamples": N_BOOT,
            "seed": SEED,
            "rng": "numpy.random.default_rng(0)",
            "statistic": "median (np.median convention, average of the two middle values for even n)",
            "interval": "percentile 2.5 to 97.5",
        },
        "histogram": {
            "bin_width_s": BIN_W,
            "edges_s": edges,
            "y_axis": "share of each outcome's own rounds, percent per bin",
            "y_axis_max_plotted_pct": ymax,
            "note": "time-out bin is clipped on the plot; all 463 time-outs sit at 115.0 s",
        },
        "outcomes": outcomes,
        "accounting": acc,
        "sensitivity_regulation_rounds_only": reg_sens,
        "sensitivity_sample_150_matches": sens,
        "key_numbers": {
            "all_clean_median_s": round(float(np.median(L)), 4),
            "all_clean_n": N,
            "time_out_all_exactly_115_s": bool(per[12]["min"] == per[12]["max"] == 115.0),
        },
        "caveats": [
            "Population is the full 2026-10 tome, not the 1 Hz state sample. The sample's 2,673 clean rounds are a different filter: its 150 matches hold 3,054 tome rounds, of which 2,841 are player_count 10 (sensitivity block above).",
            "The 149 header max_rounds = 16 matches are all Wingman (is_wingman = True) and contribute no clean 5v5 rounds, so every analysed match is a 24-round game. Of the 160 matches with no analysed rounds, 149 are these Wingman matches and 11 are 24-round non-Wingman matches whose rounds are not 10-player target-code rounds (not examined further).",
            "Overtime rounds (round number above max_rounds, 1,088 rounds) are included. Regulation-only medians are in sensitivity_regulation_rounds_only and move each median by under 0.5 s. The overtime boundary is approximate because it relies on header max_rounds.",
            "round_freeze_end is taken as the round start; the anchor is the tome event, not verified against the demo timeline. Time-out rounds all end at exactly 115.0 s, consistent with a standard 115 s round timer, but this was not checked against the header value.",
            "Win reason codes 11, 13, 17, 18 at player_count 10 are excluded (7 rounds). Rounds without a freeze-end event (39 across all codes) are excluded; none of the five target codes lost rounds this way.",
            "Intervals cover medians only, from a match-cluster bootstrap (1,000 resamples, seed 0). Per-bin histogram shares have no intervals drawn.",
            "Histograms show each outcome as a share of its own rounds; absolute counts are in hist_counts. The time-out bar is drawn clipped at the top of the plot.",
        ],
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)

    # ----- console summary for checking -----
    print(f"analysed rounds: {N}, matches: {M}, overtime rounds included: {acc['overtime_rounds_included']}")
    print(f"ymax plotted: {ymax}, raw max share (non time-out): {ymax_raw:.3f}")
    for code in ORDER:
        p = per[code]
        print(
            f"{OUTCOMES[code]['short']:<20} n={p['n']:>6} m={p['n_matches']:>5} "
            f"median={p['median']:.3f} CI=[{p['ci_lo']:.3f},{p['ci_hi']:.3f}] "
            f"p25={p['p25']:.2f} p75={p['p75']:.2f} min={p['min']:.2f} max={p['max']:.2f}"
        )
    print("hist counts sum check:", {c: sum(per[c]['counts']) for c in ORDER})
    print("sensitivity:", json.dumps(sens))
    print("accounting:", json.dumps(acc))
    print("wrote", OUT_JSON, "and", OUT_SVG)


def weighted_median(vals, mi, mult):
    w = mult[mi]
    c = np.cumsum(w)
    n = int(c[-1])
    lo = (n - 1) // 2
    hi = n // 2
    x_lo = vals[np.searchsorted(c, lo, side="right")]
    x_hi = vals[np.searchsorted(c, hi, side="right")]
    return 0.5 * (float(x_lo) + float(x_hi))


def fmt_int(n):
    return f"{int(n):,}"


def build_svg(per, edges, ymax, n_total, n_matches):
    xscale = (PR - PL) / X_MAX

    def sx(s):
        return PL + s * xscale

    def sy(p):
        return PB - (p / ymax) * (PB - PT)

    out = []
    out.append(f'<svg viewBox="0 0 {W_SVG} {H_SVG}" xmlns="http://www.w3.org/2000/svg" role="img" '
               f'aria-label="Overlaid histograms of live round length by outcome, clean 5v5 rounds">')
    out.append(f'<rect x="0" y="0" width="{W_SVG}" height="{H_SVG}" fill="#ffffff"/>')
    out.append('<g font-family="sans-serif">')

    # grid and ticks
    for t in range(0, int(X_MAX) + 1, 20):
        x = sx(t)
        out.append(f'<line x1="{x:.1f}" y1="{PT:.1f}" x2="{x:.1f}" y2="{PB:.1f}" stroke="{GRID}" stroke-width="1"/>')
        out.append(f'<text x="{x:.1f}" y="{PB + 18:.1f}" font-size="12" fill="{NEUTRAL}" text-anchor="middle">{t}</text>')
    ystep = 2
    for p in range(0, ymax + 1, ystep):
        y = sy(p)
        out.append(f'<line x1="{PL:.1f}" y1="{y:.1f}" x2="{PR:.1f}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        out.append(f'<text x="{PL - 8:.1f}" y="{y + 4:.1f}" font-size="12" fill="{NEUTRAL}" text-anchor="end">{p}</text>')

    # axes
    out.append(f'<line x1="{PL:.1f}" y1="{PB:.1f}" x2="{PR:.1f}" y2="{PB:.1f}" stroke="{NEUTRAL}" stroke-width="1.2"/>')
    out.append(f'<line x1="{PL:.1f}" y1="{PT:.1f}" x2="{PL:.1f}" y2="{PB:.1f}" stroke="{NEUTRAL}" stroke-width="1.2"/>')

    # axis titles
    out.append(f'<text x="{(PL + PR) / 2:.1f}" y="418" font-size="13" fill="{NEUTRAL}" text-anchor="middle">'
               f'Live round length, round_freeze_end to round_end (seconds)</text>')
    out.append(f'<text transform="translate(20 {(PT + PB) / 2:.1f}) rotate(-90)" font-size="13" fill="{NEUTRAL}" '
               f'text-anchor="middle">Share of each outcome\'s rounds (%)</text>')

    # sample size label
    out.append(f'<text x="{PL + 8:.1f}" y="{PT + 16:.1f}" font-size="12" fill="{NEUTRAL}">'
               f'n = {fmt_int(n_total)} clean 5v5 rounds, {fmt_int(n_matches)} matches</text>')

    # CI bands (drawn first, under everything)
    for code in ORDER:
        p = per[code]
        x0 = sx(p["ci_lo"])
        w = max(1.0, sx(p["ci_hi"]) - x0)
        out.append(f'<rect x="{x0:.2f}" y="{PT:.1f}" width="{w:.2f}" height="{PB - PT:.1f}" '
                   f'fill="{OUTCOMES[code]["color"]}" fill-opacity="0.12"/>')

    # histogram outlines, time-out first so the others draw on top
    for code in [12, 7, 8, 9, 1]:
        p = per[code]
        col = OUTCOMES[code]["color"]
        pts = [(sx(edges[0]), PB)]
        for i, sh in enumerate(p["share_pct"]):
            y = sy(min(sh, ymax))
            pts.append((sx(edges[i]), y))
            pts.append((sx(edges[i + 1]), y))
        pts.append((sx(edges[-1]), PB))
        d = "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in pts) + " Z"
        out.append(f'<path d="{d}" fill="{col}" fill-opacity="0.16" stroke="{col}" stroke-width="1.6" '
                   f'stroke-linejoin="round"/>')

    # clipped time-out label
    p12 = per[12]
    out.append(f'<text x="{sx(112):.1f}" y="{PT + 36:.1f}" font-size="12" fill="{NEUTRAL}" text-anchor="end">'
               f'Time-out: {fmt_int(p12["n"])} rounds, all at {p12["median"]:.1f} s (bar clipped)</text>')

    # median lines
    for code in ORDER:
        p = per[code]
        col = OUTCOMES[code]["color"]
        x = sx(p["median"])
        out.append(f'<line x1="{x:.2f}" y1="{PB:.1f}" x2="{x:.2f}" y2="{PT:.1f}" stroke="{col}" '
                   f'stroke-width="1.5" stroke-dasharray="5 3"/>')

    # legend
    lx = 556.0
    out.append(f'<text x="{lx:.1f}" y="44" font-size="12" fill="{NEUTRAL}">Outcome (win_reason_code)</text>')
    for i, code in enumerate(ORDER):
        p = per[code]
        col = OUTCOMES[code]["color"]
        y0 = 60 + i * 58
        out.append(f'<rect x="{lx:.1f}" y="{y0 + 1:.1f}" width="14" height="10" fill="{col}" fill-opacity="0.3" '
                   f'stroke="{col}" stroke-width="1.5"/>')
        out.append(f'<text x="{lx + 20:.1f}" y="{y0 + 11:.1f}" font-size="13" fill="#1e293b">'
                   f'{OUTCOMES[code]["short"]}</text>')
        out.append(f'<text x="{lx + 20:.1f}" y="{y0 + 27:.1f}" font-size="12" fill="{NEUTRAL}">'
                   f'n = {fmt_int(p["n"])} rounds</text>')
        out.append(f'<text x="{lx + 20:.1f}" y="{y0 + 42:.1f}" font-size="12" fill="{NEUTRAL}">'
                   f'median {p["median"]:.1f} s [CI {p["ci_lo"]:.1f}, {p["ci_hi"]:.1f}]</text>')

    # footnote
    out.append(f'<text x="{PL:.1f}" y="440" font-size="11" fill="{NEUTRAL}">'
               f'Bars: 5 s bins, % of each outcome. Dashed line: median. Band: 95% match-cluster bootstrap CI '
               f'(1,000 resamples, seed 0).</text>')
    out.append('</g>')
    out.append('</svg>')
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()
