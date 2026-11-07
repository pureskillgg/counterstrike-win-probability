#!/usr/bin/env python3
"""S2: mean freeze-end team equipment value by round number, CT vs T.

Source: data/state_sample only (1 Hz sample, 150 matches, 2,673 rounds).
Freeze-end state = the first snapshot per (match_key, round) (seconds_since_live == min).
Team value = sum of equipment_value_calc over the alive players of that side (team_code 3 = CT, 2 = T).
Side is the side the team is on in that round (sides swap after round 12).

Intervals: percentile 95% CIs from a match-cluster bootstrap. Matches are resampled with
replacement, 1000 resamples, random.Random(0). Each resample recomputes every round-number
mean from the resampled matches (a match drawn twice counts twice).

Outputs: data/viz/S2.json and data/viz/S2.svg. Stdlib plus polars only; no network.
"""
from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = ROOT / "data" / "state_sample"
OUT_JSON = ROOT / "data" / "viz" / "S2.json"
OUT_SVG = ROOT / "data" / "viz" / "S2.svg"

CT_CODE = 3
T_CODE = 2
ROUNDS = list(range(1, 25))
PISTOL_ROUNDS = [1, 13]
FULL_SIDE = 5  # a clean 5v5 side has exactly 5 alive players at freeze end
N_BOOT = 1000
SEED = 0

COL_CT = "#2563eb"
COL_T = "#d97706"
COL_NEUTRAL = "#475569"
COL_GRID = "#e2e8f0"

# SVG geometry (viewBox 0 0 800 450)
W, H = 800, 450
PL, PR, PT, PB = 84.0, 784.0, 40.0, 358.0
X_LO, X_HI = 0.5, 24.5  # round slots; round r is centred in its slot


def sx(r: float) -> float:
    return PL + (r - X_LO) / (X_HI - X_LO) * (PR - PL)


def fmt_int(v: float) -> str:
    return f"{v:,.0f}"


def load_round_table() -> pl.DataFrame:
    """One row per (match_key, round) in the sample, with freeze-end side sums."""
    rounds = (
        pl.scan_parquet(SAMPLE_DIR / "rounds.parquet")
        .select(["match_key", "round", "round_number"])
        .collect()
    )
    freeze = (
        pl.scan_parquet(SAMPLE_DIR / "players_1hz.parquet")
        .select(["match_key", "round", "seconds_since_live", "team_code", "equipment_value_calc"])
        .with_columns(
            (
                pl.col("seconds_since_live")
                == pl.col("seconds_since_live").min().over(["match_key", "round"])
            ).alias("is_first")
        )
        .filter(pl.col("is_first"))
        .group_by(["match_key", "round"])
        .agg(
            pl.col("equipment_value_calc").filter(pl.col("team_code") == CT_CODE).sum().alias("ct_equipment_calc"),
            pl.col("equipment_value_calc").filter(pl.col("team_code") == T_CODE).sum().alias("t_equipment_calc"),
            (pl.col("team_code") == CT_CODE).sum().alias("ct_alive"),
            (pl.col("team_code") == T_CODE).sum().alias("t_alive"),
            pl.col("seconds_since_live").min().alias("first_sec"),
            pl.len().alias("n_rows"),
        )
        .collect()
    )
    return rounds.join(freeze, on=["match_key", "round"], how="inner")


def percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolation percentile on an already sorted list (same as numpy default)."""
    n = len(sorted_vals)
    pos = q * (n - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def main() -> None:
    table = load_round_table()

    # ---- integrity checks on the freeze-end snapshot ----
    assert table.height == 2673, table.height
    assert table.select(pl.col("match_key").n_unique()).item() == 150
    assert bool((table["first_sec"] == 0).all()), "freeze-end snapshot is not second 0 everywhere"
    assert bool((table["n_rows"] == table["ct_alive"] + table["t_alive"]).all())
    assert bool(((table["ct_alive"] >= 0) & (table["t_alive"] >= 0)).all()), "negative player count"

    window = table.filter(pl.col("round_number").is_between(1, 24))
    ot_rounds_excluded = table.height - window.height
    incomplete = window.filter((pl.col("ct_alive") != FULL_SIDE) | (pl.col("t_alive") != FULL_SIDE))
    clean = window.filter((pl.col("ct_alive") == FULL_SIDE) & (pl.col("t_alive") == FULL_SIDE))
    assert clean.select(["match_key", "round_number"]).unique().height == clean.height, "duplicate match-round"
    assert bool((clean["ct_equipment_calc"] >= 0).all()) and bool((clean["t_equipment_calc"] >= 0).all())
    zero_side = clean.filter((pl.col("ct_equipment_calc") == 0) | (pl.col("t_equipment_calc") == 0)).height

    # cross-check against the project's round buy-type table (sums must match where present)
    buy = pl.read_parquet(SAMPLE_DIR / "round_buy_types.parquet").select(
        ["match_key", "round", "ct_equipment", "t_equipment"]
    )
    cmp = clean.join(buy, on=["match_key", "round"], how="inner")
    buy_checked = cmp.height
    buy_ct_match = int((cmp["ct_equipment_calc"] == cmp["ct_equipment"]).sum())
    buy_t_match = int((cmp["t_equipment_calc"] == cmp["t_equipment"]).sum())

    # ---- per-match structure for the cluster bootstrap ----
    all_matches = sorted(table["match_key"].unique().to_list())
    n_matches = len(all_matches)
    idx_of = {m: i for i, m in enumerate(all_matches)}
    per_match: list[list[tuple[int, float, float]]] = [[] for _ in range(n_matches)]
    seen: set[tuple[int, int]] = set()
    for row in clean.iter_rows(named=True):
        mi = idx_of[row["match_key"]]
        rn = int(row["round_number"])
        assert (mi, rn) not in seen
        seen.add((mi, rn))
        per_match[mi].append((rn, float(row["ct_equipment_calc"]), float(row["t_equipment_calc"])))

    # ---- point estimates ----
    point: dict[int, dict] = {}
    for rn in ROUNDS:
        cts = [ct for mi in range(n_matches) for (r, ct, _) in per_match[mi] if r == rn]
        ts = [t for mi in range(n_matches) for (r, _, t) in per_match[mi] if r == rn]
        point[rn] = {
            "ct_mean": statistics.fmean(cts) if cts else None,
            "t_mean": statistics.fmean(ts) if ts else None,
            "n_rounds": len(cts),
            "n_matches": len(cts),
        }

    # ---- match-cluster bootstrap ----
    rng = random.Random(SEED)
    boot_ct: dict[int, list[float]] = {rn: [] for rn in ROUNDS}
    boot_t: dict[int, list[float]] = {rn: [] for rn in ROUNDS}
    boot_pool_ct: list[float] = []
    boot_pool_t: list[float] = []
    boot_pool_ct_np: list[float] = []  # pooled over rounds 1-24, excluding pistol rounds
    boot_pool_t_np: list[float] = []
    for _ in range(N_BOOT):
        draw = [rng.randrange(n_matches) for _ in range(n_matches)]
        s_ct = {rn: 0.0 for rn in ROUNDS}
        s_t = {rn: 0.0 for rn in ROUNDS}
        n_r = {rn: 0 for rn in ROUNDS}
        for mi in draw:
            for rn, ct, t in per_match[mi]:
                s_ct[rn] += ct
                s_t[rn] += t
                n_r[rn] += 1
        tot_ct = tot_t = 0.0
        tot_n = 0
        np_ct = np_t = 0.0
        np_n = 0
        for rn in ROUNDS:
            if n_r[rn] > 0:
                boot_ct[rn].append(s_ct[rn] / n_r[rn])
                boot_t[rn].append(s_t[rn] / n_r[rn])
                tot_ct += s_ct[rn]
                tot_t += s_t[rn]
                tot_n += n_r[rn]
                if rn not in PISTOL_ROUNDS:
                    np_ct += s_ct[rn]
                    np_t += s_t[rn]
                    np_n += n_r[rn]
        if tot_n > 0:
            boot_pool_ct.append(tot_ct / tot_n)
            boot_pool_t.append(tot_t / tot_n)
        if np_n > 0:
            boot_pool_ct_np.append(np_ct / np_n)
            boot_pool_t_np.append(np_t / np_n)

    def interval(samples: list[float], mean: float) -> dict:
        s = sorted(samples)
        lo = percentile(s, 0.025)
        hi = percentile(s, 0.975)
        return {
            "ci_lo": lo,
            "ci_hi": hi,
            "boot_se": statistics.stdev(s),
            "n_valid_resamples": len(s),
            "ordered": lo <= hi,
            "point_inside": lo <= mean <= hi,
        }

    # ---- assemble per-round series ----
    series: dict[str, list[dict]] = {"CT": [], "T": []}
    order_violations = 0
    for rn in ROUNDS:
        p = point[rn]
        if p["n_rounds"] == 0:
            continue
        for side, key, boot in (("CT", "ct_mean", boot_ct), ("T", "t_mean", boot_t)):
            iv = interval(boot[rn], p[key])
            if not iv["ordered"]:
                order_violations += 1
            series[side].append(
                {
                    "round_number": rn,
                    "mean": round(p[key], 2),
                    "ci_lo": round(iv["ci_lo"], 2),
                    "ci_hi": round(iv["ci_hi"], 2),
                    "boot_se": round(iv["boot_se"], 2),
                    "n_rounds": p["n_rounds"],
                    "n_valid_resamples": iv["n_valid_resamples"],
                    "point_inside_ci": iv["point_inside"],
                    "pistol": rn in PISTOL_ROUNDS,
                }
            )

    # ---- pooled key numbers ----
    pool_ct = sum(p["ct_mean"] * p["n_rounds"] for rn, p in point.items() if p["n_rounds"]) / sum(
        p["n_rounds"] for p in point.values()
    )
    pool_t = sum(p["t_mean"] * p["n_rounds"] for rn, p in point.items() if p["n_rounds"]) / sum(
        p["n_rounds"] for p in point.values()
    )
    np_rounds = [rn for rn in ROUNDS if rn not in PISTOL_ROUNDS and point[rn]["n_rounds"]]
    np_n = sum(point[rn]["n_rounds"] for rn in np_rounds)
    np_ct_mean = sum(point[rn]["ct_mean"] * point[rn]["n_rounds"] for rn in np_rounds) / np_n
    np_t_mean = sum(point[rn]["t_mean"] * point[rn]["n_rounds"] for rn in np_rounds) / np_n
    n_rounds_plotted = sum(point[rn]["n_rounds"] for rn in ROUNDS)
    assert n_rounds_plotted == clean.height

    pooled = {
        "CT_all_rounds_mean": {"mean": round(pool_ct, 2), **{k: round(v, 2) for k, v in interval(boot_pool_ct, pool_ct).items() if k in ("ci_lo", "ci_hi", "boot_se")}},
        "T_all_rounds_mean": {"mean": round(pool_t, 2), **{k: round(v, 2) for k, v in interval(boot_pool_t, pool_t).items() if k in ("ci_lo", "ci_hi", "boot_se")}},
        "CT_non_pistol_mean": {"mean": round(np_ct_mean, 2), **{k: round(v, 2) for k, v in interval(boot_pool_ct_np, np_ct_mean).items() if k in ("ci_lo", "ci_hi", "boot_se")}},
        "T_non_pistol_mean": {"mean": round(np_t_mean, 2), **{k: round(v, 2) for k, v in interval(boot_pool_t_np, np_t_mean).items() if k in ("ci_lo", "ci_hi", "boot_se")}},
    }
    ct_series = series["CT"]
    t_series = series["T"]
    key_numbers = {
        "n_rounds_plotted": n_rounds_plotted,
        "n_matches_sample": n_matches,
        "n_sample_rounds": table.height,
        "n_rounds_round_number_1_24_before_clean_filter": window.height,
        "n_overtime_rounds_excluded_from_plot": ot_rounds_excluded,
        "n_rounds_excluded_incomplete_side": incomplete.height,
        "excluded_incomplete_rounds": incomplete.select(
            ["match_key", "round", "round_number", "ct_alive", "t_alive"]
        ).rows(named=True),
        "pistol_round_points": {
            f"round_{rn}": {
                "CT": next(x for x in ct_series if x["round_number"] == rn),
                "T": next(x for x in t_series if x["round_number"] == rn),
            }
            for rn in PISTOL_ROUNDS
        },
        "highest_mean_CT": max(ct_series, key=lambda x: x["mean"]),
        "highest_mean_T": max(t_series, key=lambda x: x["mean"]),
        "lowest_mean_CT": min(ct_series, key=lambda x: x["mean"]),
        "lowest_mean_T": min(t_series, key=lambda x: x["mean"]),
        "pooled": pooled,
    }

    checks = {
        "sample_rounds": table.height,
        "first_snapshot_is_second_0_everywhere": bool((table["first_sec"] == 0).all()),
        "rows_equal_alive_counts": bool((table["n_rows"] == table["ct_alive"] + table["t_alive"]).all()),
        "rounds_with_zero_side_sum_in_plot": zero_side,
        "buy_table_cross_check_rounds": buy_checked,
        "buy_table_ct_sum_matches": buy_ct_match,
        "buy_table_t_sum_matches": buy_t_match,
        "interval_order_violations": order_violations,
        "points_outside_their_ci": sum(1 for s in (ct_series + t_series) if not s["point_inside_ci"]),
        "min_n_rounds_per_round": min(x["n_rounds"] for x in ct_series),
        "max_n_rounds_per_round": max(x["n_rounds"] for x in ct_series),
        "all_means_nonnegative": all(x["mean"] >= 0 for x in ct_series + t_series),
    }

    result = {
        "id": "S2",
        "title": "Mean team equipment value by round number, CT vs T",
        "measure": "sum of equipment_value_calc over alive players of the side at freeze end (first snapshot per match-round)",
        "side_rule": "side the team is on that round (team_code 3 = CT, 2 = T); sides swap after round 12",
        "population": "rounds with round_number 1-24 in the 1 Hz sample, both sides 5 alive at freeze end",
        "interval": "percentile 95% CI, match-cluster bootstrap: resample matches with replacement, 1000 resamples, random.Random(0), statistics recomputed per resample",
        "bootstrap": {"n_resamples": N_BOOT, "seed": SEED, "cluster": "match_key", "n_clusters": n_matches},
        "pistol_rounds": PISTOL_ROUNDS,
        "series": {"CT": ct_series, "T": t_series},
        "key_numbers": key_numbers,
        "checks": checks,
        "sources": [
            "data/state_sample/players_1hz.parquet",
            "data/state_sample/rounds.parquet",
            "data/state_sample/round_buy_types.parquet (cross-check only)",
        ],
    }
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # ---- SVG ----
    y_top_raw = max(x["ci_hi"] for x in ct_series + t_series)
    # smallest "nice" tick step that keeps the axis to at most 7 ticks
    step = next(s for s in (500, 1000, 2000, 5000, 10000) if math.ceil(y_top_raw / s) <= 7)
    y_max = math.ceil(y_top_raw / step) * step

    def sy(v: float) -> float:
        return PB - v / y_max * (PB - PT)

    parts: list[str] = []
    parts.append(
        f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif" role="img" '
        f'aria-label="Mean team equipment value at freeze end by round number, CT and T">'
    )
    parts.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')

    # pistol bands (behind data)
    slot = (PR - PL) / (X_HI - X_LO)
    for rn in PISTOL_ROUNDS:
        parts.append(
            f'<rect x="{sx(rn) - slot / 2:.1f}" y="{PT:.1f}" width="{slot:.1f}" height="{PB - PT:.1f}" '
            f'fill="{COL_NEUTRAL}" fill-opacity="0.08"/>'
        )
        parts.append(
            f'<text x="{sx(rn):.1f}" y="{PT - 8:.1f}" font-size="12" text-anchor="middle" fill="{COL_NEUTRAL}">pistol</text>'
        )

    # y gridlines and tick labels
    for v in range(0, int(y_max) + 1, step):
        y = sy(v)
        if v > 0:
            parts.append(f'<line x1="{PL:.1f}" y1="{y:.1f}" x2="{PR:.1f}" y2="{y:.1f}" stroke="{COL_GRID}" stroke-width="1"/>')
        parts.append(
            f'<text x="{PL - 8:.1f}" y="{y + 4:.1f}" font-size="12" text-anchor="end" fill="{COL_NEUTRAL}">{fmt_int(v)}</text>'
        )

    # axes
    parts.append(f'<line x1="{PL:.1f}" y1="{PT:.1f}" x2="{PL:.1f}" y2="{PB:.1f}" stroke="{COL_NEUTRAL}" stroke-width="1"/>')
    parts.append(f'<line x1="{PL:.1f}" y1="{PB:.1f}" x2="{PR:.1f}" y2="{PB:.1f}" stroke="{COL_NEUTRAL}" stroke-width="1"/>')

    # x ticks and labels (every round)
    for rn in ROUNDS:
        x = sx(rn)
        parts.append(f'<line x1="{x:.1f}" y1="{PB:.1f}" x2="{x:.1f}" y2="{PB + 4:.1f}" stroke="{COL_NEUTRAL}" stroke-width="1"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{PB + 18:.1f}" font-size="12" text-anchor="middle" fill="{COL_NEUTRAL}">{rn}</text>'
        )

    # CI bands and lines
    def band(points: list[dict], colour: str) -> str:
        upper = [f"{sx(p['round_number']):.1f},{sy(p['ci_hi']):.1f}" for p in points]
        lower = [f"{sx(p['round_number']):.1f},{sy(p['ci_lo']):.1f}" for p in reversed(points)]
        return f'<polygon points="{" ".join(upper + lower)}" fill="{colour}" fill-opacity="0.16" stroke="none"/>'

    def line(points: list[dict], colour: str) -> str:
        pts = " ".join(f"{sx(p['round_number']):.1f},{sy(p['mean']):.1f}" for p in points)
        out = [f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="2.5" stroke-linejoin="round"/>']
        for p in points:
            out.append(
                f'<circle cx="{sx(p["round_number"]):.1f}" cy="{sy(p["mean"]):.1f}" r="3" fill="{colour}"/>'
            )
        return "".join(out)

    parts.append(band(ct_series, COL_CT))
    parts.append(band(t_series, COL_T))
    parts.append(line(ct_series, COL_CT))
    parts.append(line(t_series, COL_T))

    # y tick labels already drawn; y axis title (rotated)
    parts.append(
        f'<text transform="translate(18 {(PT + PB) / 2:.1f}) rotate(-90)" font-size="13" text-anchor="middle" '
        f'fill="{COL_NEUTRAL}">Mean team equipment value at freeze end ($)</text>'
    )
    parts.append(
        f'<text x="{(PL + PR) / 2:.1f}" y="{PB + 40:.1f}" font-size="13" text-anchor="middle" fill="{COL_NEUTRAL}">'
        f"Round number (sides swap after round 12)</text>"
    )

    # legend (top right)
    lx = PR - 170
    ly = 22
    parts.append(f'<line x1="{lx:.1f}" y1="{ly}" x2="{lx + 22:.1f}" y2="{ly}" stroke="{COL_CT}" stroke-width="2.5"/>')
    parts.append(f'<circle cx="{lx + 11:.1f}" cy="{ly}" r="3" fill="{COL_CT}"/>')
    parts.append(f'<text x="{lx + 30:.1f}" y="{ly + 4}" font-size="13" fill="{COL_NEUTRAL}">CT</text>')
    lx2 = lx + 70
    parts.append(f'<line x1="{lx2:.1f}" y1="{ly}" x2="{lx2 + 22:.1f}" y2="{ly}" stroke="{COL_T}" stroke-width="2.5"/>')
    parts.append(f'<circle cx="{lx2 + 11:.1f}" cy="{ly}" r="3" fill="{COL_T}"/>')
    parts.append(f'<text x="{lx2 + 30:.1f}" y="{ly + 4}" font-size="13" fill="{COL_NEUTRAL}">T</text>')

    # sample size and interval note (footer)
    parts.append(
        f'<text x="{PL:.1f}" y="{PB + 62:.1f}" font-size="12" fill="{COL_NEUTRAL}">'
        f"n = {n_rounds_plotted:,} rounds, {n_matches} matches (rounds 1-24, both sides 5 alive at freeze end)</text>"
    )
    parts.append(
        f'<text x="{PL:.1f}" y="{PB + 78:.1f}" font-size="12" fill="{COL_NEUTRAL}">'
        f"Bands: 95% match-cluster bootstrap CI (1,000 resamples, seed 0). Grey columns: pistol rounds.</text>"
    )
    parts.append("</svg>")

    OUT_SVG.write_text("\n".join(parts) + "\n", encoding="utf-8")

    print(json.dumps({"n_rounds_plotted": n_rounds_plotted, "y_max": y_max, "checks": checks}, indent=2))
    print("CT:", [(x["round_number"], x["mean"], x["ci_lo"], x["ci_hi"], x["n_rounds"]) for x in ct_series])
    print("T:", [(x["round_number"], x["mean"], x["ci_lo"], x["ci_hi"], x["n_rounds"]) for x in t_series])


if __name__ == "__main__":
    main()
