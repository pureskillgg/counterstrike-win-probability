"""S4: higher-equipment-side win rate by round class, with match-cluster bootstrap intervals.

Inputs : data/state_sample/round_buy_types.parquet (rounds), data/state_sample/rounds.parquet (coverage check)
Outputs: data/viz/S4.json (plotted series and key numbers), data/viz/S4.svg (chart)
Run    : G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\viz\\S4.py

Definitions
- The higher-equipment side is the side with more freeze-end equipment value (higher_side in the source).
- Ties (equal equipment value) have no higher side and are excluded from the rate (both numerator and denominator).
- Win rate = higher-side wins / decided rounds, pooled over the class (ratio of sums).
- Interval: match-cluster bootstrap. Resample the 150 match_key clusters with replacement, 1000 times,
  random.Random(0). Percentile 95% interval (2.5th and 97.5th percentile, linear interpolation).
"""
import json
import math
import random
import statistics
from pathlib import Path

import polars as pl

ROOT = Path(r"G:\pureskill\csds-cs2")
SRC_RBT = ROOT / "data" / "state_sample" / "round_buy_types.parquet"
SRC_ROUNDS = ROOT / "data" / "state_sample" / "rounds.parquet"
OUT_JSON = ROOT / "data" / "viz" / "S4.json"
OUT_SVG = ROOT / "data" / "viz" / "S4.svg"

CLASS_ORDER = [
    "pistol",
    "eco_vs_eco",
    "anti_eco",
    "force_vs_eco",
    "full_vs_force",
    "force_vs_force",
    "full_vs_full",
]
N_BOOT = 1000
SEED = 0
SMALL_N = 30  # classes with fewer decided rounds than this are drawn dashed and labelled too small


def load_and_check():
    rbt = (
        pl.scan_parquet(SRC_RBT)
        .select(["match_key", "round", "round_class", "higher_side", "higher_side_won",
                 "ct_win", "ct_equipment", "t_equipment"])
        .collect()
    )
    rounds_keys = pl.scan_parquet(SRC_ROUNDS).select(["match_key", "round"]).collect()

    checks = {}
    checks["rows_round_buy_types"] = rbt.height
    checks["rows_rounds_parquet"] = rounds_keys.height
    missing = rounds_keys.join(rbt.select(["match_key", "round"]), on=["match_key", "round"], how="anti")
    checks["rounds_in_rounds_parquet_missing_from_round_buy_types"] = missing.height
    checks["missing_examples"] = [
        f"{r['match_key']} round {r['round']}" for r in missing.head(3).iter_rows(named=True)
    ]
    checks["duplicate_match_round_rows"] = rbt.height - rbt.unique(["match_key", "round"]).height
    checks["unknown_round_classes"] = sorted(set(rbt["round_class"].unique().to_list()) - set(CLASS_ORDER))
    assert not checks["unknown_round_classes"], checks["unknown_round_classes"]
    assert checks["duplicate_match_round_rows"] == 0

    # higher_side must follow equipment value
    expected_side = (
        pl.when(pl.col("ct_equipment") > pl.col("t_equipment")).then(pl.lit("CT"))
        .when(pl.col("ct_equipment") < pl.col("t_equipment")).then(pl.lit("T"))
        .otherwise(pl.lit("tie"))
    )
    side_mismatch = rbt.filter(expected_side != pl.col("higher_side")).height
    checks["higher_side_vs_equipment_mismatches"] = side_mismatch
    assert side_mismatch == 0

    # higher_side_won must agree with ct_win for decided rounds
    decided_rows = rbt.filter(pl.col("higher_side") != "tie")
    exp_won = (
        pl.when(pl.col("higher_side") == "CT").then(pl.col("ct_win") == 1)
        .otherwise(pl.col("ct_win") == 0)
    )
    won_mismatch = decided_rows.filter(exp_won != pl.col("higher_side_won")).height
    checks["higher_side_won_vs_ct_win_mismatches"] = won_mismatch
    assert won_mismatch == 0

    # nulls in higher_side_won must be exactly the ties
    null_won = rbt.filter(pl.col("higher_side_won").is_null())
    tie_rows = rbt.filter(pl.col("higher_side") == "tie")
    checks["null_higher_side_won"] = null_won.height
    checks["ties"] = tie_rows.height
    assert null_won.height == tie_rows.height == null_won.filter(pl.col("higher_side") == "tie").height

    rbt = rbt.with_columns(
        win=pl.col("higher_side_won").fill_null(False).cast(pl.Int64),
        dec=(pl.col("higher_side") != "tie").cast(pl.Int64),
        tie=(pl.col("higher_side") == "tie").cast(pl.Int64),
    )
    return rbt, checks


def class_stats(rbt):
    g = rbt.group_by("round_class").agg(
        pl.len().alias("n_rounds"),
        pl.col("tie").sum().alias("n_ties"),
        pl.col("dec").sum().alias("n_decided"),
        pl.col("win").sum().alias("wins"),
        pl.col("match_key").n_unique().alias("n_matches"),
    )
    out = {}
    for r in g.iter_rows(named=True):
        out[r["round_class"]] = {k: int(v) for k, v in r.items() if k != "round_class"}
    for c in CLASS_ORDER:
        out.setdefault(c, {"n_rounds": 0, "n_ties": 0, "n_decided": 0, "wins": 0, "n_matches": 0})
        assert out[c]["n_rounds"] == out[c]["n_decided"] + out[c]["n_ties"]
        assert 0 <= out[c]["wins"] <= out[c]["n_decided"]
    return out


def match_class_table(rbt):
    g = rbt.group_by(["match_key", "round_class"]).agg(
        pl.col("win").sum().alias("w"),
        pl.col("dec").sum().alias("d"),
    )
    return {(r["match_key"], r["round_class"]): (int(r["w"]), int(r["d"])) for r in g.iter_rows(named=True)}


def quantile(values, q):
    s = sorted(values)
    pos = q * (len(s) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def bootstrap(matches, pm):
    """Match-cluster bootstrap: resample matches with replacement, pool wins and decided rounds."""
    rng = random.Random(SEED)
    m = len(matches)
    keys = CLASS_ORDER + ["overall"]
    draws = {k: [] for k in keys}
    zero_decided = {k: 0 for k in keys}
    for _ in range(N_BOOT):
        wins = dict.fromkeys(CLASS_ORDER, 0)
        dec = dict.fromkeys(CLASS_ORDER, 0)
        for _i in range(m):
            mk = matches[rng.randrange(m)]
            for c in CLASS_ORDER:
                w, d = pm.get((mk, c), (0, 0))
                wins[c] += w
                dec[c] += d
        for c in CLASS_ORDER:
            if dec[c] == 0:
                zero_decided[c] += 1
            else:
                draws[c].append(wins[c] / dec[c])
        tot_d = sum(dec.values())
        draws["overall"].append(sum(wins.values()) / tot_d)
    return draws, zero_decided


def build_series(stats, draws, zero_decided):
    series = []
    for c in CLASS_ORDER:
        s = stats[c]
        rate = s["wins"] / s["n_decided"]
        d = draws[c]
        lo = quantile(d, 0.025)
        hi = quantile(d, 0.975)
        assert 0.0 <= lo <= hi <= 1.0, (c, lo, hi)
        series.append({
            "round_class": c,
            "n_rounds": s["n_rounds"],
            "n_ties_excluded": s["n_ties"],
            "n_decided": s["n_decided"],
            "n_matches": s["n_matches"],
            "higher_side_wins": s["wins"],
            "win_rate": rate,
            "ci95_low": lo,
            "ci95_high": hi,
            "point_inside_ci": lo <= rate <= hi,
            "bootstrap_se": statistics.stdev(d),
            "bootstrap_valid_resamples": len(d),
            "bootstrap_resamples_with_zero_decided": zero_decided[c],
            "small_n": s["n_decided"] < SMALL_N,
        })
    return series


def overall_block(stats, draws):
    n_rounds = sum(stats[c]["n_rounds"] for c in CLASS_ORDER)
    n_dec = sum(stats[c]["n_decided"] for c in CLASS_ORDER)
    wins = sum(stats[c]["wins"] for c in CLASS_ORDER)
    d = draws["overall"]
    return {
        "n_rounds": n_rounds,
        "n_decided": n_dec,
        "n_ties_excluded": n_rounds - n_dec,
        "higher_side_wins": wins,
        "win_rate": wins / n_dec,
        "ci95_low": quantile(d, 0.025),
        "ci95_high": quantile(d, 0.975),
    }


# ----------------------------------------------------------------- SVG
W, H = 800, 450
X0, X1 = 200.0, 600.0
ROW_TOP, ROW_H = 52.0, 44.0
TICK_PCT = [0, 20, 40, 60, 80, 100]
C_NEUTRAL = "#475569"
C_GRID = "#e2e8f0"
C_ACCENT = "#059669"
C_INK = "#1e293b"


def xpos(pct):
    return X0 + (pct / 100.0) * (X1 - X0)


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size, anchor="start", fill=C_INK, weight="normal"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
            f'fill="{fill}" font-weight="{weight}">{esc(s)}</text>')


def build_svg(series, overall, n_total, n_decided, n_matches, n_ties):
    axis_y = ROW_TOP + ROW_H * len(series)
    p = []
    p.append(
        f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif" '
        f'role="img" aria-label="Higher-equipment-side win rate by round class with 95% bootstrap intervals">'
    )
    p.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')

    # sample-size label
    p.append(text(20, 22, f"n = {n_decided:,} decided rounds of {n_total:,} ({n_matches} matches; "
                          f"{n_ties} ties excluded)", 12, fill=C_NEUTRAL))

    # gridlines and x ticks
    for pct in range(0, 101, 10):
        x = xpos(pct)
        p.append(f'<line x1="{x:.1f}" y1="{ROW_TOP - 8:.1f}" x2="{x:.1f}" y2="{axis_y:.1f}" '
                 f'stroke="{C_GRID}" stroke-width="1"/>')
    for pct in TICK_PCT:
        x = xpos(pct)
        p.append(f'<line x1="{x:.1f}" y1="{axis_y:.1f}" x2="{x:.1f}" y2="{axis_y + 4:.1f}" '
                 f'stroke="{C_NEUTRAL}" stroke-width="1"/>')
        p.append(text(x, axis_y + 18, f"{pct}%", 12, anchor="middle", fill=C_NEUTRAL))

    # x axis line
    p.append(f'<line x1="{X0:.1f}" y1="{axis_y:.1f}" x2="{X1:.1f}" y2="{axis_y:.1f}" '
             f'stroke="{C_NEUTRAL}" stroke-width="1.2"/>')

    # 50% reference line
    x50 = xpos(50)
    p.append(f'<line x1="{x50:.1f}" y1="{ROW_TOP - 10:.1f}" x2="{x50:.1f}" y2="{axis_y:.1f}" '
             f'stroke="{C_ACCENT}" stroke-width="1.6" stroke-dasharray="5 4"/>')
    p.append(text(x50, ROW_TOP - 14, "50% = no edge", 12, anchor="middle", fill=C_ACCENT, weight="bold"))

    # rows
    for i, s in enumerate(series):
        cy = ROW_TOP + ROW_H * (i + 0.5)
        rate_pct = s["win_rate"] * 100
        lo_pct = s["ci95_low"] * 100
        hi_pct = s["ci95_high"] * 100
        xr, xl, xh = xpos(rate_pct), xpos(lo_pct), xpos(hi_pct)

        # class label and n
        p.append(text(188, cy - 2, s["round_class"], 13, anchor="end"))
        n_label = f"n = {s['n_decided']:,}" + (" (too few)" if s["small_n"] else "")
        p.append(text(188, cy + 13, n_label, 12, anchor="end", fill=C_NEUTRAL))

        # bar (tinted; dashed outline when too small)
        bar_w = max(xr - X0, 0.0)
        if s["small_n"]:
            p.append(f'<rect x="{X0:.1f}" y="{cy - 10:.1f}" width="{bar_w:.1f}" height="20" fill="none" '
                     f'stroke="{C_NEUTRAL}" stroke-width="1" stroke-dasharray="3 3"/>')
        else:
            p.append(f'<rect x="{X0:.1f}" y="{cy - 10:.1f}" width="{bar_w:.1f}" height="20" '
                     f'fill="{C_NEUTRAL}" fill-opacity="0.22"/>')

        # bootstrap whisker
        p.append(f'<line x1="{xl:.1f}" y1="{cy:.1f}" x2="{xh:.1f}" y2="{cy:.1f}" stroke="{C_NEUTRAL}" '
                 f'stroke-width="2"/>')
        for xc in (xl, xh):
            p.append(f'<line x1="{xc:.1f}" y1="{cy - 6:.1f}" x2="{xc:.1f}" y2="{cy + 6:.1f}" '
                     f'stroke="{C_NEUTRAL}" stroke-width="2"/>')

        # point estimate
        p.append(f'<circle cx="{xr:.1f}" cy="{cy:.1f}" r="4.5" fill="{C_NEUTRAL}"/>')

        # value label
        p.append(
            f'<text x="616" y="{cy + 4:.1f}" font-size="13" fill="{C_INK}">'
            f'<tspan font-weight="bold">{rate_pct:.1f}%</tspan>'
            f'<tspan font-size="12" fill="{C_NEUTRAL}"> [{lo_pct:.1f}, {hi_pct:.1f}]</tspan></text>'
        )

    # axis title and footnotes
    p.append(text((X0 + X1) / 2, axis_y + 42, "Win rate of the higher-equipment side (% of decided rounds)",
                  13, anchor="middle"))
    p.append(text(20, axis_y + 64,
                  f"Whiskers: 95% match-cluster bootstrap interval ({n_matches} matches resampled, "
                  f"{N_BOOT:,} resamples, seed {SEED}; percentile method).", 11, fill=C_NEUTRAL))
    p.append(text(20, axis_y + 80,
                  "Higher side = more freeze-end equipment value. n = decided rounds (ties excluded). "
                  "eco_vs_eco (n = 8) is too few to read.", 11, fill=C_NEUTRAL))
    p.append("</svg>")
    return "\n".join(p) + "\n"


def main():
    rbt, checks = load_and_check()
    stats = class_stats(rbt)
    matches = sorted(rbt["match_key"].unique().to_list())
    pm = match_class_table(rbt)
    draws, zero_decided = bootstrap(matches, pm)
    series = build_series(stats, draws, zero_decided)
    overall = overall_block(stats, draws)

    n_total = rbt.height
    n_decided = overall["n_decided"]
    n_ties = overall["n_ties_excluded"]
    n_matches = len(matches)
    assert sum(s["n_rounds"] for s in series) == n_total, "class counts do not sum to total rounds"
    assert n_decided + n_ties == n_total

    svg = build_svg(series, overall, n_total, n_decided, n_matches, n_ties)
    OUT_SVG.write_text(svg, encoding="utf-8")

    payload = {
        "id": "S4",
        "title": "Higher-equipment-side win rate by round class",
        "metric": "higher-equipment side win rate = higher-side wins / decided rounds (ties excluded)",
        "source": "data/state_sample/round_buy_types.parquet (1 Hz sample, 150 matches, 2026-10 days 01-07)",
        "interval": {
            "method": "match-cluster bootstrap, percentile 95%",
            "unit_resampled": "match_key",
            "clusters": n_matches,
            "resamples": N_BOOT,
            "seed": SEED,
            "rng": "random.Random(SEED).randrange (stdlib)",
            "statistic": "pooled ratio of sums over resampled matches",
        },
        "sample": {
            "rounds_total": n_total,
            "rounds_decided": n_decided,
            "ties_excluded": n_ties,
            "matches": n_matches,
            "rounds_parquet_rows": checks["rows_rounds_parquet"],
        },
        "checks": checks,
        "overall": overall,
        "series": series,
        "plot_order": CLASS_ORDER,
        "reference_line": 0.5,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"rounds total {n_total}, decided {n_decided}, ties {n_ties}, matches {n_matches}")
    print(f"overall {overall['win_rate']:.4f} [{overall['ci95_low']:.4f}, {overall['ci95_high']:.4f}]")
    for s in series:
        print(f"{s['round_class']:<15} n_rounds={s['n_rounds']:>5} decided={s['n_decided']:>5} "
              f"ties={s['n_ties_excluded']:>2} matches={s['n_matches']:>3} "
              f"rate={s['win_rate']:.4f} [{s['ci95_low']:.4f}, {s['ci95_high']:.4f}] "
              f"se={s['bootstrap_se']:.4f} valid={s['bootstrap_valid_resamples']} "
              f"zero_dec={s['bootstrap_resamples_with_zero_decided']}")
    print("checks", json.dumps({k: v for k, v in checks.items()}, indent=None))
    print("wrote", OUT_JSON)
    print("wrote", OUT_SVG)


if __name__ == "__main__":
    main()
