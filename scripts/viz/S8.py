"""S8: CT round win rate by decile of team money difference (non-pistol rounds).

Money difference = CT team money minus T team money, summed over the first
snapshot of each round in the 1 Hz sample (the freeze-end state). Equipment
difference is the same sum over equipment_value_calc.

Reads (read only):
  data/state_sample/players_1hz.parquet
  data/state_sample/rounds.parquet
  data/state_sample/round_buy_types.parquet
Writes:
  data/viz/S8.json
  data/viz/S8.svg

Run with the project interpreter:
  G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\viz\\S8.py

Intervals: match-cluster bootstrap. Resample the 150 matches with replacement,
1,000 resamples, numpy default_rng(0). Each resample weights every round by the
number of times its match was drawn. Percentile 95% intervals.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path("G:/pureskill/csds-cs2")
SAMPLE = ROOT / "data" / "state_sample"
OUT_JSON = ROOT / "data" / "viz" / "S8.json"
OUT_SVG = ROOT / "data" / "viz" / "S8.svg"

N_BOOT = 1000
SEED = 0
N_DEC = 10
PISTOL_ROUNDS = (1, 13)

CT_COLOR = "#2563eb"
NEUTRAL = "#475569"
GRID = "#e2e8f0"

W, H = 800, 450
X0, X1 = 90, 776
Y0, Y1 = 40, 336


# ---------------------------------------------------------------- data

def freeze_end_team_totals() -> pl.DataFrame:
    """Team money and equipment totals at each round's first snapshot."""
    players = (
        pl.scan_parquet(SAMPLE / "players_1hz.parquet")
        .select(["match_key", "round", "tick", "team_code", "money", "equipment_value_calc"])
        .collect()
    )
    t0 = players.group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("t0"))
    fe = players.join(t0, on=["match_key", "round"]).filter(pl.col("tick") == pl.col("t0"))
    return fe.group_by(["match_key", "round"]).agg(
        pl.col("money").filter(pl.col("team_code") == 3).sum().alias("ct_money_1hz"),
        pl.col("money").filter(pl.col("team_code") == 2).sum().alias("t_money_1hz"),
        pl.col("equipment_value_calc").filter(pl.col("team_code") == 3).sum().alias("ct_equip_1hz"),
        pl.col("equipment_value_calc").filter(pl.col("team_code") == 2).sum().alias("t_equip_1hz"),
        (pl.col("team_code") == 3).sum().alias("n_ct"),
        (pl.col("team_code") == 2).sum().alias("n_t"),
        pl.col("t0").first().alias("freeze_tick"),
    )


# ---------------------------------------------------------------- stats

def weighted_auc(inv: np.ndarray, n_levels: int, y: np.ndarray, w: np.ndarray) -> float:
    """AUC of score -> y, with ties counted as 0.5, via unique-score bins.

    inv is the index of each row's score among the sorted unique scores, so the
    score ordering never changes across bootstrap resamples.
    """
    pos = np.bincount(inv, weights=w * y, minlength=n_levels)
    neg = np.bincount(inv, weights=w * (1.0 - y), minlength=n_levels)
    p_tot = pos.sum()
    n_tot = neg.sum()
    if p_tot <= 0 or n_tot <= 0:
        return float("nan")
    neg_below = np.cumsum(neg) - neg
    return float(np.sum(pos * (neg_below + 0.5 * neg)) / (p_tot * n_tot))


def pct_ci(arr: np.ndarray):
    lo, hi = np.nanpercentile(arr, [2.5, 97.5], axis=0)
    return lo, hi


# ---------------------------------------------------------------- svg

def px(i: int) -> float:
    return X0 + (i + 0.5) * (X1 - X0) / N_DEC


def py(rate: float) -> float:
    return Y1 - rate * (Y1 - Y0)


def fmt_k(v: float) -> str:
    k = v / 1000.0
    sign = "-" if k < 0 else "+"
    return f"{sign}${abs(k):.1f}k"


def txt(x, y, s, size=12, anchor="middle", color=NEUTRAL, extra=""):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
        f'fill="{color}"{extra}>{s}</text>'
    )


def build_svg(series, n_analysis, n_matches, auc_m, auc_e):
    p = []
    p.append(f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">')
    p.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')

    for t in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        y = py(t)
        p.append(f'<line x1="{X0}" y1="{y:.1f}" x2="{X1}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        p.append(txt(X0 - 8, y + 4, f"{int(round(t * 100))}%", anchor="end"))

    # 50% reference line (dashed, neutral)
    y50 = py(0.5)
    p.append(
        f'<line x1="{X0}" y1="{y50:.1f}" x2="{X1}" y2="{y50:.1f}" stroke="{NEUTRAL}" '
        f'stroke-width="1" stroke-dasharray="4 4"/>'
    )

    p.append(f'<line x1="{X0}" y1="{Y0}" x2="{X0}" y2="{Y1}" stroke="{NEUTRAL}" stroke-width="1.2"/>')
    p.append(f'<line x1="{X0}" y1="{Y1}" x2="{X1}" y2="{Y1}" stroke="{NEUTRAL}" stroke-width="1.2"/>')

    # series line
    pts = " ".join(f"{px(i):.1f},{py(s['ct_win_rate']):.1f}" for i, s in enumerate(series))
    p.append(f'<polyline points="{pts}" fill="none" stroke="{CT_COLOR}" stroke-width="2"/>')

    # error bars (match-cluster bootstrap 95%) and points
    for i, s in enumerate(series):
        x = px(i)
        y_lo = py(s["ci95_lo"])
        y_hi = py(s["ci95_hi"])
        p.append(f'<line x1="{x:.1f}" y1="{y_hi:.1f}" x2="{x:.1f}" y2="{y_lo:.1f}" stroke="{CT_COLOR}" stroke-width="1.5"/>')
        p.append(f'<line x1="{x - 6:.1f}" y1="{y_hi:.1f}" x2="{x + 6:.1f}" y2="{y_hi:.1f}" stroke="{CT_COLOR}" stroke-width="1.5"/>')
        p.append(f'<line x1="{x - 6:.1f}" y1="{y_lo:.1f}" x2="{x + 6:.1f}" y2="{y_lo:.1f}" stroke="{CT_COLOR}" stroke-width="1.5"/>')
        p.append(f'<circle cx="{x:.1f}" cy="{py(s["ct_win_rate"]):.1f}" r="5" fill="{CT_COLOR}"/>')

    # x tick labels: decile number, median money difference, n
    for i, s in enumerate(series):
        x = px(i)
        p.append(txt(x, Y1 + 18, str(i + 1)))
        p.append(txt(x, Y1 + 34, fmt_k(s["diff_median_usd"])))
        p.append(txt(x, Y1 + 50, f"n={s['n_rounds']:,}"))
    p.append(txt(X0 - 4, Y1 + 34, "median", anchor="end", size=11))
    p.append(txt(X0 - 4, Y1 + 50, "rounds", anchor="end", size=11))

    p.append(
        txt(
            (X0 + X1) / 2,
            Y1 + 72,
            "Decile of team money difference, CT minus T (1 = most T-favoured, 10 = most CT-favoured)",
            size=13,
        )
    )
    p.append(
        f'<text transform="translate(22,{(Y0 + Y1) / 2:.1f}) rotate(-90)" font-size="13" '
        f'text-anchor="middle" fill="{NEUTRAL}">CT round win rate</text>'
    )

    # sample size label
    p.append(
        txt(
            X1 - 6,
            Y0 + 16,
            f"n = {n_analysis:,} non-pistol rounds, {n_matches} matches",
            size=12,
            anchor="end",
        )
    )

    # inset: AUC of money difference alone and equipment difference alone
    bx, by, bw, bh = X0 + 12, Y0 + 10, 300, 70
    p.append(
        f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" rx="4" fill="#ffffff" stroke="{GRID}" stroke-width="1"/>'
    )
    lines = [
        f"AUC, money difference alone: {auc_m['point']:.3f}",
        f"95% match-cluster CI {auc_m['ci95_lo']:.3f} to {auc_m['ci95_hi']:.3f}",
        f"AUC, equipment difference alone: {auc_e['point']:.3f}",
        f"95% match-cluster CI {auc_e['ci95_lo']:.3f} to {auc_e['ci95_hi']:.3f}",
    ]
    for k, line in enumerate(lines):
        p.append(txt(bx + 10, by + 18 + k * 14, line, size=12, anchor="start"))

    p.append(
        txt(
            X0,
            436,
            "Points: CT win rate per decile. Bars: 95% match-cluster bootstrap CI (1,000 resamples, seed 0).",
            size=11,
            anchor="start",
        )
    )
    p.append("</svg>")
    return "\n".join(p) + "\n"


# ---------------------------------------------------------------- main

def main():
    rounds = pl.read_parquet(
        SAMPLE / "rounds.parquet",
        columns=["match_key", "round", "round_number", "ct_win", "split"],
    )
    buy = pl.read_parquet(
        SAMPLE / "round_buy_types.parquet",
        columns=["match_key", "round", "is_pistol", "ct_money", "t_money", "ct_equipment", "t_equipment"],
    )
    team = freeze_end_team_totals()

    df = (
        rounds.join(team, on=["match_key", "round"], how="left")
        .join(buy, on=["match_key", "round"], how="left")
        .with_columns(
            pl.col("round").is_in(list(PISTOL_ROUNDS)).alias("pistol_rule"),
            ((pl.col("n_ct") == 5) & (pl.col("n_t") == 5)).alias("full_5v5"),
        )
    )

    n_total = df.height
    n_pistol = df.filter(pl.col("pistol_rule")).height
    excluded_df = df.filter(~pl.col("full_5v5"))
    excluded = excluded_df.select(["match_key", "round", "n_ct", "n_t", "ct_win"]).to_dicts()
    buy_pistol_mismatch = df.filter(
        pl.col("is_pistol").is_not_null() & (pl.col("is_pistol") != pl.col("pistol_rule"))
    ).height
    buy_missing_rows = df.filter(pl.col("is_pistol").is_null()).height

    analysis = (
        df.filter(~pl.col("pistol_rule") & pl.col("full_5v5") & pl.col("ct_win").is_not_null())
        .sort(["match_key", "round"])
    )
    n_analysis = analysis.height

    # cross-check the 1 Hz freeze-end sums against the round_buy_types totals
    chk = analysis.filter(pl.col("ct_money").is_not_null())
    xchk_ct_money_max_abs = int(
        chk.select((pl.col("ct_money_1hz").cast(pl.Int64) - pl.col("ct_money").cast(pl.Int64)).abs().max()).item()
    )
    xchk_t_money_max_abs = int(
        chk.select((pl.col("t_money_1hz").cast(pl.Int64) - pl.col("t_money").cast(pl.Int64)).abs().max()).item()
    )
    xchk_ct_equip_max_abs = int(
        chk.select((pl.col("ct_equip_1hz").cast(pl.Int64) - pl.col("ct_equipment").cast(pl.Int64)).abs().max()).item()
    )
    xchk_t_equip_max_abs = int(
        chk.select((pl.col("t_equip_1hz").cast(pl.Int64) - pl.col("t_equipment").cast(pl.Int64)).abs().max()).item()
    )

    y = analysis["ct_win"].to_numpy().astype(np.float64)
    money_diff = (
        analysis["ct_money_1hz"].cast(pl.Float64) - analysis["t_money_1hz"].cast(pl.Float64)
    ).to_numpy().astype(np.float64)
    equip_diff = (
        analysis["ct_equip_1hz"].cast(pl.Float64) - analysis["t_equip_1hz"].cast(pl.Float64)
    ).to_numpy().astype(np.float64)

    match_keys = sorted(analysis["match_key"].unique().to_list())
    midx = {k: i for i, k in enumerate(match_keys)}
    match_id = np.array([midx[k] for k in analysis["match_key"].to_list()], dtype=np.int64)
    M = len(match_keys)
    n_matches = M

    # deciles of money difference, fixed on the analysis sample
    qs = np.arange(1, N_DEC) / N_DEC
    edges = np.quantile(money_diff, qs)
    dec = np.searchsorted(edges, money_diff, side="right")  # 0..9

    # score index for AUC (fixed ordering)
    u_m, inv_m = np.unique(money_diff, return_inverse=True)
    u_e, inv_e = np.unique(equip_diff, return_inverse=True)
    inv_m = inv_m.astype(np.int64)
    inv_e = inv_e.astype(np.int64)

    ones = np.ones(n_analysis, dtype=np.float64)
    n_d = np.bincount(dec, minlength=N_DEC).astype(np.float64)
    w_d = np.bincount(dec, weights=y, minlength=N_DEC)
    rate = w_d / n_d
    overall_rate = float(y.mean())
    auc_m_point = weighted_auc(inv_m, len(u_m), y, ones)
    auc_e_point = weighted_auc(inv_e, len(u_e), y, ones)

    # match-cluster bootstrap
    rng = np.random.default_rng(SEED)
    boot_rate = np.full((N_BOOT, N_DEC), np.nan)
    boot_overall = np.full(N_BOOT, np.nan)
    boot_auc_m = np.full(N_BOOT, np.nan)
    boot_auc_e = np.full(N_BOOT, np.nan)
    for b in range(N_BOOT):
        draw = rng.integers(0, M, size=M)
        cnt = np.bincount(draw, minlength=M)
        w = cnt[match_id].astype(np.float64)
        nd = np.bincount(dec, weights=w, minlength=N_DEC)
        wy = np.bincount(dec, weights=w * y, minlength=N_DEC)
        with np.errstate(invalid="ignore", divide="ignore"):
            boot_rate[b] = np.where(nd > 0, wy / np.where(nd > 0, nd, 1.0), np.nan)
        boot_overall[b] = float((w * y).sum() / w.sum())
        boot_auc_m[b] = weighted_auc(inv_m, len(u_m), y, w)
        boot_auc_e[b] = weighted_auc(inv_e, len(u_e), y, w)

    lo_rate, hi_rate = pct_ci(boot_rate)
    lo_ov, hi_ov = pct_ci(boot_overall)
    lo_am, hi_am = pct_ci(boot_auc_m)
    lo_ae, hi_ae = pct_ci(boot_auc_e)

    # sklearn cross-check on point AUCs (not used for the chart)
    sk_m = sk_e = None
    try:
        from sklearn.metrics import roc_auc_score

        sk_m = float(roc_auc_score(y, money_diff))
        sk_e = float(roc_auc_score(y, equip_diff))
    except Exception as exc:  # pragma: no cover
        sk_m = f"unavailable: {exc}"

    series = []
    for d in range(N_DEC):
        sel = dec == d
        dd = money_diff[sel]
        series.append(
            {
                "decile": d + 1,
                "n_rounds": int(n_d[d]),
                "ct_wins": int(w_d[d]),
                "ct_win_rate": float(rate[d]),
                "ci95_lo": float(lo_rate[d]),
                "ci95_hi": float(hi_rate[d]),
                "diff_min_usd": float(dd.min()),
                "diff_median_usd": float(np.median(dd)),
                "diff_max_usd": float(dd.max()),
                "diff_mean_usd": float(dd.mean()),
            }
        )

    monotone = bool(np.all(np.diff(rate) >= 0))
    ordered = bool(np.all(lo_rate <= hi_rate))
    point_inside = bool(np.all((lo_rate <= rate) & (rate <= hi_rate)))

    out = {
        "id": "S8",
        "title": "CT win rate by decile of team money difference (non-pistol rounds)",
        "definition": {
            "money_difference": "CT team money minus T team money, first snapshot of each round in players_1hz (freeze-end state)",
            "equipment_difference": "CT minus T sum of equipment_value_calc at the same snapshot",
            "pistol_rounds": "round in (1, 13)",
            "analysis_rows": "non-pistol rounds with 5 CT and 5 T players at the freeze-end snapshot",
            "deciles": "np.quantile deciles of money_difference on the analysis rows (edges fixed across bootstrap)",
            "bootstrap": "match-cluster; 1000 resamples of the 150 matches with replacement; numpy default_rng(0); percentile 95%",
            "auc": "P(CT win | higher score), ties count 0.5; score = money (or equipment) difference, positive means CT ahead",
        },
        "series": series,
        "edges_usd": [float(e) for e in edges],
        "key_numbers": {
            "rounds_in_sample": n_total,
            "pistol_rounds": n_pistol,
            "excluded_not_full_5v5": excluded,
            "analysis_rounds_n": n_analysis,
            "analysis_matches": n_matches,
            "overall_ct_win_rate": overall_rate,
            "overall_ct_win_rate_ci95": [float(lo_ov), float(hi_ov)],
            "auc_money_difference_alone": {
                "point": auc_m_point,
                "ci95_lo": float(lo_am),
                "ci95_hi": float(hi_am),
                "sklearn_check": sk_m,
            },
            "auc_equipment_difference_alone": {
                "point": auc_e_point,
                "ci95_lo": float(lo_ae),
                "ci95_hi": float(hi_ae),
                "sklearn_check": sk_e,
            },
            "decile_rate_monotone_increasing": monotone,
            "decile_intervals_ordered": ordered,
            "point_inside_own_interval_all_deciles": point_inside,
        },
        "checks": {
            "pistol_flag_mismatch_vs_buy_table": buy_pistol_mismatch,
            "rounds_missing_from_buy_table": buy_missing_rows,
            "max_abs_diff_ct_money_vs_buy_table": xchk_ct_money_max_abs,
            "max_abs_diff_t_money_vs_buy_table": xchk_t_money_max_abs,
            "max_abs_diff_ct_equipment_vs_buy_table": xchk_ct_equip_max_abs,
            "max_abs_diff_t_equipment_vs_buy_table": xchk_t_equip_max_abs,
            "all_series_counts_nonnegative": bool(all(s["n_rounds"] >= 0 for s in series)),
            "series_counts_sum_to_analysis_n": int(sum(s["n_rounds"] for s in series)) == n_analysis,
            "round_number_matches_round": int(df.filter(pl.col("round_number") != pl.col("round")).height),
        },
        "caveats": [
            "Sample is 150 matches (csds/2026/10, days 01-07); descriptive, not a causal estimate.",
            "Money and equipment are read from the 1 Hz freeze-end snapshot, not from the tomes.",
            "One round (round 16 of match csds/2026/10/07/22e9c3e7-...) has only 9 players at freeze end and is excluded from the analysis.",
            "Pistol rounds are round 1 and round 13 only; some matches have no round 1 or round 13 in the clean sample.",
            "Decile edges are fixed on the full analysis set; the bootstrap does not re-estimate them.",
            "Splits (train/test) are pooled; no model is fitted, so the split does not affect these numbers.",
            "Equal-count deciles can be unequal when money values tie; counts are reported per decile.",
        ],
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # SVG series uses the bootstrap intervals
    svg_series = [
        {**s} for s in series
    ]
    svg = build_svg(
        svg_series,
        n_analysis,
        n_matches,
        {"point": auc_m_point, "ci95_lo": float(lo_am), "ci95_hi": float(hi_am)},
        {"point": auc_e_point, "ci95_lo": float(lo_ae), "ci95_hi": float(hi_ae)},
    )
    OUT_SVG.write_text(svg, encoding="utf-8")

    # console summary for the run log
    print(f"rounds in sample: {n_total}; pistol: {n_pistol}; excluded (not 5v5): {len(excluded)}")
    print(f"analysis rounds: {n_analysis}; matches: {n_matches}")
    print(f"buy-table pistol mismatches: {buy_pistol_mismatch}; rounds missing from buy table: {buy_missing_rows}")
    print(f"cross-check max abs diff ct/t money: {xchk_ct_money_max_abs}/{xchk_t_money_max_abs}; "
          f"ct/t equip: {xchk_ct_equip_max_abs}/{xchk_t_equip_max_abs}")
    print(f"overall CT win rate: {overall_rate:.4f} [{lo_ov:.4f}, {hi_ov:.4f}]")
    print(f"AUC money: {auc_m_point:.4f} [{lo_am:.4f}, {hi_am:.4f}] sklearn={sk_m}")
    print(f"AUC equip: {auc_e_point:.4f} [{lo_ae:.4f}, {hi_ae:.4f}] sklearn={sk_e}")
    for s in series:
        print(
            f"D{s['decile']:>2} n={s['n_rounds']:>4} wins={s['ct_wins']:>4} "
            f"rate={s['ct_win_rate']:.3f} [{s['ci95_lo']:.3f}, {s['ci95_hi']:.3f}] "
            f"median_diff={s['diff_median_usd']:.0f} range=[{s['diff_min_usd']:.0f}, {s['diff_max_usd']:.0f}]"
        )
    print(f"monotone: {monotone}; intervals ordered: {ordered}; point inside own CI: {point_inside}")


if __name__ == "__main__":
    main()
