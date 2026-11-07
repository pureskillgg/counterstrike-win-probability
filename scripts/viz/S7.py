"""S7: CT round win rate by CT-minus-T score lead at round start.

Reads (read only):
  data/state_sample/rounds.parquet
  data/state_sample/round_buy_types.parquet   (cross-check of ct_win only)
Writes:
  data/viz/S7.json
  data/viz/S7.svg

Intervals: match-cluster bootstrap. The 150 matches are resampled with
replacement, 1000 resamples, seed 0; the rate is recomputed on all rounds of the
drawn matches. Percentile 95% interval with linear interpolation.
Pure Python for the bootstrap and the SVG (no plotting library).
"""

import json
import math
import random
from pathlib import Path

import polars as pl

ROOT = Path("G:/pureskill/csds-cs2")
RS = ROOT / "data" / "state_sample"
OUT_JSON = ROOT / "data" / "viz" / "S7.json"
OUT_SVG = ROOT / "data" / "viz" / "S7.svg"

MIN_N = 30
LEAD_MIN, LEAD_MAX = -8, 8
N_BOOT = 1000
SEED = 0

CT_COL = "#2563eb"
NEUTRAL = "#475569"
GRID = "#e2e8f0"

W, H = 800, 450
X0, X1 = 92, 772
Y0, Y1 = 30, 320


def group_of(lead):
    if lead >= 1:
        return "ahead"
    if lead == 0:
        return "level"
    return "behind"


def quantile(vals, q):
    """Linear interpolation between order statistics (numpy default convention)."""
    pos = (len(vals) - 1) * q
    lo = math.floor(pos)
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def lead_label(lead):
    if lead > 0:
        return f"+{lead}"
    if lead < 0:
        return f"\u2212{abs(lead)}"
    return "0"


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def xpx(lead):
    return X0 + (lead - LEAD_MIN + 0.5) / (LEAD_MAX - LEAD_MIN + 1) * (X1 - X0)


def ypx(p):
    return Y1 - p * (Y1 - Y0)


def text(x, y, s, size=12, anchor="start", fill=NEUTRAL, extra=""):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
        f'fill="{fill}"{extra}>{esc(s)}</text>'
    )


def build_svg(series, n_of, n_rounds, n_matches, overall):
    p = []
    p.append(
        f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">'
    )
    p.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')

    # horizontal grid and y ticks (0, 25, 50, 75, 100 %)
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = ypx(frac)
        if frac == 0.5:
            p.append(
                f'<line x1="{X0}" y1="{y:.1f}" x2="{X1}" y2="{y:.1f}" stroke="{NEUTRAL}" '
                f'stroke-width="1" stroke-dasharray="5 4"/>'
            )
        elif frac != 0.0:
            p.append(
                f'<line x1="{X0}" y1="{y:.1f}" x2="{X1}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>'
            )
        p.append(text(X0 - 10, y + 4, f"{int(round(frac * 100))}%", anchor="end"))

    # zero-lead reference
    x0line = xpx(0)
    p.append(
        f'<line x1="{x0line:.1f}" y1="{Y0}" x2="{x0line:.1f}" y2="{Y1}" stroke="{GRID}" '
        f'stroke-width="1" stroke-dasharray="3 3"/>'
    )

    # axes
    p.append(f'<line x1="{X0}" y1="{Y0}" x2="{X0}" y2="{Y1}" stroke="{NEUTRAL}" stroke-width="1"/>')
    p.append(f'<line x1="{X0}" y1="{Y1}" x2="{X1}" y2="{Y1}" stroke="{NEUTRAL}" stroke-width="1"/>')

    # x ticks, tick labels, and the n row
    for lead in range(LEAD_MIN, LEAD_MAX + 1):
        x = xpx(lead)
        p.append(f'<line x1="{x:.1f}" y1="{Y1}" x2="{x:.1f}" y2="{Y1 + 5}" stroke="{NEUTRAL}" stroke-width="1"/>')
        p.append(text(x, Y1 + 18, lead_label(lead), anchor="middle"))
        p.append(text(x, Y1 + 36, f"{n_of.get(lead, 0):,}", anchor="middle"))
    p.append(text(X0 - 10, Y1 + 36, "n", anchor="end"))

    # error bars and points for leads with n >= MIN_N
    for s in series:
        x = xpx(s["lead"])
        y_lo, y_hi, y_c = ypx(s["ci_lo"]), ypx(s["ci_hi"]), ypx(s["ct_win_rate"])
        p.append(
            f'<line x1="{x:.1f}" y1="{y_hi:.1f}" x2="{x:.1f}" y2="{y_lo:.1f}" stroke="{NEUTRAL}" stroke-width="1.5"/>'
        )
        p.append(f'<line x1="{x - 5:.1f}" y1="{y_hi:.1f}" x2="{x + 5:.1f}" y2="{y_hi:.1f}" stroke="{NEUTRAL}" stroke-width="1.5"/>')
        p.append(f'<line x1="{x - 5:.1f}" y1="{y_lo:.1f}" x2="{x + 5:.1f}" y2="{y_lo:.1f}" stroke="{NEUTRAL}" stroke-width="1.5"/>')
        tip = (
            f"Lead {lead_label(s['lead'])}: CT win {s['ct_win_rate'] * 100:.1f}% "
            f"(95% CI {s['ci_lo'] * 100:.1f} to {s['ci_hi'] * 100:.1f}), n = {s['n']:,}, {s['matches']} matches"
        )
        p.append(
            f'<circle cx="{x:.1f}" cy="{y_c:.1f}" r="4.5" fill="{CT_COL}" stroke="#ffffff" stroke-width="1.2">'
            f"<title>{esc(tip)}</title></circle>"
        )

    # axis titles
    p.append(
        text((X0 + X1) / 2, Y1 + 58, "CT score lead at round start (CT score minus T score)", size=13, anchor="middle")
    )
    p.append(
        f'<text transform="translate(22,{(Y0 + Y1) / 2:.1f}) rotate(-90)" font-size="13" '
        f'text-anchor="middle" fill="{NEUTRAL}">CT round win rate</text>'
    )

    # footer: sample size and interval definition
    p.append(
        text(
            X0,
            Y1 + 84,
            f"n = {n_rounds:,} rounds, {n_matches} matches. Overall CT win rate {overall['ct_win_rate'] * 100:.1f}%. "
            f"Points only where n >= {MIN_N}.",
        )
    )
    p.append(
        text(
            X0,
            Y1 + 102,
            "Bars: 95% match-cluster bootstrap intervals (1,000 resamples of matches, seed 0).",
        )
    )
    p.append("</svg>")
    return "\n".join(p) + "\n"


def main():
    rounds = pl.read_parquet(
        str(RS / "rounds.parquet"),
        columns=["match_key", "round", "ct_win", "round_number", "ct_score", "t_score"],
    )
    n_rounds = rounds.height
    match_keys = sorted(rounds["match_key"].unique().to_list())
    n_matches = len(match_keys)
    midx = {k: i for i, k in enumerate(match_keys)}

    checks = {
        "rounds_rows": n_rounds,
        "matches": n_matches,
        "ct_win_values": sorted(rounds["ct_win"].unique().to_list()),
        "duplicate_match_round_rows": n_rounds - rounds.unique(["match_key", "round"]).height,
        "score_sum_equals_round_number_minus_1_all_rows": rounds.filter(
            pl.col("ct_score") + pl.col("t_score") != pl.col("round_number") - 1
        ).height
        == 0,
    }

    bt = pl.read_parquet(str(RS / "round_buy_types.parquet"), columns=["match_key", "round", "ct_win"])
    joined = rounds.select(["match_key", "round", "ct_win"]).join(
        bt.select(["match_key", "round", pl.col("ct_win").alias("ct_win_bt")]),
        on=["match_key", "round"],
        how="inner",
    )
    checks["ct_win_mismatch_vs_round_buy_types"] = joined.filter(
        pl.col("ct_win").cast(pl.Int64) != pl.col("ct_win_bt").cast(pl.Int64)
    ).height
    checks["rounds_without_buy_type_row"] = n_rounds - joined.height

    # side-swap check: the stored ct_score should follow the team that is on CT now
    grid = {}
    for mk, rnd, cts, ts, cw in rounds.select(["match_key", "round", "ct_score", "t_score", "ct_win"]).rows():
        grid[(mk, rnd)] = (cts, ts, cw)
    swap_ok = swap_n = ns_ok = ns_n = 0
    for (mk, rnd), (cts, ts, cw) in grid.items():
        nxt = grid.get((mk, rnd + 1))
        if nxt is None:
            continue
        ncts, nts, _ = nxt
        if rnd == 12:
            swap_n += 1
            if ncts == ts + (1 - cw) and nts == cts + cw:
                swap_ok += 1
        elif rnd < 12:
            ns_n += 1
            if ncts == cts + cw and nts == ts + (1 - cw):
                ns_ok += 1
    checks["half_time_swap_12_to_13_matches_team_on_CT"] = f"{swap_ok} of {swap_n}"
    checks["no_swap_transitions_1_to_12_consistent"] = f"{ns_ok} of {ns_n}"

    # per match, per key: [rounds, ct wins]; keys are leads, groups and "all"
    recs = rounds.with_columns((pl.col("ct_score") - pl.col("t_score")).alias("lead")).select(
        ["match_key", "lead", "ct_win"]
    ).rows()
    contrib = [dict() for _ in range(n_matches)]
    for mk, lead, cw in recs:
        d = contrib[midx[mk]]
        for key in (lead, group_of(lead), "all"):
            cell = d.get(key)
            if cell is None:
                d[key] = [1, int(cw)]
            else:
                cell[0] += 1
                cell[1] += int(cw)

    totals = {}
    for d in contrib:
        for key, (n, w) in d.items():
            t = totals.setdefault(key, [0, 0, 0])
            t[0] += n
            t[1] += w
            t[2] += 1
    keys = list(totals.keys())

    # match-cluster bootstrap
    rng = random.Random(SEED)
    boot = {key: [] for key in keys}
    for _ in range(N_BOOT):
        acc = {}
        for _m in range(n_matches):
            d = contrib[rng.randrange(n_matches)]
            for key, (n, w) in d.items():
                cell = acc.get(key)
                if cell is None:
                    acc[key] = [n, w]
                else:
                    cell[0] += n
                    cell[1] += w
        for key, (n, w) in acc.items():
            if n > 0:
                boot[key].append(w / n)

    def summarize(key):
        n, w, m = totals[key]
        vals = sorted(boot[key])
        return {
            "n": n,
            "ct_wins": w,
            "ct_win_rate": round(w / n, 6),
            "matches": m,
            "ci_lo": round(quantile(vals, 0.025), 6),
            "ci_hi": round(quantile(vals, 0.975), 6),
            "valid_draws": len(vals),
        }

    series, not_drawn = [], []
    n_of = {}
    for lead in range(LEAD_MIN, LEAD_MAX + 1):
        if lead not in totals:
            n_of[lead] = 0
            not_drawn.append({"lead": lead, "n": 0, "reason": "no rounds"})
            continue
        s = summarize(lead)
        n_of[lead] = s["n"]
        if s["n"] >= MIN_N:
            series.append({"lead": lead, **s})
        else:
            not_drawn.append({"lead": lead, "n": s["n"], "reason": f"n < {MIN_N}"})

    outside = {
        lead: totals[lead][0]
        for lead in sorted(k for k in totals if isinstance(k, int) and not (LEAD_MIN <= k <= LEAD_MAX))
    }
    n_in_axis = sum(n_of.values())
    checks["n_sum_axis_plus_outside_equals_rounds"] = n_in_axis + sum(outside.values()) == n_rounds
    checks["all_counts_nonnegative"] = all(v[0] >= 0 and v[1] >= 0 for v in totals.values())
    checks["points_drawn"] = len(series)
    checks["min_n_drawn"] = min(s["n"] for s in series)
    checks["max_n_drawn"] = max(s["n"] for s in series)
    checks["intervals_ordered_and_contain_point"] = all(
        s["ci_lo"] <= s["ct_win_rate"] <= s["ci_hi"] for s in series
    )
    checks["min_valid_draws_drawn_leads"] = min(s["valid_draws"] for s in series)

    overall = summarize("all")
    groups = {g: summarize(g) for g in ("ahead", "level", "behind")}

    payload = {
        "id": "S7",
        "title": "CT win rate by CT-minus-T score lead at round start",
        "chart": {
            "file": "data/viz/S7.svg",
            "viewBox": f"0 0 {W} {H}",
            "x": "score lead at round start, CT score minus T score, drawn -8 to +8",
            "y": "CT round win rate, 0 to 100%",
            "error_bars": "95% match-cluster bootstrap percentile interval",
            "min_n_drawn": MIN_N,
        },
        "sample": {
            "source": "data/state_sample/rounds.parquet (1 Hz sample, 2026-10 days 01-07)",
            "rounds": n_rounds,
            "matches": n_matches,
        },
        "definition": {
            "lead": "ct_score minus t_score as stored in rounds.parquet (pre-round scores: round 1 is 0-0 and ct_score + t_score = round_number - 1 in every row)",
            "ct_win": "rounds.parquet ct_win (1 = CT won the round)",
        },
        "interval": {
            "method": "match-cluster bootstrap: resample the matches with replacement; rate recomputed on all rounds of drawn matches",
            "resamples": N_BOOT,
            "seed": SEED,
            "type": "2.5th and 97.5th percentile, linear interpolation",
        },
        "series": series,
        "not_drawn": not_drawn,
        "counts_outside_axis": outside,
        "overall": overall,
        "groups_by_sign_of_lead": groups,
        "checks": checks,
        "caveats": [
            "Matches that run past 24 rounds are selected on the future outcome; leads with n < 30 are not drawn.",
            "Rounds from the same match share a cluster; intervals are match-cluster bootstrap, not row-level.",
            "The lead is a state at round start and is correlated with team strength, economy and momentum; the figure is descriptive, not causal.",
            "Sample is 150 matches from days 01-07 of 2026-10 (1 Hz sample); not verified against the full 1,752-match tome set.",
            "Score convention verified on the sample (pre-round scores; stored ct_score follows the team on CT across the half-time swap).",
        ],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    svg = build_svg(series, n_of, n_rounds, n_matches, overall)
    OUT_SVG.write_text(svg, encoding="utf-8")

    print(json.dumps({"overall": overall, "groups": groups, "checks": checks}, indent=2, ensure_ascii=False))
    print("SERIES")
    for s in series:
        print(s)
    print("NOT_DRAWN", not_drawn)
    print("OUTSIDE", outside)


if __name__ == "__main__":
    main()
