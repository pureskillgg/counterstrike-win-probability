"""S3: freeze-end team equipment histogram in 1,000-USD bins.

Main series: non-pistol rounds (is_pistol == False), both sides pooled
(one observation per team per round). Second series: pistol rounds (1 and 13).
Vertical lines at 10,000 and 20,000 are UNVERIFIED cuts (drawn, not tested).

Intervals: match-cluster bootstrap percentile intervals. Resample the 150
matches with replacement, 1,000 resamples, seed 0. Rows are never resampled
individually.

Read-only inputs:
  data/state_sample/players_1hz.parquet      (projected columns only)
  data/state_sample/round_buy_types.parquet
  data/state_sample/rounds.parquet
Writes only:
  data/viz/S3.json, data/viz/S3.svg
"""

import json
import math
import random
from collections import Counter
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
P1HZ = ROOT / "data" / "state_sample" / "players_1hz.parquet"
RBT = ROOT / "data" / "state_sample" / "round_buy_types.parquet"
ROUNDS = ROOT / "data" / "state_sample" / "rounds.parquet"
OUT_JSON = ROOT / "data" / "viz" / "S3.json"
OUT_SVG = ROOT / "data" / "viz" / "S3.svg"

BIN_W = 1000
ECO_CUT = 10000
FULL_CUT = 20000
N_BOOT = 1000
SEED = 0
PISTOL_ROUND_NUMBERS = (1, 13)
CT_CODE, T_CODE = 3, 2
X_TICK_STEP = 5000

# palette from the brief
C_CT = "#2563eb"  # not used in this chart (sides pooled)
C_T = "#d97706"   # not used in this chart (sides pooled)
C_NEUTRAL = "#475569"
C_GRID = "#e2e8f0"
C_ACCENT = "#059669"
C_ACCENT_DARK = "#047857"
C_INK = "#1e293b"

W, H = 800, 450
PL, PR, PT, PB = 74, 20, 66, 58
PW = W - PL - PR
PH = H - PT - PB


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def median_sorted(v):
    n = len(v)
    if n == 0:
        return None
    if n % 2:
        return float(v[n // 2])
    return (v[n // 2 - 1] + v[n // 2]) / 2.0


def pct(sorted_vals, q):
    n = len(sorted_vals)
    pos = q * (n - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def r4(x):
    return None if x is None else round(float(x), 4)


def load_rounds():
    # 1) freeze-end snapshot per (match, round, team): the first tick of each round
    snap = (
        pl.scan_parquet(P1HZ)
        .select(
            [
                "match_key",
                "round",
                "tick",
                "seconds_since_live",
                "team_code",
                "player_id",
                "equipment_value_calc",
            ]
        )
        .with_columns(pl.col("tick").min().over(["match_key", "round"]).alias("t0"))
        .filter(pl.col("tick") == pl.col("t0"))
        .group_by(["match_key", "round", "team_code"])
        .agg(
            pl.len().alias("n_players"),
            pl.col("equipment_value_calc").sum().cast(pl.Int64).alias("equip_calc"),
            pl.col("seconds_since_live").max().alias("sl_max"),
            pl.col("t0").first().alias("freeze_tick"),
        )
        .with_columns(pl.col("round").cast(pl.Int64))
        .collect()
    )

    rbt = pl.read_parquet(RBT).with_columns(pl.col("round").cast(pl.Int64))
    rounds = pl.read_parquet(ROUNDS, columns=["match_key", "round", "alive_gap_snapshots"]).with_columns(
        pl.col("round").cast(pl.Int64)
    )

    def side(code):
        return snap.filter(pl.col("team_code") == code).select(
            "match_key",
            "round",
            pl.col("n_players").alias(f"n_{code}"),
            pl.col("equip_calc").alias(f"calc_{code}"),
            pl.col("sl_max").alias(f"slmax_{code}"),
        )

    rd = (
        rbt.join(side(CT_CODE), on=["match_key", "round"], how="left")
        .join(side(T_CODE), on=["match_key", "round"], how="left")
    )

    # excluded rounds: in rounds.parquet but not in round_buy_types
    missing = rounds.join(rbt.select(["match_key", "round"]), on=["match_key", "round"], how="anti")
    excluded = []
    for r in missing.iter_rows(named=True):
        s = snap.filter((pl.col("match_key") == r["match_key"]) & (pl.col("round") == r["round"]))
        per_team = {int(t): int(n) for t, n in zip(s["team_code"].to_list(), s["n_players"].to_list())}
        excluded.append(
            {
                "match_key": r["match_key"],
                "round": int(r["round"]),
                "reason": "absent from round_buy_types (not a clean 5v5 at freeze end)",
                "players_at_freeze_end_ct": per_team.get(CT_CODE, 0),
                "players_at_freeze_end_t": per_team.get(T_CODE, 0),
                "alive_gap_snapshots": int(r["alive_gap_snapshots"]),
            }
        )

    checks = {
        "rounds_in_round_buy_types": rd.height,
        "rounds_in_rounds_parquet": rounds.height,
        "ct_equipment_mismatch_vs_snapshot_sum": rd.filter(pl.col("ct_equipment") != pl.col("calc_3")).height,
        "t_equipment_mismatch_vs_snapshot_sum": rd.filter(pl.col("t_equipment") != pl.col("calc_2")).height,
        "rounds_without_5_ct_at_freeze_end": rd.filter(pl.col("n_3") != 5).height,
        "rounds_without_5_t_at_freeze_end": rd.filter(pl.col("n_2") != 5).height,
        "nonzero_seconds_since_live_at_first_snapshot": rd.filter(
            (pl.col("slmax_3") != 0) | (pl.col("slmax_2") != 0)
        ).height,
        "pistol_flag_disagrees_with_round_number": rd.filter(
            pl.col("is_pistol") != pl.col("round").is_in(list(PISTOL_ROUND_NUMBERS))
        ).height,
        "duplicate_match_round_keys": rd.height - rd.unique(["match_key", "round"]).height,
    }
    for k, v in checks.items():
        if k.endswith("mismatch_vs_snapshot_sum") or k.startswith("rounds_without") or k.startswith("nonzero") or k.startswith("pistol") or k.startswith("duplicate"):
            assert v == 0, f"check failed: {k}={v}"
    return rd, excluded, checks, rounds


def main():
    rd, excluded, checks, rounds = load_rounds()

    # one observation per team per round, both sides pooled
    long = pl.concat(
        [
            rd.select(
                "match_key",
                "round",
                "is_pistol",
                pl.lit("CT").alias("side"),
                pl.col("ct_equipment").alias("equip"),
            ),
            rd.select(
                "match_key",
                "round",
                "is_pistol",
                pl.lit("T").alias("side"),
                pl.col("t_equipment").alias("equip"),
            ),
        ]
    ).with_columns((pl.col("equip") // BIN_W).cast(pl.Int64).alias("bin"))

    matches = sorted(rd["match_key"].unique().to_list())
    n_matches = len(matches)
    midx = {m: i for i, m in enumerate(matches)}
    max_bin = int(long["bin"].max())
    NB = max_bin + 1
    XMAX = NB * BIN_W

    # per-match structures for the cluster bootstrap
    PM = [
        {
            "np_bins": [0] * NB,
            "p_bins": [0] * NB,
            "np_vals": [],
            "p_vals": [],
        }
        for _ in matches
    ]
    for r in long.iter_rows(named=True):
        pm = PM[midx[r["match_key"]]]
        if r["is_pistol"]:
            pm["p_vals"].append(int(r["equip"]))
            pm["p_bins"][int(r["bin"])] += 1
        else:
            pm["np_vals"].append(int(r["equip"]))
            pm["np_bins"][int(r["bin"])] += 1
    for pm in PM:
        pm["np_sorted"] = sorted(pm["np_vals"])
        pm["p_sorted"] = sorted(pm["p_vals"])

    def summarize(vals_np, bins_np, vals_p, bins_p):
        """Point summaries for a set of non-pistol and pistol team-round values."""
        n = len(vals_np)
        out = {
            "np_n": n,
            "np_below": sum(1 for v in vals_np if v < ECO_CUT),
            "np_mid": sum(1 for v in vals_np if ECO_CUT <= v < FULL_CUT),
            "np_full": sum(1 for v in vals_np if v >= FULL_CUT),
            "np_sum": sum(vals_np),
            "np_bins": bins_np,
            "p_n": len(vals_p),
            "p_below": sum(1 for v in vals_p if v < ECO_CUT),
            "p_sum": sum(vals_p),
            "p_bins": bins_p,
        }
        return out

    # point estimates from all matches
    all_np, all_p = [], []
    for pm in PM:
        all_np.extend(pm["np_vals"])
        all_p.extend(pm["p_vals"])
    np_bins = [sum(pm["np_bins"][b] for pm in PM) for b in range(NB)]
    p_bins = [sum(pm["p_bins"][b] for pm in PM) for b in range(NB)]
    point = summarize(all_np, np_bins, all_p, p_bins)
    assert sum(np_bins) == len(all_np) and sum(p_bins) == len(all_p)

    # match-cluster bootstrap: resample matches with replacement, seed 0
    rng = random.Random(SEED)
    b_np_bins = [[] for _ in range(NB)]
    b_p_bins = [[] for _ in range(NB)]
    b_share_below = []
    b_share_mid = []
    b_share_full = []
    b_np_mean = []
    b_np_median = []
    b_p_share_below = []
    b_p_mean = []
    for _ in range(N_BOOT):
        draws = [rng.randrange(n_matches) for _ in range(n_matches)]
        mult = Counter(draws)
        npb = [0] * NB
        pb = [0] * NB
        nn = nb = nm = nf = ns = 0
        pn = pbel = ps = 0
        vals = []
        for mi, k in mult.items():
            pm = PM[mi]
            for b in range(NB):
                npb[b] += k * pm["np_bins"][b]
                pb[b] += k * pm["p_bins"][b]
            for v in pm["np_vals"]:
                nn += k
                ns += k * v
                if v < ECO_CUT:
                    nb += k
                elif v < FULL_CUT:
                    nm += k
                else:
                    nf += k
            for v in pm["p_vals"]:
                pn += k
                ps += k * v
                if v < ECO_CUT:
                    pbel += k
            if k:
                vals.extend(pm["np_sorted"] * k)
        vals.sort()
        for b in range(NB):
            b_np_bins[b].append(npb[b])
            b_p_bins[b].append(pb[b])
        b_share_below.append(nb / nn)
        b_share_mid.append(nm / nn)
        b_share_full.append(nf / nn)
        b_np_mean.append(ns / nn)
        b_np_median.append(median_sorted(vals))
        b_p_share_below.append(pbel / pn if pn else None)
        b_p_mean.append(ps / pn if pn else None)

    def ci(vals):
        s = sorted(v for v in vals if v is not None)
        return pct(s, 0.025), pct(s, 0.975)

    # bins in the output: 0..NB-1, each [lo, lo+1000)
    bins_out = []
    peak_np = 0
    ymax_raw = 0.0
    for b in range(NB):
        lo_ci_np, hi_ci_np = ci(b_np_bins[b])
        lo_ci_p, hi_ci_p = ci(b_p_bins[b])
        est_np = np_bins[b]
        est_p = p_bins[b]
        if not (lo_ci_np - 1e-9 <= est_np <= hi_ci_np + 1e-9):
            print(f"note: non-pistol estimate outside bootstrap CI in bin {b}: {est_np} vs [{lo_ci_np}, {hi_ci_np}]")
        if not (lo_ci_p - 1e-9 <= est_p <= hi_ci_p + 1e-9):
            print(f"note: pistol estimate outside bootstrap CI in bin {b}: {est_p} vs [{lo_ci_p}, {hi_ci_p}]")
        assert lo_ci_np <= hi_ci_np and lo_ci_p <= hi_ci_p
        peak_np = max(peak_np, est_np)
        ymax_raw = max(ymax_raw, hi_ci_np, hi_ci_p)
        bins_out.append(
            {
                "bin_lo_usd": b * BIN_W,
                "bin_hi_usd": (b + 1) * BIN_W,
                "nonpistol_count": est_np,
                "nonpistol_ci95_lo": r4(lo_ci_np),
                "nonpistol_ci95_hi": r4(hi_ci_np),
                "pistol_count": est_p,
                "pistol_ci95_lo": r4(lo_ci_p),
                "pistol_ci95_hi": r4(hi_ci_p),
            }
        )

    n_np = point["np_n"]
    n_p = point["p_n"]
    n_rounds_np = int(rd.filter(~pl.col("is_pistol")).height)
    n_rounds_p = int(rd.filter(pl.col("is_pistol")).height)
    med_np_point = median_sorted(sorted(all_np))
    med_np_ci = ci(b_np_median)
    mean_np_point = point["np_sum"] / n_np
    mean_np_ci = ci(b_np_mean)
    sh_below_pt = point["np_below"] / n_np
    sh_mid_pt = point["np_mid"] / n_np
    sh_full_pt = point["np_full"] / n_np
    p_sh_below_pt = point["p_below"] / n_p
    p_mean_pt = point["p_sum"] / n_p
    summary = {
        "non_pistol_team_rounds": {
            "n": n_np,
            "share_below_10k": {"est": r4(sh_below_pt), "ci95": [r4(x) for x in ci(b_share_below)]},
            "share_10k_to_under_20k": {"est": r4(sh_mid_pt), "ci95": [r4(x) for x in ci(b_share_mid)]},
            "share_20k_plus": {"est": r4(sh_full_pt), "ci95": [r4(x) for x in ci(b_share_full)]},
            "mean_usd": {"est": r4(mean_np_point), "ci95": [r4(x) for x in mean_np_ci]},
            "median_usd": {"est": r4(med_np_point), "ci95": [r4(x) for x in med_np_ci]},
            "min_usd": min(all_np),
            "max_usd": max(all_np),
            "modal_bin_lo_usd": int(max(range(NB), key=lambda b: np_bins[b]) * BIN_W),
            "modal_bin_count": peak_np,
        },
        "pistol_team_rounds": {
            "n": n_p,
            "share_below_10k": {"est": r4(p_sh_below_pt), "ci95": [r4(x) for x in ci(b_p_share_below)]},
            "mean_usd": {"est": r4(p_mean_pt), "ci95": [r4(x) for x in ci(b_p_mean)]},
            "min_usd": min(all_p),
            "max_usd": max(all_p),
        },
    }

    out = {
        "id": "S3",
        "title": "Team equipment value at freeze end, 1,000 USD bins",
        "status": "UNVERIFIED cut lines at 10,000 and 20,000",
        "source": {
            "sample": "1 Hz sample, 150 matches, 2026-10 days 01-07",
            "tables": [
                "data/state_sample/players_1hz.parquet (first snapshot per match, round)",
                "data/state_sample/round_buy_types.parquet",
                "data/state_sample/rounds.parquet",
            ],
            "tomes_used": False,
        },
        "definition": {
            "value": "sum of equipment_value_calc over a team's players at the first snapshot of each round (freeze end)",
            "unit": "team-round (each team in each round is one observation; both sides pooled)",
            "bin": "[k*1000, (k+1)*1000) USD; value exactly 10,000 falls in the 10k-to-under-20k group",
            "non_pistol": "is_pistol == False",
            "pistol": "is_pistol == True (rounds 1 and 13)",
        },
        "bootstrap": {
            "method": "match-cluster percentile bootstrap",
            "resample_unit": "match_key (150 matches, with replacement)",
            "n_resamples": N_BOOT,
            "seed": SEED,
            "interval": "95% (2.5th to 97.5th percentile)",
        },
        "sample": {
            "n_matches": n_matches,
            "n_rounds_used": rd.height,
            "n_rounds_nonpistol": n_rounds_np,
            "n_rounds_pistol": n_rounds_p,
            "n_team_rounds_nonpistol": n_np,
            "n_team_rounds_pistol": n_p,
            "n_team_rounds_total": n_np + n_p,
            "rounds_in_rounds_parquet": rounds.height,
            "excluded_rounds": excluded,
        },
        "checks": checks,
        "x_axis": {"bin_width_usd": BIN_W, "range_usd": [0, XMAX], "max_bin_lo_usd": (NB - 1) * BIN_W},
        "cuts_unverified_usd": [ECO_CUT, FULL_CUT],
        "bins": bins_out,
        "key_numbers": summary,
        "assumptions": [
            "equipment_value_calc (team sum at freeze end) is taken as the team equipment value; its definition was not checked against game rules, only that it reproduces ct_equipment and t_equipment in round_buy_types exactly.",
            "The first snapshot per (match_key, round) has seconds_since_live == 0 and is the freeze-end state (checked).",
            "Cut lines at 10,000 and 20,000 are UNVERIFIED thresholds drawn for reference; no outcome is tested against them.",
            "Team-rounds are pooled across sides and treated as draws with match-level clustering only; no adjustment for map, side, or score state.",
            "The 1 Hz sample is a subsample of 150 matches, not the full 1,752-match tome.",
        ],
    }

    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ SVG
    XMAX_f = float(XMAX)
    ymax_candidates = [10, 20, 25, 50, 100]
    ystep = next(s for s in ymax_candidates if math.ceil(ymax_raw * 1.06 / s) <= 8)
    YMAX = math.ceil(ymax_raw * 1.06 / ystep) * ystep

    def X(v):
        return PL + v / XMAX_f * PW

    def Y(c):
        return PT + PH - c / YMAX * PH

    bw_px = PW / NB
    parts = []
    add = parts.append
    add('<svg viewBox="0 0 800 450" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">')
    add('<rect x="0" y="0" width="800" height="450" fill="#ffffff"/>')

    # horizontal grid and y tick labels
    for v in range(0, int(YMAX) + 1, ystep):
        y = Y(v)
        add(f'<line x1="{PL}" y1="{y:.1f}" x2="{PL + PW}" y2="{y:.1f}" stroke="{C_GRID}" stroke-width="1"/>')
        add(f'<text x="{PL - 8}" y="{y + 4:.1f}" font-size="12" text-anchor="end" fill="{C_NEUTRAL}">{v}</text>')

    # bars: non-pistol (full bin width, neutral)
    for b in range(NB):
        x0 = X(b * BIN_W)
        x1 = X((b + 1) * BIN_W)
        c = np_bins[b]
        if c > 0:
            add(
                f'<rect x="{x0 + 0.8:.2f}" y="{Y(c):.2f}" width="{(x1 - x0) - 1.6:.2f}" '
                f'height="{Y(0) - Y(c):.2f}" fill="{C_NEUTRAL}" fill-opacity="0.45" stroke="{C_NEUTRAL}" stroke-width="0.6"/>'
            )

    # pistol bars: narrow, centred in the bin, accent
    for b in range(NB):
        x0 = X(b * BIN_W)
        x1 = X((b + 1) * BIN_W)
        c = p_bins[b]
        cx = (x0 + x1) / 2
        w = (x1 - x0) * 0.36
        if c > 0:
            add(
                f'<rect x="{cx - w / 2:.2f}" y="{Y(c):.2f}" width="{w:.2f}" '
                f'height="{Y(0) - Y(c):.2f}" fill="{C_ACCENT}" fill-opacity="0.9"/>'
            )

    # interval bars at bin centres
    for b in range(NB):
        x0 = X(b * BIN_W)
        x1 = X((b + 1) * BIN_W)
        cx = (x0 + x1) / 2
        for est_c, lo, hi, col, sw in [
            (np_bins[b], bins_out[b]["nonpistol_ci95_lo"], bins_out[b]["nonpistol_ci95_hi"], C_INK, 1.0),
            (p_bins[b], bins_out[b]["pistol_ci95_lo"], bins_out[b]["pistol_ci95_hi"], C_ACCENT_DARK, 1.2),
        ]:
            if hi is None or hi <= 0 or hi - lo <= 0:
                continue
            add(f'<line x1="{cx:.2f}" y1="{Y(lo):.2f}" x2="{cx:.2f}" y2="{Y(hi):.2f}" stroke="{col}" stroke-width="{sw}"/>')
            for yy in (lo, hi):
                add(f'<line x1="{cx - 3:.2f}" y1="{Y(yy):.2f}" x2="{cx + 3:.2f}" y2="{Y(yy):.2f}" stroke="{col}" stroke-width="{sw}"/>')

    # axes
    add(f'<line x1="{PL}" y1="{PT + PH}" x2="{PL + PW}" y2="{PT + PH}" stroke="{C_NEUTRAL}" stroke-width="1"/>')
    add(f'<line x1="{PL}" y1="{PT}" x2="{PL}" y2="{PT + PH}" stroke="{C_NEUTRAL}" stroke-width="1"/>')

    # x ticks
    for v in range(0, XMAX + 1, X_TICK_STEP):
        x = X(v)
        add(f'<line x1="{x:.1f}" y1="{PT + PH}" x2="{x:.1f}" y2="{PT + PH + 4}" stroke="{C_NEUTRAL}" stroke-width="1"/>')
        lab = "$0" if v == 0 else f"${v // 1000}k"
        add(f'<text x="{x:.1f}" y="{PT + PH + 18}" font-size="12" text-anchor="middle" fill="{C_NEUTRAL}">{lab}</text>')

    # axis titles
    add(
        f'<text x="{PL + PW / 2:.1f}" y="{H - 12}" font-size="13" text-anchor="middle" fill="{C_INK}">'
        f"Team equipment value at freeze end (USD, sum over the team)</text>"
    )
    yc = PT + PH / 2
    add(
        f'<text x="18" y="{yc:.1f}" font-size="13" text-anchor="middle" fill="{C_INK}" '
        f'transform="rotate(-90 18 {yc:.1f})">Team-rounds per $1,000 bin</text>'
    )

    # unverified cut lines
    for cut, lab in [(ECO_CUT, "eco cut $10k"), (FULL_CUT, "full-buy cut $20k")]:
        x = X(cut)
        add(
            f'<line x1="{x:.1f}" y1="{PT}" x2="{x:.1f}" y2="{PT + PH}" stroke="{C_INK}" '
            f'stroke-width="1.5" stroke-dasharray="6 4"/>'
        )
        add(f'<text x="{x + 5:.1f}" y="{PT + 15}" font-size="12" text-anchor="start" fill="{C_INK}">{lab}</text>')

    # sample-size annotation (upper right, inside plot)
    add(
        f'<text x="{PL + PW - 6}" y="{PT + 44}" font-size="12" text-anchor="end" fill="{C_INK}">'
        f"n = {rd.height:,} rounds from {n_matches} matches</text>"
    )
    add(
        f'<text x="{PL + PW - 6}" y="{PT + 60}" font-size="12" text-anchor="end" fill="{C_INK}">'
        f"{n_np + n_p:,} team-rounds, both sides pooled</text>"
    )

    # legend (two rows above the plot)
    ly1, ly2 = 20, 42
    add(f'<rect x="{PL}" y="{ly1 - 9}" width="14" height="10" fill="{C_NEUTRAL}" fill-opacity="0.45" stroke="{C_NEUTRAL}" stroke-width="0.6"/>')
    add(f'<text x="{PL + 20}" y="{ly1}" font-size="12" fill="{C_INK}">Non-pistol rounds ({n_np:,} team-rounds)</text>')
    x_p = PL + 300
    add(f'<rect x="{x_p}" y="{ly1 - 9}" width="14" height="10" fill="{C_ACCENT}"/>')
    add(f'<text x="{x_p + 20}" y="{ly1}" font-size="12" fill="{C_INK}">Pistol rounds 1 and 13 ({n_p:,} team-rounds)</text>')

    add(f'<line x1="{PL}" y1="{ly2 - 4}" x2="{PL + 14}" y2="{ly2 - 4}" stroke="{C_INK}" stroke-width="1.5" stroke-dasharray="6 4"/>')
    add(f'<text x="{PL + 20}" y="{ly2}" font-size="12" fill="{C_INK}">Eco cut $10k (UNVERIFIED)</text>')
    x2 = PL + 200
    add(f'<line x1="{x2}" y1="{ly2 - 4}" x2="{x2 + 14}" y2="{ly2 - 4}" stroke="{C_INK}" stroke-width="1.5" stroke-dasharray="6 4"/>')
    add(f'<text x="{x2 + 20}" y="{ly2}" font-size="12" fill="{C_INK}">Full-buy cut $20k (UNVERIFIED)</text>')
    x3 = PL + 440
    add(f'<line x1="{x3 + 7}" y1="{ly2 - 12}" x2="{x3 + 7}" y2="{ly2 + 2}" stroke="{C_INK}" stroke-width="1"/>')
    add(f'<line x1="{x3 + 4}" y1="{ly2 - 12}" x2="{x3 + 10}" y2="{ly2 - 12}" stroke="{C_INK}" stroke-width="1"/>')
    add(f'<line x1="{x3 + 4}" y1="{ly2 + 2}" x2="{x3 + 10}" y2="{ly2 + 2}" stroke="{C_INK}" stroke-width="1"/>')
    add(f'<text x="{x3 + 20}" y="{ly2}" font-size="12" fill="{C_INK}">95% match-cluster bootstrap CI</text>')

    add("</svg>")
    OUT_SVG.write_text("\n".join(parts) + "\n", encoding="utf-8")

    # console summary for verification
    print(f"matches={n_matches} rounds={rd.height} nonpistol_rounds={n_rounds_np} pistol_rounds={n_rounds_p}")
    print(f"team-rounds nonpistol={n_np} pistol={n_p}; NB={NB} XMAX={XMAX} YMAX={YMAX} ystep={ystep}")
    print("non-pistol share <10k: %.4f CI %s" % (sh_below_pt, summary["non_pistol_team_rounds"]["share_below_10k"]["ci95"]))
    print("non-pistol share 10k-<20k: %.4f CI %s" % (sh_mid_pt, summary["non_pistol_team_rounds"]["share_10k_to_under_20k"]["ci95"]))
    print("non-pistol share >=20k: %.4f CI %s" % (sh_full_pt, summary["non_pistol_team_rounds"]["share_20k_plus"]["ci95"]))
    print("non-pistol mean %.1f CI %s; median %s CI %s" % (mean_np_point, summary["non_pistol_team_rounds"]["mean_usd"]["ci95"], med_np_point, summary["non_pistol_team_rounds"]["median_usd"]["ci95"]))
    print("pistol share <10k: %.4f; mean %.1f" % (p_sh_below_pt, p_mean_pt))
    print("bins (lo, np, np_ci, p, p_ci):")
    for bo in bins_out:
        if bo["nonpistol_count"] or bo["pistol_count"]:
            print(
                bo["bin_lo_usd"],
                bo["nonpistol_count"], (bo["nonpistol_ci95_lo"], bo["nonpistol_ci95_hi"]),
                bo["pistol_count"], (bo["pistol_ci95_lo"], bo["pistol_ci95_hi"]),
            )
    print("checks:", checks)
    print("excluded:", excluded)


if __name__ == "__main__":
    main()
