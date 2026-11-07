"""S5: team spread over the round, from the 2026-10 1 Hz state sample.

For each round and side, the spread is the mean pairwise 2D distance (x_pos, y_pos)
between that side's living players at each second since live (0..90). Round values are
averaged by side and outcome (won / lost) per second. Intervals are a match-cluster
percentile bootstrap (150 matches resampled with replacement, 1000 resamples, seed 0).

Writes:
  data/viz/S5.json
  data/viz/S5.svg
Run with:
  G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\viz\\S5.py
"""

import json
import math
import random
import statistics
from collections import Counter

import polars as pl

ROOT = "G:/pureskill/csds-cs2"
SAMPLE = ROOT + "/data/state_sample"
OUT_JSON = ROOT + "/data/viz/S5.json"
OUT_SVG = ROOT + "/data/viz/S5.svg"

MAX_SEC = 90
NSEC = MAX_SEC + 1
MIN_ROUNDS = 100
N_BOOT = 1000
SEED = 0
KEY_SECS = [0, 15, 30, 45, 60, 75, 90]

W, H = 800, 450
PL, PR, PT, PB = 74, 780, 46, 352

# series index si: 0 CT won, 1 CT lost, 2 T won, 3 T lost
SERIES = [
    {"key": "CT_won", "side": "CT", "outcome": "won", "label": "CT won", "color": "#2563eb", "dashed": False},
    {"key": "CT_lost", "side": "CT", "outcome": "lost", "label": "CT lost", "color": "#2563eb", "dashed": True},
    {"key": "T_won", "side": "T", "outcome": "won", "label": "T won", "color": "#d97706", "dashed": False},
    {"key": "T_lost", "side": "T", "outcome": "lost", "label": "T lost", "color": "#d97706", "dashed": True},
]
NS = len(SERIES)
NFLAT = NS * NSEC


def flat(si, sec):
    return si * NSEC + sec


def rnd(v, nd=4):
    return None if v is None else round(v, nd)


# ---------------------------------------------------------------- 1. load
players = (
    pl.scan_parquet(SAMPLE + "/players_1hz.parquet")
    .select(["match_key", "round", "seconds_since_live", "player_id", "team_code", "x_pos", "y_pos"])
    .filter(pl.col("seconds_since_live").is_between(0, MAX_SEC))
    .filter(pl.col("x_pos").is_finite() & pl.col("y_pos").is_finite())
    .with_columns(pl.col("x_pos").cast(pl.Float64), pl.col("y_pos").cast(pl.Float64))
    .collect()
)
rounds = (
    pl.read_parquet(SAMPLE + "/rounds.parquet", columns=["match_key", "round", "ct_win"])
    .with_columns(pl.col("round").cast(pl.Int32))
)

team_codes = sorted(players["team_code"].unique().to_list())
assert team_codes == [2, 3], team_codes  # 2 = T, 3 = CT

match_keys = sorted(players["match_key"].unique().to_list())
M = len(match_keys)
mk_map = pl.DataFrame({"match_key": match_keys, "mi": list(range(M))}).with_columns(pl.col("mi").cast(pl.Int32))
players = players.join(mk_map, on="match_key", how="left").drop("match_key")
rounds = rounds.join(mk_map, on="match_key", how="left")
assert players["mi"].null_count() == 0
assert rounds["mi"].null_count() == 0

n_rounds_sample = players.select(pl.struct(["mi", "round"]).n_unique()).item()
print("matches:", M, "rounds in 1 Hz sample:", n_rounds_sample)

# ---------------------------------------------------------------- 2. spreads
KEYS = ["mi", "round", "seconds_since_live", "team_code"]

alive = players.group_by(KEYS).agg(pl.len().alias("k"))

a = players.select(KEYS + [pl.col("player_id").alias("pid_a"), pl.col("x_pos").alias("xa"), pl.col("y_pos").alias("ya")])
b = players.select(KEYS + [pl.col("player_id").alias("pid_b"), pl.col("x_pos").alias("xb"), pl.col("y_pos").alias("yb")])
pairs = (
    a.join(b, on=KEYS, how="inner")
    .filter(pl.col("pid_a") < pl.col("pid_b"))
    .with_columns(((pl.col("xa") - pl.col("xb")) ** 2 + (pl.col("ya") - pl.col("yb")) ** 2).sqrt().alias("d"))
)

spread = (
    pairs.group_by(KEYS)
    .agg(pl.col("d").mean().alias("spread"), pl.len().alias("npairs"))
    .join(alive, on=KEYS, how="left")
    .join(rounds.select(["mi", "round", "ct_win"]), on=["mi", "round"], how="left")
)
assert spread["ct_win"].null_count() == 0
assert spread.select((pl.col("npairs") == pl.col("k") * (pl.col("k") - 1) // 2).all()).item(), "pair count mismatch"

spread = spread.with_columns(
    pl.when(pl.col("team_code") == 3).then(pl.lit(0, dtype=pl.Int32)).otherwise(pl.lit(2, dtype=pl.Int32)).alias("side_i"),
    pl.when(pl.col("team_code") == 3).then(pl.col("ct_win") == 1).otherwise(pl.col("ct_win") == 0).alias("won"),
)
spread = spread.with_columns(
    (pl.col("side_i") + pl.when(pl.col("won")).then(0).otherwise(1)).cast(pl.Int32).alias("si")
)

agg = spread.group_by(["mi", "si", "seconds_since_live"]).agg(
    pl.col("spread").sum().alias("S"),
    pl.len().alias("N"),
    pl.col("k").sum().alias("K"),
)

S_m = [[0.0] * NFLAT for _ in range(M)]
N_m = [[0] * NFLAT for _ in range(M)]
K_m = [[0] * NFLAT for _ in range(M)]
for row in agg.iter_rows(named=True):
    f = flat(int(row["si"]), int(row["seconds_since_live"]))
    m = int(row["mi"])
    S_m[m][f] = float(row["S"])
    N_m[m][f] = int(row["N"])
    K_m[m][f] = int(row["K"])

N_tot = [sum(N_m[m][f] for m in range(M)) for f in range(NFLAT)]
S_tot = [sum(S_m[m][f] for m in range(M)) for f in range(NFLAT)]
K_tot = [sum(K_m[m][f] for m in range(M)) for f in range(NFLAT)]
mean_pt = [S_tot[f] / N_tot[f] if N_tot[f] > 0 else None for f in range(NFLAT)]
mean_alive = [K_tot[f] / N_tot[f] if N_tot[f] > 0 else None for f in range(NFLAT)]
plotted = [N_tot[f] >= MIN_ROUNDS for f in range(NFLAT)]

assert all(n >= 0 for n in N_tot)

# ---------------------------------------------------------------- 3. match-cluster bootstrap
rng = random.Random(SEED)
boot = [[] for _ in range(NFLAT)]
boot_diff = {"CT": [[] for _ in range(NSEC)], "T": [[] for _ in range(NSEC)]}
for _ in range(N_BOOT):
    draws = Counter(rng.randrange(M) for _ in range(M))
    num = [0.0] * NFLAT
    den = [0] * NFLAT
    for m, c in draws.items():
        num = [x + c * y for x, y in zip(num, S_m[m])]
        den = [x + c * y for x, y in zip(den, N_m[m])]
    means = [(num[f] / den[f]) if den[f] > 0 else None for f in range(NFLAT)]
    for f in range(NFLAT):
        if means[f] is not None:
            boot[f].append(means[f])
    for side, wi, li in (("CT", 0, 1), ("T", 2, 3)):
        for sec in range(NSEC):
            fw, fl = flat(wi, sec), flat(li, sec)
            if means[fw] is not None and means[fl] is not None:
                boot_diff[side][sec].append(means[fw] - means[fl])


def pct(sorted_vals, q):
    n = len(sorted_vals)
    pos = q * (n - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, n - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


ci_lo = [None] * NFLAT
ci_hi = [None] * NFLAT
se = [None] * NFLAT
for f in range(NFLAT):
    vals = boot[f]
    if len(vals) >= 2:
        sv = sorted(vals)
        ci_lo[f] = pct(sv, 0.025)
        ci_hi[f] = pct(sv, 0.975)
        se[f] = statistics.stdev(vals)

diff = {}
for side, wi, li in (("CT", 0, 1), ("T", 2, 3)):
    diff[side] = []
    for sec in range(NSEC):
        fw, fl = flat(wi, sec), flat(li, sec)
        pt = (mean_pt[fw] - mean_pt[fl]) if (mean_pt[fw] is not None and mean_pt[fl] is not None) else None
        vals = boot_diff[side][sec]
        lo = hi = None
        if pt is not None and len(vals) >= 2:
            sv = sorted(vals)
            lo, hi = pct(sv, 0.025), pct(sv, 0.975)
        diff[side].append({
            "sec": sec,
            "diff": rnd(pt, 2),
            "ci_low": rnd(lo, 2),
            "ci_high": rnd(hi, 2),
            "plotted": bool(plotted[fw] and plotted[fl]),
        })

# sanity checks
viol = [f for f in range(NFLAT) if plotted[f] and not (ci_lo[f] <= mean_pt[f] <= ci_hi[f])]
print("interval ordering violations among plotted points:", len(viol))
assert not viol

# ---------------------------------------------------------------- 4. JSON
series_out = []
for si, s in enumerate(SERIES):
    series_out.append({
        "key": s["key"],
        "side": s["side"],
        "outcome": s["outcome"],
        "label": s["label"],
        "dashed": s["dashed"],
        "color": s["color"],
        "seconds": list(range(NSEC)),
        "n_rounds": [N_tot[flat(si, sec)] for sec in range(NSEC)],
        "mean_alive_teammates": [rnd(mean_alive[flat(si, sec)], 3) for sec in range(NSEC)],
        "mean": [rnd(mean_pt[flat(si, sec)], 2) for sec in range(NSEC)],
        "ci_low": [rnd(ci_lo[flat(si, sec)], 2) for sec in range(NSEC)],
        "ci_high": [rnd(ci_hi[flat(si, sec)], 2) for sec in range(NSEC)],
        "se": [rnd(se[flat(si, sec)], 2) for sec in range(NSEC)],
        "plotted": [plotted[flat(si, sec)] for sec in range(NSEC)],
    })


def point(si, sec):
    f = flat(si, sec)
    return {
        "n_rounds": N_tot[f],
        "mean": rnd(mean_pt[f], 2),
        "ci_low": rnd(ci_lo[f], 2),
        "ci_high": rnd(ci_hi[f], 2),
        "mean_alive_teammates": rnd(mean_alive[f], 2),
        "plotted": plotted[f],
    }


last_plotted = {}
for si, s in enumerate(SERIES):
    secs = [sec for sec in range(NSEC) if plotted[flat(si, sec)]]
    last_plotted[s["key"]] = max(secs) if secs else None

key_numbers = {
    "n_rounds_sample": n_rounds_sample,
    "n_matches": M,
    "min_rounds_per_plotted_point": MIN_ROUNDS,
    "last_plotted_second": last_plotted,
    "points": {s["key"]: {str(sec): point(si, sec) for sec in KEY_SECS} for si, s in enumerate(SERIES)},
    "won_minus_lost": {side: {str(sec): diff[side][sec] for sec in KEY_SECS} for side in ("CT", "T")},
}

out = {
    "id": "S5",
    "title": "Team spread over the round",
    "metric": "mean pairwise 2D distance between a side's living players, per round, averaged over rounds",
    "units": "game units (not converted)",
    "x": "seconds_since_live = (tick - live_start_tick) / 64; verified exact on the sample",
    "sample": {
        "file": "data/state_sample/players_1hz.parquet + rounds.parquet",
        "n_rounds": n_rounds_sample,
        "n_matches": M,
        "split_note": "train and test rows both included; no model fitted",
    },
    "rules": {
        "max_second": MAX_SEC,
        "pairwise_needs_alive_at_least": 2,
        "plot_rule": "a series is drawn at a second only where it has at least 100 rounds with 2+ living players on that side",
        "round_weighting": "equal weight per round within a series and second",
        "bootstrap": {"method": "match-cluster percentile", "resamples": N_BOOT, "seed": SEED,
                      "ci": 0.95, "unit": "match (150 resampled with replacement)"},
    },
    "series": series_out,
    "won_minus_lost": diff,
    "key_numbers": key_numbers,
}

# ---------------------------------------------------------------- 5. SVG
plotted_f = [f for f in range(NFLAT) if plotted[f]]
lo_all = min(ci_lo[f] for f in plotted_f)
hi_all = max(ci_hi[f] for f in plotted_f)


def nice_step(span, target=5):
    raw = span / target
    mag = 10 ** math.floor(math.log10(raw))
    for mult in (1, 2, 2.5, 5, 10):
        if mult * mag >= raw:
            return mult * mag


step = nice_step(hi_all - lo_all)
ymin_t = math.floor(lo_all / step) * step
ymax_t = math.ceil(hi_all / step) * step


def xp(sec):
    return PL + sec * (PR - PL) / MAX_SEC


def yp(v):
    return PB - (v - ymin_t) * (PB - PT) / (ymax_t - ymin_t)


def runs(si):
    out_runs, cur = [], []
    for sec in range(NSEC):
        if plotted[flat(si, sec)]:
            cur.append(sec)
        elif cur:
            out_runs.append(cur)
            cur = []
    if cur:
        out_runs.append(cur)
    return out_runs


def band_path(si, secs):
    top = [f"{xp(s):.1f},{yp(ci_hi[flat(si, s)]):.1f}" for s in secs]
    bot = [f"{xp(s):.1f},{yp(ci_lo[flat(si, s)]):.1f}" for s in reversed(secs)]
    return "M " + " L ".join(top + bot) + " Z"


def build_svg():
    parts = [
        f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">',
        f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>',
    ]
    v = ymin_t
    ticks = []
    while v <= ymax_t + 1e-9:
        ticks.append(v)
        v += step
    for v in ticks:
        y = yp(v)
        parts.append(f'<line x1="{PL}" y1="{y:.1f}" x2="{PR}" y2="{y:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        parts.append(f'<text x="{PL - 8}" y="{y + 4:.1f}" font-size="12" text-anchor="end" fill="#475569">{v:,.0f}</text>')
    for s in range(0, MAX_SEC + 1, 10):
        x = xp(s)
        parts.append(f'<line x1="{x:.1f}" y1="{PB}" x2="{x:.1f}" y2="{PB + 4}" stroke="#475569" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{PB + 18}" font-size="12" text-anchor="middle" fill="#475569">{s}</text>')
    for si, s in enumerate(SERIES):
        for secs in runs(si):
            parts.append(f'<path d="{band_path(si, secs)}" fill="{s["color"]}" fill-opacity="0.15" stroke="none"/>')
    for si, s in enumerate(SERIES):
        dash = ' stroke-dasharray="6 4"' if s["dashed"] else ""
        for secs in runs(si):
            pts = " ".join(f"{xp(sec):.1f},{yp(mean_pt[flat(si, sec)]):.1f}" for sec in secs)
            parts.append(
                f'<polyline points="{pts}" fill="none" stroke="{s["color"]}" stroke-width="2.2" '
                f'stroke-linejoin="round" stroke-linecap="round"{dash}/>'
            )
    parts.append(f'<line x1="{PL}" y1="{PT}" x2="{PL}" y2="{PB}" stroke="#475569" stroke-width="1"/>')
    parts.append(f'<line x1="{PL}" y1="{PB}" x2="{PR}" y2="{PB}" stroke="#475569" stroke-width="1"/>')
    parts.append(
        f'<text x="{(PL + PR) / 2:.1f}" y="{PB + 40}" font-size="13" text-anchor="middle" fill="#475569">'
        f'Seconds since round went live</text>'
    )
    parts.append(
        f'<text transform="translate(18,{(PT + PB) / 2:.1f}) rotate(-90)" font-size="13" text-anchor="middle" '
        f'fill="#475569">Mean teammate distance (game units)</text>'
    )
    for i, s in enumerate(SERIES):
        x0 = PL + i * 118
        dash = ' stroke-dasharray="6 4"' if s["dashed"] else ""
        parts.append(
            f'<line x1="{x0}" y1="20" x2="{x0 + 24}" y2="20" stroke="{s["color"]}" stroke-width="2.2"{dash}/>'
        )
        parts.append(f'<text x="{x0 + 30}" y="24" font-size="12" text-anchor="start" fill="#475569">{s["label"]}</text>')
    parts.append(
        f'<text x="{PR}" y="24" font-size="12" text-anchor="end" fill="#475569">'
        f'n = {n_rounds_sample:,} rounds, {M} matches</text>'
    )
    parts.append(
        f'<text x="{PL}" y="416" font-size="12" text-anchor="start" fill="#475569">'
        f'Shaded bands: 95% match-cluster bootstrap interval (150 matches, 1,000 resamples, seed 0).</text>'
    )
    parts.append(
        f'<text x="{PL}" y="434" font-size="12" text-anchor="start" fill="#475569">'
        f'Lines drawn only where a series has at least 100 rounds with 2 or more living players on that side.</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


svg = build_svg()

with open(OUT_JSON, "w", encoding="utf-8", newline="\n") as fh:
    json.dump(out, fh, indent=2)
with open(OUT_SVG, "w", encoding="utf-8", newline="\n") as fh:
    fh.write(svg)

# ---------------------------------------------------------------- 6. print for checking
print("step", step, "y range", ymin_t, ymax_t)
print("last plotted second:", last_plotted)
for si, s in enumerate(SERIES):
    print(s["label"])
    for sec in KEY_SECS:
        p_ = point(si, sec)
        print(f"  sec {sec:>2}  n={p_['n_rounds']:>5}  mean={p_['mean']}  ci=[{p_['ci_low']}, {p_['ci_high']}]  alive={p_['mean_alive_teammates']}  plotted={p_['plotted']}")
for side in ("CT", "T"):
    print(side, "won minus lost")
    for sec in KEY_SECS:
        d = diff[side][sec]
        print(f"  sec {sec:>2}  diff={d['diff']}  ci=[{d['ci_low']}, {d['ci_high']}]  plotted={d['plotted']}")
print("svg bytes", len(svg), "json written:", OUT_JSON, "svg written:", OUT_SVG)
