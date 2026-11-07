"""S6: bunching chart.

Share of alive players with a teammate within 500 game units (2D, x/y only),
by seconds since live (0-90), split by side (CT/T) and by whether that side won.
Also: correlation between each team-round's mean bunching over the first 30 s
and that team winning the round, with a match-cluster bootstrap interval.

Descriptive only. Not causal.

Outputs: data/viz/S6.json (series and key numbers), data/viz/S6.svg (chart).
Run with: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\viz\\S6.py
"""

import json
import math
import warnings
from pathlib import Path

import numpy as np
import polars as pl

warnings.filterwarnings("ignore", category=RuntimeWarning)

REPO = Path(__file__).resolve().parents[2]
PLAYERS = REPO / "data" / "state_sample" / "players_1hz.parquet"
ROUNDS = REPO / "data" / "state_sample" / "rounds.parquet"
OUT_DIR = REPO / "data" / "viz"
OUT_JSON = OUT_DIR / "S6.json"
OUT_SVG = OUT_DIR / "S6.svg"

SEC_MAX = 90
WINDOW = 30
RADIUS = 500.0
N_BOOT = 1000
SEED = 0
MIN_ROUNDS = 30  # a (series, second) point is drawn only if this many rounds contribute
NSEC = SEC_MAX + 1

# (team_code, won, label, colour, dashed)  team_code 3 = CT, 2 = T
SERIES = [
    (3, True, "CT won", "#2563eb", False),
    (3, False, "CT lost", "#2563eb", True),
    (2, True, "T won", "#d97706", False),
    (2, False, "T lost", "#d97706", True),
]
NS = len(SERIES)
KEY = ["mi", "round", "seconds_since_live", "team_code"]


def load():
    p = (
        pl.read_parquet(
            PLAYERS,
            columns=["match_key", "round", "seconds_since_live", "player_id", "team_code", "x_pos", "y_pos"],
        )
        .filter(pl.col("seconds_since_live").is_between(0, SEC_MAX))
        .with_columns(pl.col("x_pos").cast(pl.Float64), pl.col("y_pos").cast(pl.Float64))
    )
    rd = pl.read_parquet(ROUNDS, columns=["match_key", "round", "ct_win"]).with_columns(
        pl.col("round").cast(pl.Int32), pl.col("ct_win").cast(pl.Int8)
    )
    matches = sorted(rd["match_key"].unique().to_list())
    mmap = pl.DataFrame(
        {"match_key": matches, "mi": pl.Series("mi", list(range(len(matches))), dtype=pl.Int32)}
    )
    n_before = p.height
    p = p.join(mmap, on="match_key", how="inner").join(rd, on=["match_key", "round"], how="inner")
    assert p.height == n_before, "player rows lost in join to rounds"
    return p, rd, len(matches)


def flag_bunched(p):
    """Mark each alive player-second that has a teammate (same side, same snapshot) within RADIUS."""
    a = p.select(KEY + [pl.col("player_id").alias("pid_a"), pl.col("x_pos").alias("xa"), pl.col("y_pos").alias("ya")])
    b = p.select(KEY + [pl.col("player_id").alias("pid_b"), pl.col("x_pos").alias("xb"), pl.col("y_pos").alias("yb")])
    pairs = (
        a.join(b, on=KEY, how="inner")
        .filter(pl.col("pid_a") < pl.col("pid_b"))
        .filter((pl.col("xa") - pl.col("xb")) ** 2 + (pl.col("ya") - pl.col("yb")) ** 2 <= RADIUS**2)
    )
    close = pl.concat(
        [
            pairs.select(KEY + [pl.col("pid_a").alias("player_id")]),
            pairs.select(KEY + [pl.col("pid_b").alias("player_id")]),
        ]
    ).unique()
    close = close.with_columns(pl.lit(1, dtype=pl.Int32).alias("close_flag"))
    out = p.join(close, on=KEY + ["player_id"], how="left").with_columns(
        pl.col("close_flag").fill_null(0).alias("bunched")
    )
    assert out.height == p.height, "duplicate rows after bunching join"
    return out.drop("close_flag")


def team_seconds(p):
    return (
        p.group_by(["mi", "round", "seconds_since_live", "team_code", "ct_win"])
        .agg(pl.len().alias("alive"), pl.col("bunched").sum().cast(pl.Int64).alias("bunched"))
        .with_columns(
            (
                ((pl.col("team_code") == 3) & (pl.col("ct_win") == 1))
                | ((pl.col("team_code") == 2) & (pl.col("ct_win") == 0))
            ).alias("won"),
        )
        .with_columns((pl.col("bunched").cast(pl.Float64) / pl.col("alive").cast(pl.Float64)).alias("share"))
    )


def pearson_from_sums(n, sx, sy, sxx, syy, sxy):
    num = n * sxy - sx * sy
    den = np.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy))
    return num / den


def main():
    p, rd, M = load()
    n_rounds = rd.height
    n_matches = M
    print(f"rounds={n_rounds} matches={n_matches} player-second rows(0..{SEC_MAX})={p.height}")

    p = flag_bunched(p)
    ts = team_seconds(p)
    n_player_seconds = int(ts["alive"].sum())

    team = ts["team_code"].to_numpy().astype(np.int64)
    won = ts["won"].to_numpy().astype(bool)
    sec = ts["seconds_since_live"].to_numpy().astype(np.int64)
    mi = ts["mi"].to_numpy().astype(np.int64)
    alive = ts["alive"].to_numpy().astype(np.float64)
    bunched = ts["bunched"].to_numpy().astype(np.float64)
    sidx = np.where(team == 3, 0, 2) + np.where(won, 0, 1)
    assert sidx.min() >= 0 and sidx.max() < NS

    # Per-match sums for the cluster bootstrap
    Bm = np.zeros((M, NS, NSEC))
    Am = np.zeros((M, NS, NSEC))
    Nm = np.zeros((M, NS, NSEC))
    np.add.at(Bm, (mi, sidx, sec), bunched)
    np.add.at(Am, (mi, sidx, sec), alive)
    np.add.at(Nm, (mi, sidx, sec), 1.0)
    assert (Bm <= Am + 1e-9).all(), "bunched exceeds alive"
    assert (Bm >= 0).all() and (Am >= 0).all() and (Nm >= 0).all()

    Bt, At, Nt = Bm.sum(0), Am.sum(0), Nm.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        share_pt = np.where(At > 0, Bt / At, np.nan)
        mean_alive = np.where(Nt > 0, At / Nt, np.nan)

    # Match-cluster bootstrap: resample matches with replacement, 1000 times, seed 0
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, M, size=(N_BOOT, M))
    W = np.zeros((N_BOOT, M))
    np.add.at(W, (np.repeat(np.arange(N_BOOT), M), idx.ravel()), 1.0)

    # einsum instead of @: the BLAS matmul path crashes this interpreter on Windows (fault 0xc06d007f)
    Bb = np.einsum("bm,mk->bk", W, Bm.reshape(M, -1)).reshape(N_BOOT, NS, NSEC)
    Ab = np.einsum("bm,mk->bk", W, Am.reshape(M, -1)).reshape(N_BOOT, NS, NSEC)
    with np.errstate(divide="ignore", invalid="ignore"):
        Rb = np.where(Ab > 0, Bb / Ab, np.nan)
    ci_lo, ci_hi = np.nanpercentile(Rb, [2.5, 97.5], axis=0)

    valid = Nt >= MIN_ROUNDS
    # Sanity: intervals ordered and points inside them where drawn
    bad = valid & ~((ci_lo <= share_pt + 1e-12) & (share_pt <= ci_hi + 1e-12))
    print("points outside own interval:", int(bad.sum()))
    assert not bad.any(), "point estimate outside its bootstrap interval"

    # Window (0 to WINDOW-1 s) pooled bunching per series, and won-minus-lost differences
    Bw_b = Bb[:, :, :WINDOW].sum(-1)
    Aw_b = Ab[:, :, :WINDOW].sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        wshare_b = Bw_b / Aw_b
    wshare_pt = Bt[:, :WINDOW].sum(-1) / At[:, :WINDOW].sum(-1)
    wshare_lo, wshare_hi = np.nanpercentile(wshare_b, [2.5, 97.5], axis=0)
    diff_ct_b = wshare_b[:, 0] - wshare_b[:, 1]
    diff_t_b = wshare_b[:, 2] - wshare_b[:, 3]
    diff_ct_pt = wshare_pt[0] - wshare_pt[1]
    diff_t_pt = wshare_pt[2] - wshare_pt[3]
    diff_ct_lo, diff_ct_hi = np.nanpercentile(diff_ct_b, [2.5, 97.5])
    diff_t_lo, diff_t_hi = np.nanpercentile(diff_t_b, [2.5, 97.5])

    # Team-round level: mean of per-second bunching over seconds 0..WINDOW-1 (seconds with >=1 alive player)
    tr = (
        ts.filter(pl.col("seconds_since_live") < WINDOW)
        .group_by(["mi", "round", "team_code", "won"])
        .agg(pl.col("share").mean().alias("x"), pl.len().alias("n_sec"))
    )
    x = tr["x"].to_numpy().astype(np.float64)
    y = tr["won"].to_numpy().astype(np.float64)
    mi_tr = tr["mi"].to_numpy().astype(np.int64)
    n_team_rounds = tr.height
    S = np.stack(
        [
            np.bincount(mi_tr, minlength=M).astype(np.float64),
            np.bincount(mi_tr, weights=x, minlength=M),
            np.bincount(mi_tr, weights=y, minlength=M),
            np.bincount(mi_tr, weights=x * x, minlength=M),
            np.bincount(mi_tr, weights=y * y, minlength=M),
            np.bincount(mi_tr, weights=x * y, minlength=M),
        ],
        axis=1,
    )
    Q = np.einsum("bm,mk->bk", W, S)
    r_b = pearson_from_sums(Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 3], Q[:, 4], Q[:, 5])
    r_lo, r_hi = np.nanpercentile(r_b, [2.5, 97.5])
    tot = S.sum(0)
    r_pt = float(pearson_from_sums(tot[0], tot[1], tot[2], tot[3], tot[4], tot[5]))
    # independent check of the sums-based Pearson r without BLAS (plain Python loop over team-rounds)
    xs, ys = x.tolist(), y.tolist()
    n_ = len(xs)
    mx_, my_ = sum(xs) / n_, sum(ys) / n_
    cov_ = sum((a - mx_) * (b - my_) for a, b in zip(xs, ys))
    r_loop = cov_ / math.sqrt(sum((a - mx_) ** 2 for a in xs) * sum((b - my_) ** 2 for b in ys))
    assert abs(r_loop - r_pt) < 1e-9, "pearson formula mismatch"
    assert r_lo <= r_pt <= r_hi, "correlation point outside its interval"

    def corr_block(mask):
        """Team-round Pearson r for the rows selected by mask, with the same match-cluster draws W."""
        Sm = np.stack(
            [
                np.bincount(mi_tr[mask], minlength=M).astype(np.float64),
                np.bincount(mi_tr[mask], weights=x[mask], minlength=M),
                np.bincount(mi_tr[mask], weights=y[mask], minlength=M),
                np.bincount(mi_tr[mask], weights=x[mask] ** 2, minlength=M),
                np.bincount(mi_tr[mask], weights=y[mask] ** 2, minlength=M),
                np.bincount(mi_tr[mask], weights=x[mask] * y[mask], minlength=M),
            ],
            axis=1,
        )
        tm = Sm.sum(0)
        rp = float(pearson_from_sums(tm[0], tm[1], tm[2], tm[3], tm[4], tm[5]))
        Qm = np.einsum("bm,mk->bk", W, Sm)
        rb = pearson_from_sums(Qm[:, 0], Qm[:, 1], Qm[:, 2], Qm[:, 3], Qm[:, 4], Qm[:, 5])
        lo_, hi_ = np.nanpercentile(rb, [2.5, 97.5])
        def _f(v):
            return round(float(v), 6) if np.isfinite(v) else None

        return {"n_team_rounds": int(mask.sum()), "r": _f(rp), "ci_lo": _f(lo_), "ci_hi": _f(hi_)}

    tr_team = tr["team_code"].to_numpy()
    within = {"CT": corr_block(tr_team == 3), "T": corr_block(tr_team == 2)}
    print("within-side r", within)

    mean_x_by_series = []
    for s in range(NS):
        sel = y == 1.0 if SERIES[s][1] else y == 0.0
        sel = sel & ((tr["team_code"].to_numpy() == SERIES[s][0]))
        mean_x_by_series.append(float(x[sel].mean()))

    # ----- JSON -----
    def clean(v):
        if v is None:
            return None
        if isinstance(v, (float, np.floating)):
            return None if (math.isnan(float(v)) or math.isinf(float(v))) else round(float(v), 6)
        return v

    series_out = []
    for s, (tc, won_flag, label, colour, dashed) in enumerate(SERIES):
        pts = []
        for t in range(NSEC):
            v = valid[s, t]
            pts.append(
                {
                    "sec": t,
                    "share": clean(share_pt[s, t]) if v else None,
                    "ci_lo": clean(ci_lo[s, t]) if v else None,
                    "ci_hi": clean(ci_hi[s, t]) if v else None,
                    "n_rounds": int(Nt[s, t]),
                    "n_player_seconds": int(At[s, t]),
                    "n_bunched": int(Bt[s, t]),
                    "mean_alive": clean(mean_alive[s, t]),
                }
            )
        series_out.append(
            {
                "label": label,
                "team": "CT" if tc == 3 else "T",
                "side_won": won_flag,
                "colour": colour,
                "dashed": dashed,
                "window_share": clean(wshare_pt[s]),
                "window_ci_lo": clean(wshare_lo[s]),
                "window_ci_hi": clean(wshare_hi[s]),
                "points": pts,
            }
        )

    def at(s, t):
        return {
            "share": clean(share_pt[s, t]),
            "ci_lo": clean(ci_lo[s, t]),
            "ci_hi": clean(ci_hi[s, t]),
            "n_rounds": int(Nt[s, t]),
        }

    key_points = {
        SERIES[s][2]: {str(t): at(s, t) for t in (0, 15, 30, 60, 90)} for s in range(NS)
    }

    out = {
        "id": "S6",
        "title": "Bunching by side and round outcome",
        "sample": {
            "source": "data/state_sample/players_1hz.parquet + rounds.parquet (1 Hz in-round sample)",
            "matches": n_matches,
            "rounds": n_rounds,
            "rounds_with_both_teams_team_rounds": n_team_rounds,
            "player_second_rows_0_to_90": int(n_player_seconds),
            "days": "2026-10 days 01-07 (per handoff)",
        },
        "definitions": {
            "bunching_share": "for a side at a second: alive players with >=1 same-side teammate within 500 game units (2D x/y, distance <= 500), divided by alive players on that side; pooled over rounds as sum(bunched)/sum(alive)",
            "series": "CT or T side, split by whether that side won the round (ct_win = 1 means CT won)",
            "window": f"seconds 0 to {WINDOW - 1} (first 30 s)",
            "correlation_unit": "team-round: each side in each round, x = mean of its per-second bunching shares over seconds 0-29 (seconds with >=1 alive player), y = 1 if that side won",
        },
        "bootstrap": {
            "method": "match-cluster (resample matches with replacement)",
            "resamples": N_BOOT,
            "seed": SEED,
            "interval": "percentile 2.5 to 97.5",
            "min_rounds_plotted": MIN_ROUNDS,
        },
        "series": series_out,
        "window_0_29s": {
            "CT_won_minus_CT_lost": {"point": clean(diff_ct_pt), "ci_lo": clean(diff_ct_lo), "ci_hi": clean(diff_ct_hi)},
            "T_won_minus_T_lost": {"point": clean(diff_t_pt), "ci_lo": clean(diff_t_lo), "ci_hi": clean(diff_t_hi)},
            "mean_team_round_x_by_series": {SERIES[s][2]: clean(mean_x_by_series[s]) for s in range(NS)},
        },
        "correlation_first30_bunching_vs_win": {
            "unit": "team-round",
            "n_team_rounds": n_team_rounds,
            "r": clean(r_pt),
            "ci_lo": clean(r_lo),
            "ci_hi": clean(r_hi),
            "method": "Pearson correlation (point-biserial, y binary); match-cluster bootstrap, 1000 resamples, seed 0",
            "pooled_over_both_sides": "the r above; T has higher bunching and a lower win rate, so pooling mixes a side-level difference with the within-side association",
            "within_side": within,
        },
        "key_points": key_points,
        "caveats": [
            "Descriptive association only; not causal. Outcome is known only after the round.",
            "Share is mechanically lower when fewer players are alive (a lone survivor cannot be bunched), so late-round values partly reflect attrition.",
            "Pooled shares weight each second by alive player-seconds; rounds that end early drop out, so n thins after about 60 s (see n_rounds).",
            "2D distance uses x/y only (z ignored). The 500-unit radius was not varied; no sensitivity check was run.",
            "Sample is 150 matches from 2026-10 days 01-07. Batch effects (2026-10-08 is most clean rounds per handoff) were not separately checked.",
            "Round 16 of match m_7392c5c6d537a57c2e735e98 is flagged as an anomaly in the handoff; it is included (not separately tested).",
            "Overtime rounds (round_number up to 29) are included.",
            "Each round contributes one row per side; the two rows are linked by the complementary outcome, handled by the match-level bootstrap.",
        ],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, allow_nan=False), encoding="utf-8")

    # ----- SVG -----
    svg = build_svg(share_pt, ci_lo, ci_hi, valid, n_rounds, n_matches)
    OUT_SVG.write_text(svg, encoding="utf-8")

    # ----- console summary for checking -----
    print("n_team_rounds", n_team_rounds, "r", round(r_pt, 4), "CI", round(float(r_lo), 4), round(float(r_hi), 4))
    print("window diffs CT", round(diff_ct_pt, 4), (round(float(diff_ct_lo), 4), round(float(diff_ct_hi), 4)))
    print("window diffs T ", round(diff_t_pt, 4), (round(float(diff_t_lo), 4), round(float(diff_t_hi), 4)))
    for s in range(NS):
        row = []
        for t in (0, 15, 30, 45, 60, 75, 90):
            row.append(f"{t}s:{share_pt[s, t]:.3f}[{ci_lo[s, t]:.3f},{ci_hi[s, t]:.3f}] n={int(Nt[s, t])}")
        print(SERIES[s][2], " | ".join(row))
    print("window share", [round(float(v), 4) for v in wshare_pt])
    print("max n_rounds per second (any series):", int(Nt.max()))
    print("written", OUT_JSON, OUT_SVG)


def build_svg(share_pt, ci_lo, ci_hi, valid, n_rounds, n_matches):
    W_, H_ = 800, 450
    X0, X1 = 74, 784
    Y0, Y1 = 58, 362
    XMAX = SEC_MAX

    vs_lo = [ci_lo[s, t] for s in range(NS) for t in range(NSEC) if valid[s, t]]
    vs_hi = [ci_hi[s, t] for s in range(NS) for t in range(NSEC) if valid[s, t]]
    ymin = max(0.0, math.floor((min(vs_lo) - 0.02) * 20) / 20)
    ymax = min(1.0, math.ceil((max(vs_hi) + 0.02) * 20) / 20)
    step = 0.05 if (ymax - ymin) <= 0.4 else 0.1
    yticks = [round(ymin + i * step, 4) for i in range(int(round((ymax - ymin) / step)) + 1)]

    def sx(t):
        return X0 + t / XMAX * (X1 - X0)

    def sy(v):
        return Y1 - (v - ymin) / (ymax - ymin) * (Y1 - Y0)

    parts = []
    parts.append(f'<svg viewBox="0 0 {W_} {H_}" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">')
    parts.append(f'<rect x="0" y="0" width="{W_}" height="{H_}" fill="#ffffff"/>')

    # gridlines and y tick labels
    for v in yticks:
        y = sy(v)
        parts.append(f'<line x1="{X0}" y1="{y:.1f}" x2="{X1}" y2="{y:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        parts.append(
            f'<text x="{X0 - 8}" y="{y + 4:.1f}" font-size="12" fill="#475569" text-anchor="end">{v * 100:.0f}%</text>'
        )
    # x ticks
    for t in range(0, XMAX + 1, 10):
        x = sx(t)
        parts.append(f'<line x1="{x:.1f}" y1="{Y1}" x2="{x:.1f}" y2="{Y1 + 5}" stroke="#475569" stroke-width="1"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{Y1 + 19}" font-size="12" fill="#475569" text-anchor="middle">{t}</text>'
        )
    # axes
    parts.append(f'<line x1="{X0}" y1="{Y0}" x2="{X0}" y2="{Y1}" stroke="#475569" stroke-width="1"/>')
    parts.append(f'<line x1="{X0}" y1="{Y1}" x2="{X1}" y2="{Y1}" stroke="#475569" stroke-width="1"/>')
    # axis titles
    parts.append(
        f'<text x="{(X0 + X1) / 2:.1f}" y="{Y1 + 40}" font-size="13" fill="#475569" text-anchor="middle">Seconds since live</text>'
    )
    parts.append(
        f'<text transform="translate(22 {(Y0 + Y1) / 2:.1f}) rotate(-90)" font-size="13" fill="#475569" text-anchor="middle">Share with a teammate within 500 units</text>'
    )
    # reference line at the 30 s window
    x30 = sx(WINDOW)
    parts.append(
        f'<line x1="{x30:.1f}" y1="{Y0}" x2="{x30:.1f}" y2="{Y1}" stroke="#94a3b8" stroke-width="1" stroke-dasharray="3 3"/>'
    )
    parts.append(
        f'<text x="{x30 + 5:.1f}" y="{Y0 + 14}" font-size="12" fill="#475569" text-anchor="start">30 s window</text>'
    )

    # bands then lines
    for s in range(NS):
        _, _, label, colour, dashed = SERIES[s]
        segs = []
        cur = []
        for t in range(NSEC):
            if valid[s, t]:
                cur.append(t)
            else:
                if cur:
                    segs.append(cur)
                cur = []
        if cur:
            segs.append(cur)
        for seg in segs:
            if len(seg) < 2:
                continue
            up = " ".join(f"{sx(t):.1f},{sy(ci_hi[s, t]):.1f}" for t in seg)
            lo = " ".join(f"{sx(t):.1f},{sy(ci_lo[s, t]):.1f}" for t in reversed(seg))
            parts.append(f'<polygon points="{up} {lo}" fill="{colour}" fill-opacity="0.14" stroke="none"/>')
    for s in range(NS):
        _, _, label, colour, dashed = SERIES[s]
        segs = []
        cur = []
        for t in range(NSEC):
            if valid[s, t]:
                cur.append(t)
            else:
                if cur:
                    segs.append(cur)
                cur = []
        if cur:
            segs.append(cur)
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        for seg in segs:
            if len(seg) < 2:
                continue
            pts = " ".join(f"{sx(t):.1f},{sy(share_pt[s, t]):.1f}" for t in seg)
            parts.append(
                f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="2.2"{dash} stroke-linejoin="round"/>'
            )

    # legend (top row)
    lx = X0
    ly = 30
    for s in range(NS):
        _, _, label, colour, dashed = SERIES[s]
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        parts.append(
            f'<line x1="{lx}" y1="{ly}" x2="{lx + 26}" y2="{ly}" stroke="{colour}" stroke-width="2.2"{dash}/>'
        )
        parts.append(f'<text x="{lx + 33}" y="{ly + 4}" font-size="12" fill="#475569">{label}</text>')
        lx += 120
    parts.append(
        f'<rect x="{lx + 10}" y="{ly - 8}" width="22" height="14" fill="#475569" fill-opacity="0.14"/>'
    )
    parts.append(f'<text x="{lx + 38}" y="{ly + 4}" font-size="12" fill="#475569">95% interval</text>')

    # caption: sample size and method
    parts.append(
        f'<text x="{X0}" y="424" font-size="12" fill="#475569">n = {n_rounds:,} rounds, {n_matches} matches. Points shown where at least {MIN_ROUNDS} rounds contribute.</text>'
    )
    parts.append(
        f'<text x="{X0}" y="441" font-size="12" fill="#475569">Bands: match-cluster bootstrap, 95%, 1,000 resamples, seed 0. Descriptive association, not causal.</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


if __name__ == "__main__":
    main()
