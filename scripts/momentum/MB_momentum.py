"""MB: momentum across rounds (team-round level), 24-round regulation matches, 2026 tome.

Question
  Does a team's previous round result (prev_won), or its signed streak, predict winning the next round,
  before and after controlling for score lead and round number? Does the association survive once the
  freeze-end equipment and money difference are in (1 Hz sample)?

Population and definitions
  Matches: header max_rounds == 24, round_end rounds form 1..n with n >= 24, n == header total
  (t_starters_score_final + ct_starters_score_final), and the derived team scores equal the header
  totals (team A = starts on T in round 1; sides: rounds 1-12 A on T, 13-24 A on CT; overtime
  6-round blocks, A on T in rounds 25-27 and so on).
  Team-rounds: for each match, round r in 2..24 except 13 (pistol excluded as target), for each of
  the two teams. y = team won round r. prev_won = team won round r-1 (1/0). streak = signed run length
  of the team's results through round r-1, capped at +/-5 (rounds 1..r-1, carried across the swap).
  lead = team rounds won minus opponent rounds won before round r. rnd = round number.
  Rows for the two teams in one round are mirror images in outcome; match-cluster bootstrap is used
  for all intervals (1000 resamples of matches, seed 0).

1 Hz sample (data/state_sample/players_1hz.parquet, freeze-end snapshot seconds_since_live == 0)
  eq_diff = sum of equipment_value_calc (team minus opponent), in $1000s; money_diff likewise.
  Rounds with fewer than 5 players on a side are dropped.

Models
  Logistic regression: RidgeLogit (numpy Newton, standardised L2, C = 1, intercept unpenalised), copied
  from scripts/model/per_tick_eval.py. sklearn and scipy are not used (native crash in this env).
  Coefficients are reported on the raw scale. Predicted curves use marginal standardisation: for each
  signed streak k, set prev_won = (k > 0) and streak = k for every row, average the predicted probability
  over the observed rows. In the equipment model, eq_diff and money_diff are fixed at their sample means
  (the average equipment).

Run (project interpreter, explicit):
  "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\momentum\MB_momentum.py
Writes data\viz\MB.json and data\viz\MB.svg. Reads the tome and state_sample only.
"""

import collections
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
RE_PATH = TOME / "basic_round_end.2026-09-18,2026-10-08" / "dataframe_00000"
HD_PATH = TOME / "header" / "dataframe_00000"
SAMPLE = ROOT / "data" / "state_sample"
OUT_JSON = ROOT / "data" / "viz" / "MB.json"
OUT_SVG = ROOT / "data" / "viz" / "MB.svg"

SEED = 0
N_BOOT = 1000
STREAK_CAP = 5
KS = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
T_CODE, CT_CODE = 2, 3
BAKEOFF_KEPT_N = 1521  # reference only (sufficiency kept set), not used to filter


class RidgeLogit:
    """Standardised L2 logistic regression (C = 1, intercept unpenalised), Newton steps, numpy only.
    Copied from scripts/model/per_tick_eval.py."""

    def fit(self, X, y, C=1.0, iters=30):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        Z = np.hstack([np.ones((len(X), 1)), (X - self.mu) / self.sd])
        lam = np.full(Z.shape[1], 1.0 / C)
        lam[0] = 0.0
        w = np.zeros(Z.shape[1])
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-(Z @ w)))
            g = Z.T @ (p - y) + lam * w
            H = (Z * (p * (1.0 - p))[:, None]).T @ Z + np.diag(lam)
            step = np.linalg.solve(H, g)
            w = w - step
            if np.abs(step).max() < 1e-9:
                break
        self.w = w
        return self

    def predict(self, X):
        Z = np.hstack([np.ones((len(X), 1)), (X - self.mu) / self.sd])
        return 1.0 / (1.0 + np.exp(-(Z @ self.w)))

    def raw(self):
        """Intercept and slopes on the raw feature scale."""
        beta = self.w[1:] / self.sd
        b0 = self.w[0] - np.sum(self.w[1:] * self.mu / self.sd)
        return b0, beta


def side_a(r):
    """Side of team A (started on T) in round r. Regulation: T for 1-12, CT for 13-24.
    Overtime: swap every 3 rounds starting at round 25 (A on T in 25-27)."""
    if r <= 12:
        return T_CODE
    if r <= 24:
        return CT_CODE
    return T_CODE if ((r - 25) // 3) % 2 == 0 else CT_CODE


def upd(s, won):
    if won:
        s = s + 1 if s > 0 else 1
    else:
        s = s - 1 if s < 0 else -1
    return max(-STREAK_CAP, min(STREAK_CAP, s))


def pct(a):
    lo, hi = np.nanpercentile(a, [2.5, 97.5], axis=0)
    return lo, hi


def build_matches():
    h = (pl.scan_parquet(HD_PATH)
         .select(["key", "max_rounds", "t_starters_score_final", "ct_starters_score_final"])
         .rename({"key": "match_key"}).collect())
    re = pl.scan_parquet(RE_PATH).select(["match_key", "round", "winner_team_code"]).collect()
    wins = {}
    for mk, rd, wc in zip(re["match_key"].to_list(), re["round"].to_list(), re["winner_team_code"].to_list()):
        wins.setdefault(mk, {})[int(rd)] = int(wc)
    flow = collections.Counter()
    keep = []
    decided_all = 0
    for mk, mr, tf, cf in zip(h["match_key"].to_list(), h["max_rounds"].to_list(),
                              h["t_starters_score_final"].to_list(), h["ct_starters_score_final"].to_list()):
        flow["header_rows"] += 1
        if mr != 24:
            flow["excluded_not_24_round"] += 1
            continue
        flow["matches_24_round"] += 1
        rd = wins.get(mk, {})
        if not rd:
            flow["no_round_end_rows"] += 1
            continue
        n = max(rd)
        if sorted(rd) != list(range(1, n + 1)):
            flow["round_gap"] += 1
            continue
        if n != tf + cf:
            flow["round_count_ne_header_total"] += 1
            continue
        # A 24-round (MR12) match ends at 13 round wins, so n is usually below 24. Complete matches only.
        if tf != cf:
            decided_all += 1
        else:
            flow["tied_header_total"] += 1
        if n <= 24:
            a_wins = sum(1 for r in range(1, n + 1) if rd[r] == side_a(r))
            if not (a_wins == tf and n - a_wins == cf):
                flow["regulation_team_score_ne_header"] += 1
                continue
            flow["regulation_team_score_ok"] += 1
        else:
            flow["overtime_match"] += 1
        flow["primary_matches"] += 1
        keep.append((mk, n, tf, cf, rd))
    flow["decided_match_winner_matches_all_24_round"] = decided_all
    return keep, flow


def build_rows(keep):
    """One tuple per team-round: (match_idx, team 0=A 1=B, round, y, prev_won, streak, lead, match_key)."""
    rows = []
    for mi, (mk, n, tf, cf, rd) in enumerate(keep):
        sA = sB = 0
        cA = cB = 0
        for r in range(1, min(n, 24) + 1):
            aw = 1 if rd[r] == side_a(r) else 0
            bw = 1 - aw
            if r >= 2 and r != 13:
                rows.append((mi, 0, r, aw, 1 if sA > 0 else 0, sA, cA - cB, mk))
                rows.append((mi, 1, r, bw, 1 if sB > 0 else 0, sB, cB - cA, mk))
            cA += aw
            cB += bw
            sA = upd(sA, aw)
            sB = upd(sB, bw)
    return rows


def load_equipment():
    """(match_key, round, team_code) -> (equipment sum, money sum, n_players) at freeze end."""
    df = (pl.scan_parquet(SAMPLE / "players_1hz.parquet")
          .filter(pl.col("seconds_since_live") == 0)
          .group_by(["match_key", "round", "team_code"])
          .agg(pl.col("equipment_value_calc").sum().alias("eq"),
               pl.col("money").sum().alias("money"),
               pl.len().alias("n"))
          .collect())
    out = {}
    for mk, rd, tc, eq, mo, n in zip(df["match_key"].to_list(), df["round"].to_list(),
                                     df["team_code"].to_list(), df["eq"].to_list(),
                                     df["money"].to_list(), df["n"].to_list()):
        out[(mk, int(rd), int(tc))] = (float(eq), float(mo), int(n))
    return out


def attach_equipment(rows, eqd, sample_keys):
    """Adds eq_diff and money_diff ($1000s) to the 1 Hz sample team-rounds only.
    Returns rows (sample matches only, with equipment) and coverage counts for sample team-rounds."""
    cov = collections.Counter()
    out = []
    for (mi, team, r, y, prev, streak, lead, mk) in rows:
        if mk not in sample_keys:
            continue
        cov["team_rounds_in_1hz_matches"] += 1
        sx = side_a(r) if team == 0 else (T_CODE + CT_CODE - side_a(r))
        so = T_CODE + CT_CODE - sx
        a = eqd.get((mk, r, sx))
        b = eqd.get((mk, r, so))
        if a is None or b is None:
            cov["team_round_no_freeze_snapshot"] += 1
            continue
        if a[2] < 5 or b[2] < 5:
            cov["team_round_fewer_than_5_players"] += 1
            continue
        eqv = (a[0] - b[0]) / 1000.0
        mov = (a[1] - b[1]) / 1000.0
        cov["team_round_with_equipment"] += 1
        out.append((mi, team, r, y, prev, streak, lead, mk, eqv, mov))
    return out, cov, None


def design(rowsel):
    """Dense design: columns prev_won, streak, lead, rnd, eq_diff, money_diff."""
    mis = np.array([r[0] for r in rowsel], dtype=np.int64)
    uniq, dense = np.unique(mis, return_inverse=True)
    M = len(uniq)
    starts = np.searchsorted(dense, np.arange(M), side="left")
    ends = np.searchsorted(dense, np.arange(M), side="right")
    X = np.array([[r[4], r[5], r[6], r[2], r[8], r[9]] for r in rowsel], dtype=np.float64)
    y = np.array([r[3] for r in rowsel], dtype=np.float64)
    return {"X": X, "y": y, "M": M, "starts": starts, "ends": ends, "n": len(rowsel)}


def fit_block(X, y, cols):
    """Coefficients (raw scale, intercept first) and the predicted streak curve (marginal)."""
    Xc = X[:, cols]
    m = RidgeLogit().fit(Xc, y)
    b0, beta = m.raw()
    coefs = np.concatenate([[b0], beta])
    base = Xc.copy()
    pi, si = cols.index(0), cols.index(1)
    if 4 in cols:
        base[:, cols.index(4)] = X[:, 4].mean()
        base[:, cols.index(5)] = X[:, 5].mean()
    curve = []
    for k in KS:
        Z = base.copy()
        Z[:, pi] = 1.0 if k > 0 else 0.0
        Z[:, si] = k
        curve.append(float(m.predict(Z).mean()))
    return coefs, np.array(curve)


def stats_full(X, y):
    prev = X[:, 0]
    rnd = X[:, 3]
    p1 = y[prev == 1].mean()
    p0 = y[prev == 0].mean()
    r2 = y[(rnd == 2) & (prev == 1)].mean()
    c, _ = fit_block(X, y, [0, 1, 2, 3])
    return np.concatenate([[p1, p0, p1 - p0, r2], c])


def stats_1hz(X, y):
    prev = X[:, 0]
    p1 = y[prev == 1].mean()
    p0 = y[prev == 0].mean()
    by_k = np.array([y[X[:, 1] == k].mean() if np.any(X[:, 1] == k) else np.nan for k in KS])
    c2, cur2 = fit_block(X, y, [0, 1, 2, 3])
    c3, cur3 = fit_block(X, y, [0, 1, 2, 3, 4, 5])
    eq_p1, eq_p0 = X[prev == 1, 4].mean(), X[prev == 0, 4].mean()
    mo_p1, mo_p0 = X[prev == 1, 5].mean(), X[prev == 0, 5].mean()
    return np.concatenate([[p1, p0, p1 - p0], by_k, c2, cur2, c3, cur3,
                           [eq_p1, eq_p0, eq_p1 - eq_p0, mo_p1, mo_p0, mo_p1 - mo_p0]])


def bootstrap(D, fn, seed=SEED):
    """Match-cluster bootstrap: N_BOOT draws of matches with replacement; fn is applied to each draw."""
    rng = np.random.default_rng(seed)
    M = D["M"]
    ranges = [np.arange(s, e) for s, e in zip(D["starts"], D["ends"])]
    draws = rng.integers(0, M, size=(N_BOOT, M))
    out = []
    for b in range(N_BOOT):
        rows = np.concatenate([ranges[m] for m in draws[b]])
        out.append(fn(D["X"][rows], D["y"][rows]))
    return np.array(out)


def within_match_lpm(D):
    """Linear probability model with match fixed effects (within-match demeaning), point estimate."""
    gid = np.repeat(np.arange(D["M"]), D["ends"] - D["starts"])
    cnt = np.bincount(gid, minlength=D["M"]).astype(np.float64)
    cols = [0, 1, 2, 3]
    Xc = D["X"][:, cols]
    means = np.stack([np.bincount(gid, weights=Xc[:, j], minlength=D["M"]) / cnt for j in range(len(cols))], 1)
    ym = np.bincount(gid, weights=D["y"], minlength=D["M"]) / cnt
    Xd = Xc - means[gid]
    yd = D["y"] - ym[gid]
    beta, *_ = np.linalg.lstsq(Xd, yd, rcond=None)
    return beta


def ci(boot, idx):
    lo, hi = np.percentile(boot[:, idx], [2.5, 97.5])
    return [float(lo), float(hi)]


def svg_chart(series, xlabels, ylo, yhi, title, subtitle):
    W, H = 800, 450
    L, R, T, B = 78, 230, 72, 62
    pw, ph = W - L - R, H - T - B
    n = len(xlabels)

    def X(i):
        return L + pw * i / (n - 1)

    def Y(v):
        return T + ph * (yhi - v) / (yhi - ylo)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
           f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>',
           f'<text x="{L}" y="26" font-family="sans-serif" font-size="16" font-weight="bold" fill="#0f172a">{title}</text>',
           f'<text x="{L}" y="46" font-family="sans-serif" font-size="11" fill="#475569">{subtitle}</text>']
    step = 0.1
    v = np.ceil(ylo / step) * step
    while v <= yhi + 1e-9:
        y = Y(v)
        out.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L + pw}" y2="{y:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        out.append(f'<text x="{L - 8}" y="{y + 4:.1f}" font-family="sans-serif" font-size="11" text-anchor="end" fill="#475569">{v * 100:.0f}%</text>')
        v += step
    for i, lab in enumerate(xlabels):
        out.append(f'<text x="{X(i):.1f}" y="{T + ph + 18}" font-family="sans-serif" font-size="11" text-anchor="middle" fill="#475569">{lab}</text>')
    out.append(f'<line x1="{L}" y1="{T + ph}" x2="{L + pw}" y2="{T + ph}" stroke="#475569" stroke-width="1"/>')
    out.append(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{T + ph}" stroke="#475569" stroke-width="1"/>')
    out.append(f'<text x="{L + pw / 2:.1f}" y="{T + ph + 40}" font-family="sans-serif" font-size="12" text-anchor="middle" fill="#0f172a">Signed streak before the round (team wins +, losses -; capped at 5)</text>')
    out.append(f'<text transform="translate(18 {T + ph / 2:.1f}) rotate(-90)" font-family="sans-serif" font-size="12" text-anchor="middle" fill="#0f172a">Win rate of the team in the round</text>')
    for s in series:
        if s.get("band") is not None:
            lo, hi = s["band"]
            pts = [f"{X(i):.1f},{Y(hi[i]):.1f}" for i in range(n)] + [f"{X(i):.1f},{Y(lo[i]):.1f}" for i in reversed(range(n))]
            out.append(f'<polygon points="{" ".join(pts)}" fill="{s["color"]}" fill-opacity="0.14" stroke="none"/>')
    for s in series:
        pts = " ".join(f"{X(i):.1f},{Y(s['values'][i]):.1f}" for i in range(n))
        dash = f' stroke-dasharray="{s["dash"]}"' if s.get("dash") else ""
        out.append(f'<polyline points="{pts}" fill="none" stroke="{s["color"]}" stroke-width="2.2"{dash}/>')
        for i in range(n):
            out.append(f'<circle cx="{X(i):.1f}" cy="{Y(s["values"][i]):.1f}" r="3.2" fill="{s["color"]}"/>')
    lx, ly = L + pw + 22, T + 10
    for j, s in enumerate(series):
        yy = ly + j * 46
        dash = f' stroke-dasharray="{s["dash"]}"' if s.get("dash") else ""
        out.append(f'<line x1="{lx}" y1="{yy}" x2="{lx + 26}" y2="{yy}" stroke="{s["color"]}" stroke-width="2.2"{dash}/>')
        out.append(f'<text x="{lx + 34}" y="{yy + 4}" font-family="sans-serif" font-size="11" fill="#0f172a">{s["name"]}</text>')
        for k, line in enumerate(s["sub"]):
            out.append(f'<text x="{lx + 34}" y="{yy + 17 + 12 * k}" font-family="sans-serif" font-size="10" fill="#475569">{line}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main():
    t0 = time.time()
    print("== MB: momentum across rounds ==", flush=True)
    keep, flow = build_matches()
    print(f"matches (flow): {dict(flow)}", flush=True)
    rows = build_rows(keep)
    eqd = load_equipment()
    sample_keys = {k[0] for k in eqd}
    rows_1, cov, _ = attach_equipment(rows, eqd, sample_keys)
    rows_all = [(*r, float("nan"), float("nan")) for r in rows]
    print(f"team-round rows (primary, full): {len(rows_all)}; 1Hz team-rounds with equipment: {len(rows_1)}; "
          f"coverage {dict(cov)}", flush=True)

    one_hz_matches_in_keep = sum(1 for (mk, *_rest) in keep if mk in sample_keys)
    target_rounds_sample = sum(1 for (mk, n, tf, cf, rd) in keep if mk in sample_keys
                               for r in range(2, min(n, 24) + 1) if r != 13)
    rounds_with_equipment = len({(r[7], r[2]) for r in rows_1})
    print(f"1Hz matches in primary set: {one_hz_matches_in_keep}; 1Hz team-round targets {target_rounds_sample}; "
          f"round targets with equipment (both teams) {rounds_with_equipment}", flush=True)

    D_full = design(rows_all)
    D_1 = design(rows_1)
    print(f"full design: rows {D_full['n']}, matches {D_full['M']}; 1Hz design: rows {D_1['n']}, matches {D_1['M']}", flush=True)

    # point estimates
    pt_full = stats_full(D_full["X"], D_full["y"])
    pt_1 = stats_1hz(D_1["X"], D_1["y"])
    # extra point models
    X, y = D_full["X"], D_full["y"]
    m_a = RidgeLogit().fit(X[:, [0, 2, 3]], y)
    m_b = RidgeLogit().fit(X[:, [1, 2, 3]], y)
    a_b0, a_beta = m_a.raw()
    b_b0, b_beta = m_b.raw()
    lpm = within_match_lpm(D_full)
    print("point estimates done", flush=True)

    print("bootstrap full sample...", flush=True)
    B_full = bootstrap(D_full, stats_full)
    print("bootstrap 1Hz...", flush=True)
    B_1 = bootstrap(D_1, stats_1hz)
    print(f"bootstrap done in {time.time() - t0:.1f}s", flush=True)

    # index maps
    f = {"p1": 0, "p0": 1, "diff": 2, "r2_p1": 3}
    f_coef = {"intercept": 0, "prev_won": 1, "streak": 2, "lead": 3, "round": 4}
    g = {"p1": 0, "p0": 1, "diff": 2}
    g_by_k = slice(3, 3 + 10)
    g_c2 = slice(13, 13 + 5)
    g_cur2 = slice(18, 18 + 10)
    g_c3 = slice(28, 28 + 7)
    g_cur3 = slice(35, 35 + 10)
    g_eq = {"eq_p1": 45, "eq_p0": 46, "eq_diff": 47, "mo_p1": 48, "mo_p0": 49, "mo_diff": 50}

    def cp(arr, idx):
        return float(arr[idx])

    crude_full = {"p_win_given_prev_won": cp(pt_full, 0), "p_win_given_prev_lost": cp(pt_full, 1),
                  "diff": cp(pt_full, 2),
                  "ci95_p_win_given_prev_won": ci(B_full, 0), "ci95_p_win_given_prev_lost": ci(B_full, 1),
                  "ci95_diff": ci(B_full, 2), "team_rounds": D_full["n"],
                  "round2_p_win_given_won_round1": cp(pt_full, 3), "ci95_round2_p_win_given_won_round1": ci(B_full, 3)}
    lr_full_names = ["intercept", "prev_won", "streak", "lead", "round"]
    lr_full = {}
    for j, nm in enumerate(lr_full_names):
        idx = 4 + j
        lr_full[nm] = {"coef": cp(pt_full, idx), "ci95": ci(B_full, idx),
                       "odds_ratio": float(np.exp(pt_full[idx])), "or_ci95": [float(np.exp(ci(B_full, idx)[0])), float(np.exp(ci(B_full, idx)[1]))]}
    lr_full["prev_won_only_model_with_lead_round_point"] = float(a_beta[0])
    lr_full["streak_only_model_with_lead_round_point"] = float(b_beta[0])
    lr_full["within_match_LPM_point"] = {"prev_won": float(lpm[0]), "streak": float(lpm[1]),
                                         "lead": float(lpm[2]), "round": float(lpm[3])}

    crude_1 = {"p_win_given_prev_won": cp(pt_1, 0), "p_win_given_prev_lost": cp(pt_1, 1), "diff": cp(pt_1, 2),
               "ci95_p_win_given_prev_won": ci(B_1, 0), "ci95_p_win_given_prev_lost": ci(B_1, 1),
               "ci95_diff": ci(B_1, 2), "team_rounds": D_1["n"]}

    def block(name_list, start, boot_start):
        out = {}
        for j, nm in enumerate(name_list):
            idx = start + j
            out[nm] = {"coef": float(pt_1[idx]), "ci95": ci(B_1, idx)}
        return out

    m2_names = ["intercept", "prev_won", "streak", "lead", "round"]
    m3_names = ["intercept", "prev_won", "streak", "lead", "round", "eq_diff_per_1000", "money_diff_per_1000"]
    m2 = block(m2_names, 13, None)
    m3 = block(m3_names, 28, None)
    for nm in m2:
        m2[nm]["odds_ratio"] = float(np.exp(m2[nm]["coef"]))
    for nm in m3:
        m3[nm]["odds_ratio"] = float(np.exp(m3[nm]["coef"]))

    curve_k = {}
    by_k = pt_1[3:13]
    lo_by_k = np.nanpercentile(B_1[:, 3:13], 2.5, axis=0)
    hi_by_k = np.nanpercentile(B_1[:, 3:13], 97.5, axis=0)
    cur2 = pt_1[18:28]
    cur2_lo = np.percentile(B_1[:, 18:28], 2.5, axis=0)
    cur2_hi = np.percentile(B_1[:, 18:28], 97.5, axis=0)
    cur3 = pt_1[35:45]
    cur3_lo = np.percentile(B_1[:, 35:45], 2.5, axis=0)
    cur3_hi = np.percentile(B_1[:, 35:45], 97.5, axis=0)
    for i, k in enumerate(KS):
        curve_k[str(k)] = {"crude_1hz": float(by_k[i]), "crude_1hz_ci95": [float(lo_by_k[i]), float(hi_by_k[i])],
                           "model_lead_round_1hz": float(cur2[i]), "model_lead_round_1hz_ci95": [float(cur2_lo[i]), float(cur2_hi[i])],
                           "model_with_equipment_at_avg_1hz": float(cur3[i]), "model_with_equipment_at_avg_1hz_ci95": [float(cur3_lo[i]), float(cur3_hi[i])]}

    eq_by_prev = {"eq_diff_mean_after_win_k": float(pt_1[45]), "eq_diff_mean_after_loss_k": float(pt_1[46]),
                  "diff": float(pt_1[47]), "ci95_diff": ci(B_1, 47),
                  "money_diff_mean_after_win_k": float(pt_1[48]), "money_diff_mean_after_loss_k": float(pt_1[49]),
                  "money_diff": float(pt_1[50]), "ci95_money_diff": ci(B_1, 50)}

    # SVG (lines: crude 1Hz, lead+round model, lead+round+equipment at average)
    labels = [("+" + str(k) if k > 0 else str(k)) for k in KS]
    series = [
        {"name": "No controls (crude)", "color": "#475569", "dash": "5 4",
         "values": list(by_k), "band": None,
         "sub": [f"1 Hz sample, n = {D_1['n']} team-rounds"]},
        {"name": "Controls: lead, round", "color": "#2563eb", "values": list(cur2),
         "band": (cur2_lo, cur2_hi), "sub": ["model: prev_won, streak, lead, round"]},
        {"name": "Controls + equipment at avg", "color": "#d97706", "values": list(cur3),
         "band": (cur3_lo, cur3_hi), "sub": ["adds eq and money diff at sample mean"]},
    ]
    all_vals = np.concatenate([by_k, cur2, cur3, lo_by_k, cur2_lo, cur3_lo, hi_by_k, cur2_hi, cur3_hi])
    lo_v = max(0.0, np.floor((np.nanmin(all_vals) - 0.02) / 0.1) * 0.1)
    hi_v = min(1.0, np.ceil((np.nanmax(all_vals) + 0.02) / 0.1) * 0.1)
    svg = svg_chart(series, labels, lo_v, hi_v,
                    "Win rate by signed streak: crude vs controlled",
                    f"Team-rounds, rounds 2-24 except 13, complete 24-round matches; 1 Hz sample ({D_1['M']} matches); bands 95% match-cluster bootstrap")
    OUT_SVG.write_text(svg, encoding="utf-8")

    out = {
        "id": "MB",
        "title": "Momentum across rounds (team-round level)",
        "population": {"header_flow": dict(flow), "primary_matches": len(keep),
                       "primary_matches_reference_kept_set": BAKEOFF_KEPT_N,
                       "match_winner_decided_matches_all_24_round": flow["decided_match_winner_matches_all_24_round"],
                       "team_rounds_full": D_full["n"],
                       "one_hz_matches": D_1["M"], "one_hz_team_rounds": D_1["n"],
                       "one_hz_coverage": dict(cov),
                       "one_hz_team_rounds_possible": 2 * target_rounds_sample},
        "crude_full_sample": crude_full,
        "logit_full_sample_prev_won_streak_lead_round": lr_full,
        "crude_1hz": crude_1,
        "logit_1hz_no_equipment": m2,
        "logit_1hz_with_equipment_and_money": m3,
        "streak_curve": curve_k,
        "equipment_by_prior_result_1hz": eq_by_prev,
        "bootstrap": {"n": N_BOOT, "seed": SEED, "unit": "match"},
        "units": {"eq_diff_per_1000": "sum of equipment_value_calc, team minus opponent, USD thousands",
                  "money_diff_per_1000": "sum of money at freeze end, team minus opponent, USD thousands",
                  "lead": "team rounds won minus opponent rounds won before the round"},
        "run_seconds": round(time.time() - t0, 1),
    }
    OUT_JSON.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out, indent=1, default=float)[:6000], flush=True)
    print(f"wrote {OUT_JSON} and {OUT_SVG}; done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
