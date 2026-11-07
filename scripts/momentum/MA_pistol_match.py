"""MA: pistol round to match outcome, 24-round 5v5 matches (association only).

Question
  Does winning the first pistol (round 1) or the second-half pistol (round 13) go with winning the
  match? Are the two pistol results linked? Does a map control move the round 1 odds ratio?

Population (funnel written to data/viz/MA.json)
  1. Header rows, one per match key (1,752).
  2. Duplicate copies: a key whose round_end sequence (round, tick, winner) is identical to an
     earlier key is dropped. The lowest key is kept.
  3. max_rounds == 24 (5v5). Wingman (max_rounds 16, 2v2) is a sensitivity only.
  4. Complete match: number of round_end rounds == header t_starters_score_final +
     ct_starters_score_final (the brief's rule).
  5. Decided winner: the two header final counts differ.
  6. Team identity: team A starts on T, team B on CT. The side-swap rule (copied from
     scripts/model/bakeoff.py swap_expected) must reproduce A = t_starters_score_final and
     B = ct_starters_score_final. Matches that fail are excluded and counted.
  7. Round 1 and round 13 both present in round_end.
  Match winner = team with the larger header final count (equal to derived A > B).

Team rows and clusters
  Each match gives two team rows (A and B). Each pistol has exactly one winning team, so a
  match's team rows have complementary outcomes. The four pistol patterns are:
  won both (team won R1 and R13), won R1 only, won R13 only, lost both.
  Intervals: match-cluster bootstrap (both team rows of a match resampled together),
  1000 resamples, seed 0, percentile 95%.

Odds ratios (both reported)
  Team-row OR (winner vs loser over team rows): for one pistol with k matches won by the
  pistol winner out of N, the team-row 2x2 gives OR = (k/(N-k))^2, because the winner and
  loser rows of each match carry complementary outcomes. This is the brief's "winner vs loser".
  Match-level odds = k/(N-k), one count per match, which is the square root of the team-row OR.

Control
  Team-row logistic regression (RidgeLogit: standardised L2, C = 1, intercept unpenalised,
  copied from scripts/model/per_tick_eval.py because sklearn/scipy crash here). Outcome: team
  wins the match. Covariates: won round 1; map one-hot (reference = most common map; maps with
  fewer than 20 matches pooled as other_rare). Match-cluster bootstrap for intervals.
  Caveat built into this design: each match has one winner row and one loser row, so the
  per-map win totals are fixed and the map coefficients come out exactly zero. The map-adjusted
  logit therefore cannot move the odds ratio. The informative map check is a Mantel-Haenszel
  team-row OR stratified by map (same pooling), plus per-map rates.

Sensitivities
  pistol rounds both player_count == 10; regulation only (no overtime); wingman (pistol 2 =
  round 9, team-consistent matches only); all matches (main + wingman).

Run from the project root:
  "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\momentum\\MA_pistol_match.py
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
BASE = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
RE_PATH = BASE / "basic_round_end.2026-09-18,2026-10-08" / "dataframe_00000"
HDR_PATH = BASE / "header" / "dataframe_00000"
OUT_JSON = ROOT / "data" / "viz" / "MA.json"
OUT_SVG = ROOT / "data" / "viz" / "MA.svg"

SEED = 0
N_BOOT = 1000
T, CT = 2, 3  # round_end.winner_team_code: 2 = T, 3 = CT
MIN_MAP_MATCHES = 20

# Published 2.95 (CS:GO Majors, 334 matches, unverified snippet) is compared, not used.
SNIPPET_OR = 2.95


def swap_expected(r, max_rounds):
    """Copied from scripts/model/bakeoff.py swap_expected (bakeoff not imported here)."""
    return r == max_rounds // 2 or (r > max_rounds and (r - max_rounds - 3) % 6 == 0)


def load():
    re = pl.read_parquet(str(RE_PATH), columns=["match_key", "round", "tick", "winner_team_code",
                                                 "player_count"])
    hdr = pl.read_parquet(str(HDR_PATH), columns=["key", "map_name", "max_rounds",
                                                   "t_starters_score_final",
                                                   "ct_starters_score_final", "match_type"])
    seqs = collections.defaultdict(list)
    for r in re.sort(["match_key", "round", "tick"]).iter_rows(named=True):
        seqs[r["match_key"]].append((int(r["round"]), int(r["tick"]), int(r["winner_team_code"]),
                                     int(r["player_count"])))
    H = {x["key"]: x for x in hdr.iter_rows(named=True)}
    return H, seqs, re.height, hdr.height


def dedupe(seqs):
    groups = collections.defaultdict(list)
    for m in sorted(seqs):
        sig = tuple((r, t, w) for r, t, w, _ in seqs[m])
        groups[sig].append(m)
    drop = set()
    for g in groups.values():
        drop.update(g[1:])
    n_groups = sum(1 for g in groups.values() if len(g) > 1)
    return drop, n_groups


def derive(seq, mr):
    """Team A starts on T, team B on CT. Returns round -> winning team index (0 = A, 1 = B),
    and the derived A and B round totals."""
    side_a = T
    win_team = {}
    a = b = 0
    for r, _, w, _ in seq:
        if w == side_a:
            win_team[r] = 0
            a += 1
        else:
            win_team[r] = 1
            b += 1
        if swap_expected(r, mr):
            side_a = CT if side_a == T else T
    return win_team, a, b


def build(H, seqs, after_dedupe, mr, p2):
    """Funnel and per-match records for one max_rounds value. p2 = second pistol round."""
    funnel = []
    pool = [m for m in after_dedupe if H[m]["max_rounds"] == mr]
    funnel.append({"step": f"max_rounds == {mr}", "remaining": len(pool)})
    comp = [m for m in pool
            if len(seqs[m]) == H[m]["t_starters_score_final"] + H[m]["ct_starters_score_final"]]
    funnel.append({"step": "round_end rounds == header final total", "remaining": len(comp)})
    dec = [m for m in comp if H[m]["t_starters_score_final"] != H[m]["ct_starters_score_final"]]
    funnel.append({"step": "header finals differ (decided winner)", "remaining": len(dec)})
    recs, team_bad, pistol_missing = [], [], []
    for m in dec:
        h = H[m]
        seq = seqs[m]
        wt, a, b = derive(seq, mr)
        if (a, b) != (h["t_starters_score_final"], h["ct_starters_score_final"]):
            team_bad.append(m)
            continue
        pc = {r: p for r, _, _, p in seq}
        if 1 not in wt or p2 not in wt:
            pistol_missing.append(m)
            continue
        rw = {r: w for r, _, w, _ in seq}
        recs.append({
            "match": m, "mr": mr, "n_rounds": len(seq), "overtime": len(seq) > mr,
            "win": 0 if a > b else 1,
            "r1w": wt[1], "r2w": wt[p2],
            "wr": [wt.get(k, -1) for k in range(1, 14)],
            "pc1": pc[1], "pc2": pc[p2],
            "map": h["map_name"],
            "r2_same": int(rw[1] == rw[2]) if 2 in rw else -1,
        })
    funnel.append({"step": "team identity: derived A/B totals equal header (t_starters/ct_starters)",
                   "remaining": len(recs) + len(pistol_missing)})
    funnel.append({"step": f"pistol rounds 1 and {p2} present", "remaining": len(recs)})
    return recs, funnel, {"team_identity_excluded": len(team_bad),
                          "pistol_missing_excluded": len(pistol_missing)}


def to_arrays(recs):
    return {
        "r1w": np.array([r["r1w"] for r in recs]),
        "r2w": np.array([r["r2w"] for r in recs]),
        "win": np.array([r["win"] for r in recs]),
        "map": [r["map"] for r in recs],
        "pc1": np.array([r["pc1"] for r in recs]),
        "pc2": np.array([r["pc2"] for r in recs]),
        "overtime": np.array([r["overtime"] for r in recs]),
        "r2_same": np.array([r["r2_same"] for r in recs]),
        "wr": np.array([r["wr"] for r in recs]),
    }


def subset(d, mask):
    out = {}
    for k, v in d.items():
        if isinstance(v, list):
            out[k] = [x for x, keep in zip(v, mask) if keep]
        else:
            out[k] = v[mask]
    return out


def boot_counts(M, n_boot, seed):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, M, size=(n_boot, M))
    cnt = np.zeros((n_boot, M))
    for b in range(n_boot):
        cnt[b] = np.bincount(idx[b], minlength=M)
    return cnt


def stats_from_counts(c, W1, W2, S):
    one = c.sum(axis=1)
    cW1 = c @ W1
    cW2 = c @ W2
    cS = c @ S
    cSW = c @ (S * W1)
    cSL = c @ (S * (1 - W1))
    cNS = one - cS
    cNSW = c @ ((1 - S) * W1)
    p1 = cW1 / one
    p2 = cW2 / one
    sb = cS / one
    q_tt = cSW / cS
    q_tf = cNSW / cNS
    return {
        "N_matches": one,
        "P_R1_winner_wins_match": p1,
        "P_R13_winner_wins_match": p2,
        "P_same_team_won_R1_and_R13": sb,
        "P_won_both_team_wins_match": q_tt,
        "P_lost_both_team_wins_match": 1 - q_tt,
        "P_won_R1_only_team_wins_match": q_tf,
        "P_won_R13_only_team_wins_match": 1 - q_tf,
        "P_R13_win_given_won_R1": sb,
        "P_R13_win_given_lost_R1": 1 - sb,
        "diff_R13_win_given_R1_minus_not": 2 * sb - 1,
        "OR_team_R1_winner_vs_loser": (p1 / (1 - p1)) ** 2,
        "OR_match_R1_odds": p1 / (1 - p1),
        "OR_team_R13_winner_vs_loser": (p2 / (1 - p2)) ** 2,
        "OR_match_R13_odds": p2 / (1 - p2),
        "OR_team_both_vs_neither": (cSW / cSL) ** 2,
        "OR_match_both_vs_neither": cSW / cSL,
        "OR_team_R1_R13_link": (cS / cNS) ** 2,
        "OR_match_R1_R13_link": cS / cNS,
    }


def analyse(d, seed=SEED):
    """Point estimates and match-cluster bootstrap intervals for the pistol statistics."""
    W1 = (d["r1w"] == d["win"]).astype(float)
    W2 = (d["r2w"] == d["win"]).astype(float)
    S = (d["r1w"] == d["r2w"]).astype(float)
    M = len(W1)
    pt_c = np.ones((1, M))
    pt = stats_from_counts(pt_c, W1, W2, S)
    cnt = boot_counts(M, N_BOOT, seed)
    bt = stats_from_counts(cnt, W1, W2, S)
    out = {}
    for k in pt:
        lo, hi = np.percentile(bt[k], [2.5, 97.5])
        out[k] = {"point": float(pt[k][0]), "lo": float(lo), "hi": float(hi)}
    out["counts"] = {
        "matches": int(M),
        "won_both_matches": int(S.sum()),
        "won_R1_only_matches": int(M - S.sum()),
        "won_R13_only_matches": int(M - S.sum()),
        "lost_both_matches": int(S.sum()),
        "R1_winner_wins": int(W1.sum()),
        "R13_winner_wins": int(W2.sum()),
        "same_team_won_both_pistols": int(S.sum()),
    }
    return out


def r2_check(d, seed=SEED):
    """P(round 2 winner = round 1 winner), one indicator per match (no swap between rounds 1 and 2)."""
    ind = d["r2_same"].astype(float)
    M = len(ind)
    pt = float(ind.mean())
    cnt = boot_counts(M, N_BOOT, seed)
    bt = cnt @ ind / cnt.sum(axis=1)
    lo, hi = np.percentile(bt, [2.5, 97.5])
    return {"matches": int(M), "P_win_R2_given_won_R1": pt, "lo": float(lo), "hi": float(hi)}


def ridge_or(model, j):
    beta = model.w[j + 1] / model.sd[j]
    return float(np.exp(beta)), float(beta)


class RidgeLogit:
    """Copied from scripts/model/per_tick_eval.py (standardised L2 logistic regression, C = 1,
    intercept unpenalised, numpy Newton steps)."""

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


def control(d, n_boot=N_BOOT, seed=SEED):
    M = len(d["win"])
    cnt_map = collections.Counter(d["map"])
    grp = [m if cnt_map[m] >= MIN_MAP_MATCHES else "other_rare" for m in d["map"]]
    gcount = collections.Counter(grp)
    ref = gcount.most_common(1)[0][0]
    levels = sorted(lv for lv in gcount if lv != ref)
    D = np.column_stack([[1.0 if g == lv else 0.0 for g in grp] for lv in levels])

    def rowify(a, b):
        out = np.empty(2 * M)
        out[0::2] = a
        out[1::2] = b
        return out

    x1 = rowify((d["r1w"] == 0).astype(float), (d["r1w"] == 1).astype(float))
    x13 = rowify((d["r2w"] == 0).astype(float), (d["r2w"] == 1).astype(float))
    y = rowify((d["win"] == 0).astype(float), (d["win"] == 1).astype(float))
    Dr = np.repeat(D, 2, axis=0)
    specs = {
        "unadjusted_R1": (np.column_stack([x1]), [0]),
        "adjusted_map_R1": (np.column_stack([x1, Dr]), [0]),
        "adjusted_map_both_pistols": (np.column_stack([x1, x13, Dr]), [0, 1]),
    }
    out = {"reference_map": ref, "map_levels_dummied": levels,
           "map_counts": dict(gcount), "rows": int(2 * M), "matches": int(M)}
    for name, (X, js) in specs.items():
        m = RidgeLogit().fit(X, y)
        out[name] = {}
        for j, lab in zip(js, ["R1", "R13"]):
            orr, beta = ridge_or(m, j)
            out[name][f"OR_{lab}"] = {"point": orr, "beta": beta}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, M, size=(n_boot, M))
    boots = {name: [] for name in specs}
    off = np.array([0, 1])
    for b in range(n_boot):
        rws = (2 * idx[b][:, None] + off).ravel()
        for name, (X, js) in specs.items():
            m = RidgeLogit().fit(X[rws], y[rws])
            boots[name].append([ridge_or(m, j)[0] for j in js])
    for name, (X, js) in specs.items():
        arr = np.array(boots[name])
        for col, (j, lab) in enumerate(zip(js, ["R1", "R13"])):
            lo, hi = np.percentile(arr[:, col], [2.5, 97.5])
            out[name][f"OR_{lab}"]["lo"] = float(lo)
            out[name][f"OR_{lab}"]["hi"] = float(hi)
    return out


def mh_map_stratified(d, n_boot=N_BOOT, seed=SEED):
    """Mantel-Haenszel team-row OR for R1 winner vs loser, stratified by map (same pooling as the
    logit). Each stratum is a 2x2 team-row table with a = d = k_m and b = c = N_m - k_m, so
    OR_MH = sum(k_m^2 / N_m) / sum((N_m - k_m)^2 / N_m). Map terms in the team-row logit are
    identically zero, so this is the informative map-stratified check."""
    W1 = (d["r1w"] == d["win"]).astype(float)
    M = len(W1)
    cnt_map = collections.Counter(d["map"])
    grp = np.array([m if cnt_map[m] >= MIN_MAP_MATCHES else "other_rare" for m in d["map"]])
    levels = sorted(set(grp.tolist()))
    G = np.column_stack([(grp == lv).astype(float) for lv in levels])

    def or_mh(c):
        Nm = c @ G
        km = c @ (G * W1[:, None])
        return (km ** 2 / Nm).sum(axis=1) / ((Nm - km) ** 2 / Nm).sum(axis=1)

    pt = float(or_mh(np.ones((1, M)))[0])
    bt = or_mh(boot_counts(M, n_boot, seed))
    lo, hi = np.percentile(bt, [2.5, 97.5])
    per_map = {}
    for lv in levels:
        idx = grp == lv
        Nm = int(idx.sum())
        km = int(W1[idx].sum())
        per_map[lv] = {"matches": Nm, "R1_winner_wins": km,
                       "P_R1_winner_wins": km / Nm,
                       "OR_team_R1_within_map": (km / (Nm - km)) ** 2 if km != Nm - km else float("nan")}
    return {"point": pt, "lo": float(lo), "hi": float(hi), "levels": levels, "per_map": per_map}


def round_profile(d, seed=SEED):
    """P(team that won round k wins the match), k = 1..13, main population, match-cluster bootstrap.
    Rounds 2-12 are not pistols; a flat profile means the pistol rounds are not special."""
    wr = d["wr"]
    M = wr.shape[0]
    cnt = boot_counts(M, N_BOOT, seed)
    Hm = (wr == d["win"][:, None]).astype(float)  # M x 13: did the round-k winner win the match
    out = {}
    for k in range(1, 14):
        hit = Hm[:, k - 1]
        pt = float(hit.mean())
        bt = cnt @ hit / cnt.sum(axis=1)
        lo, hi = np.percentile(bt, [2.5, 97.5])
        out[str(k)] = {"point": pt, "lo": float(lo), "hi": float(hi), "matches": int(M)}
    # matched differences: round k minus the per-match average over rounds 2-12 (same matches)
    avg_2_12 = Hm[:, 1:12].mean(axis=1)
    for k in (1, 13):
        dd = Hm[:, k - 1] - avg_2_12
        pt = float(dd.mean())
        bt = cnt @ dd / cnt.sum(axis=1)
        lo, hi = np.percentile(bt, [2.5, 97.5])
        out[f"diff_R{k}_minus_avg_R2_to_R12"] = {"point": pt, "lo": float(lo), "hi": float(hi),
                                                  "matches": int(M)}
    return out


def null_benchmark(n_sim=2_000_000, seed=SEED):
    """Fair-coin null: 24 independent 50/50 rounds, team X wins the match with >= 13 round wins
    (equivalent to first to 13 with virtual completion); 12-12 is decided by a coin (OT stand-in).
    Returns P(team wins match | pattern of pistols for that team). Monte Carlo error about 0.0004."""
    rng = np.random.default_rng(seed)
    R = rng.integers(0, 2, size=(n_sim, 24), dtype=np.int8)
    total = R.sum(axis=1, dtype=np.int32)
    coin = rng.integers(0, 2, size=n_sim, dtype=np.int8)
    win = (total >= 13) | ((total == 12) & (coin == 1))
    r1 = R[:, 0] == 1
    r13 = R[:, 12] == 1
    def rate(mask):
        return float(win[mask].mean())
    return {
        "n_sim": n_sim,
        "P_win_given_won_R1": rate(r1),
        "P_win_given_won_R13": rate(r13),
        "P_win_given_won_both": rate(r1 & r13),
        "P_win_given_lost_both": rate(~r1 & ~r13),
        "P_win_given_won_R1_only": rate(r1 & ~r13),
        "P_win_given_won_R13_only": rate(~r1 & r13),
        "OR_team_R1_null": (rate(r1) / (1 - rate(r1))) ** 2,
        "OR_team_R13_null": (rate(r13) / (1 - rate(r13))) ** 2,
        "OR_team_both_vs_neither_null": (rate(r1 & r13) / (1 - rate(r1 & r13))) ** 2,
        "P_same_team_won_both_null": 0.5,
    }


def implied_p(orv, team_row):
    r = np.sqrt(orv) if team_row else orv
    return float(r / (1 + r))


def svg_chart(bars, n_matches):
    W, Hh = 800, 450
    x0, x1, y_top, y_bot = 84, 776, 96, 350

    def yv(v):
        return y_bot - (y_bot - y_top) * v

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {Hh}" width="{W}" height="{Hh}" '
        f'font-family="Helvetica, Arial, sans-serif">',
        f'<rect x="0" y="0" width="{W}" height="{Hh}" fill="#ffffff"/>',
        '<text x="40" y="34" font-size="18" font-weight="600" fill="#0f172a">'
        'Match win rate by pistol outcome</text>',
        f'<text x="40" y="55" font-size="12" fill="#475569">24-round 5v5 matches with a decided winner, '
        f'N = {n_matches:,} matches.</text>',
        '<text x="40" y="71" font-size="12" fill="#475569">Bars: share of teams in each pattern that won '
        'the match; 95% match-cluster bootstrap interval.</text>',
    ]
    for g in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = yv(g)
        if g == 0.5:
            parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="#475569" '
                         f'stroke-width="1" stroke-dasharray="4 4"/>')
        else:
            parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="#e2e8f0" '
                         f'stroke-width="1"/>')
        parts.append(f'<text x="{x0 - 10}" y="{y + 4:.1f}" font-size="11" text-anchor="end" '
                     f'fill="#475569">{int(round(g * 100))}%</text>')
    slot = (x1 - x0) / len(bars)
    bw = 110
    for i, (label, val, lo, hi, n) in enumerate(bars):
        cx = x0 + slot * (i + 0.5)
        parts.append(f'<rect x="{cx - bw / 2:.1f}" y="{yv(val):.1f}" width="{bw}" '
                     f'height="{yv(0) - yv(val):.1f}" fill="#475569"/>')
        parts.append(f'<line x1="{cx:.1f}" y1="{yv(lo):.1f}" x2="{cx:.1f}" y2="{yv(hi):.1f}" '
                     f'stroke="#0f172a" stroke-width="1.5"/>')
        for yy in (yv(lo), yv(hi)):
            parts.append(f'<line x1="{cx - 9:.1f}" y1="{yy:.1f}" x2="{cx + 9:.1f}" y2="{yy:.1f}" '
                         f'stroke="#0f172a" stroke-width="1.5"/>')
        parts.append(f'<text x="{cx:.1f}" y="{yv(hi) - 9:.1f}" font-size="13" font-weight="600" '
                     f'text-anchor="middle" fill="#0f172a">{100 * val:.1f}%</text>')
        parts.append(f'<text x="{cx:.1f}" y="{y_bot + 20}" font-size="13" font-weight="600" '
                     f'text-anchor="middle" fill="#0f172a">{label}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{y_bot + 38}" font-size="11" text-anchor="middle" '
                     f'fill="#475569">n = {n} matches</text>')
        parts.append(f'<text x="{cx:.1f}" y="{y_bot + 54}" font-size="11" text-anchor="middle" '
                     f'fill="#475569">95% CI {100 * lo:.1f} to {100 * hi:.1f}</text>')
    parts.append('<text x="40" y="428" font-size="11" fill="#475569">Association only, no causal reading. '
                 'Each match gives two team rows with complementary outcomes.</text>')
    parts.append('<text x="40" y="444" font-size="11" fill="#475569">Pistol 2 = round 13. Wingman, tied, '
                 'incomplete and duplicate-copy matches are excluded.</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main():
    t0 = time.time()
    H, seqs, re_rows, hdr_rows = load()
    print(f"header rows {hdr_rows}; round_end rows {re_rows}; matches with round_end {len(seqs)}")

    drop, n_groups = dedupe(seqs)
    keys_all = sorted(H)
    after_dedupe = [m for m in keys_all if m not in drop and m in seqs]
    n_copies = len(keys_all) - len(after_dedupe)
    print(f"duplicate groups {n_groups}, copies removed {n_copies}, remaining {len(after_dedupe)}")
    base_funnel = [
        {"step": "header rows (one per match key)", "remaining": hdr_rows},
        {"step": f"duplicate copies removed (identical round_end sequence, {n_groups} groups)",
         "remaining": len(after_dedupe)},
    ]

    main_recs, main_f, main_x = build(H, seqs, after_dedupe, 24, 13)
    wing_recs, wing_f, wing_x = build(H, seqs, after_dedupe, 16, 9)
    main_funnel = base_funnel + main_f
    print("== main funnel")
    for s in main_funnel:
        print(f"  {s['remaining']:>6}  {s['step']}")
    print(f"  team identity excluded (main): {main_x}")
    print(f"== wingman funnel (sensitivity)")
    for s in wing_f:
        print(f"  {s['remaining']:>6}  {s['step']}")
    print(f"  wingman team identity excluded: {wing_x}")

    d = to_arrays(main_recs)
    res = analyse(d)
    print("\n== MAIN (24-round 5v5, decided, complete, team-consistent)")
    print(f"N matches {res['counts']['matches']}")
    for k in ["P_R1_winner_wins_match", "OR_team_R1_winner_vs_loser", "OR_match_R1_odds",
              "P_R13_winner_wins_match", "OR_team_R13_winner_vs_loser", "OR_match_R13_odds",
              "P_won_both_team_wins_match", "P_won_R1_only_team_wins_match",
              "P_won_R13_only_team_wins_match", "P_lost_both_team_wins_match",
              "OR_team_both_vs_neither", "OR_match_both_vs_neither",
              "P_same_team_won_R1_and_R13", "OR_team_R1_R13_link", "OR_match_R1_R13_link"]:
        v = res[k]
        print(f"  {k:<32} {v['point']:.4f}  [{v['lo']:.4f}, {v['hi']:.4f}]")
    print(f"  counts {res['counts']}")

    r2 = r2_check(d)
    print(f"round-2 check P(win R2|won R1) {r2['P_win_R2_given_won_R1']:.4f} "
          f"[{r2['lo']:.4f}, {r2['hi']:.4f}] n={r2['matches']}")
    prof = round_profile(d)
    print("\n== ROUND PROFILE: P(team that won round k wins match), main population")
    for k in range(1, 14):
        v = prof[str(k)]
        print(f"  round {k:>2}  {v['point']:.4f}  [{v['lo']:.4f}, {v['hi']:.4f}]")
    for k in (1, 13):
        v = prof[f"diff_R{k}_minus_avg_R2_to_R12"]
        print(f"  diff round {k} minus avg rounds 2-12: {v['point']:+.4f}  [{v['lo']:+.4f}, {v['hi']:+.4f}]")
    nb = null_benchmark()
    print("\n== NULL BENCHMARK (fair independent rounds, first to 13 of 24 virtual rounds, coin on 12-12)")
    for k, v in nb.items():
        if k != "n_sim":
            print(f"  {k:<32} {v:.4f}")

    ctl = control(d)
    mh = mh_map_stratified(d)
    ctl["map_stratified_MH_team_row_R1"] = {k: mh[k] for k in ("point", "lo", "hi")}
    ctl["per_map_R1"] = mh["per_map"]
    print("\n== CONTROL (team-row logit, RidgeLogit C=1)")
    print(f"  map-stratified Mantel-Haenszel team-row OR R1 {mh['point']:.4f} [{mh['lo']:.4f}, {mh['hi']:.4f}]")
    for lv, v in mh["per_map"].items():
        print(f"    {lv:<12} n={v['matches']:>4} P(R1 winner wins)={v['P_R1_winner_wins']:.4f}")
    for name in ["unadjusted_R1", "adjusted_map_R1", "adjusted_map_both_pistols"]:
        for lab in ["OR_R1", "OR_R13"]:
            if lab in ctl[name]:
                v = ctl[name][lab]
                print(f"  {name:<28} {lab:<6} {v['point']:.4f} [{v['lo']:.4f}, {v['hi']:.4f}]")
    print(f"  reference map {ctl['reference_map']}; dummied {ctl['map_levels_dummied']}")
    exact = (res['counts']['R1_winner_wins'] / (res['counts']['matches'] - res['counts']['R1_winner_wins'])) ** 2
    print(f"  exact team-row 2x2 OR for R1 {exact:.4f} (unadjusted logit should match closely)")

    # sensitivities
    sens = {}
    sens["pistols_both_player_count_10"] = subset(d, (d["pc1"] == 10) & (d["pc2"] == 10))
    sens["regulation_only_no_overtime"] = subset(d, ~d["overtime"])
    wing_d = to_arrays(wing_recs)
    sens["wingman_16_round_pistol2_is_round9"] = wing_d
    all_d = {k: (v + wing_d[k] if isinstance(v, list) else np.concatenate([v, wing_d[k]]))
             for k, v in d.items()}
    sens["all_decided_main_plus_wingman"] = all_d
    sens_out = {}
    for name, dd in sens.items():
        rr = analyse(dd)
        sens_out[name] = {k: rr[k] for k in ["P_R1_winner_wins_match", "OR_team_R1_winner_vs_loser",
                                              "P_R13_winner_wins_match", "OR_team_R13_winner_vs_loser",
                                              "P_same_team_won_R1_and_R13", "P_won_both_team_wins_match",
                                              "OR_team_both_vs_neither"]}
        sens_out[name]["counts"] = rr["counts"]
        print(f"  sens {name:<40} N={rr['counts']['matches']:>5} P_R1={rr['P_R1_winner_wins_match']['point']:.4f} "
              f"P_R13={rr['P_R13_winner_wins_match']['point']:.4f} S={rr['P_same_team_won_R1_and_R13']['point']:.4f}")

    # all deduped decided matches with rounds 1 and 2 (any max_rounds): context only
    ctx_recs = []
    for m in after_dedupe:
        h = H[m]
        if h["t_starters_score_final"] == h["ct_starters_score_final"]:
            continue
        seq = seqs[m]
        if len(seq) < 2:
            continue
        ctx_recs.append({"r2_same": int(seq[0][2] == seq[1][2])})
    ctx_ind = np.array([r["r2_same"] for r in ctx_recs], dtype=float)
    ctx = {"matches": int(len(ctx_ind)), "P_win_R2_given_won_R1": float(ctx_ind.mean())}

    bars_spec = [
        ("Won both pistols", "P_won_both_team_wins_match", res["counts"]["won_both_matches"]),
        ("Won round 1 only", "P_won_R1_only_team_wins_match", res["counts"]["won_R1_only_matches"]),
        ("Won round 13 only", "P_won_R13_only_team_wins_match", res["counts"]["won_R13_only_matches"]),
        ("Lost both pistols", "P_lost_both_team_wins_match", res["counts"]["lost_both_matches"]),
    ]
    bars = [(lab, res[k]["point"], res[k]["lo"], res[k]["hi"], n) for lab, k, n in bars_spec]
    svg = svg_chart(bars, res["counts"]["matches"])
    OUT_SVG.write_text(svg, encoding="utf-8")

    orr = res["OR_team_R1_winner_vs_loser"]["point"]
    comparison = {
        "snippet_OR": SNIPPET_OR,
        "snippet_status": "unverified snippet (CS:GO Majors, 334 matches); not checked here",
        "implied_P_R1_if_snippet_is_team_row_OR": implied_p(SNIPPET_OR, True),
        "implied_P_R1_if_snippet_is_match_level_odds": implied_p(SNIPPET_OR, False),
        "our_P_R1": res["P_R1_winner_wins_match"]["point"],
        "our_P_R1_ci": [res["P_R1_winner_wins_match"]["lo"], res["P_R1_winner_wins_match"]["hi"]],
        "our_OR_team_R1": orr,
        "our_OR_team_R1_ci": [res["OR_team_R1_winner_vs_loser"]["lo"],
                              res["OR_team_R1_winner_vs_loser"]["hi"]],
        "our_OR_match_R1": res["OR_match_R1_odds"]["point"],
        "our_OR_match_R1_ci": [res["OR_match_R1_odds"]["lo"], res["OR_match_R1_odds"]["hi"]],
    }

    out = {
        "id": "MA",
        "task": "Pistol round to match outcome (association only)",
        "data": {"round_end": str(RE_PATH), "header": str(HDR_PATH)},
        "bootstrap": {"unit": "match", "resamples": N_BOOT, "seed": SEED, "interval": "percentile 95%"},
        "funnel_main": main_funnel,
        "funnel_wingman": base_funnel + wing_f,
        "duplicates": {"groups": n_groups, "copies_removed": n_copies},
        "team_identity_excluded_main": main_x["team_identity_excluded"],
        "team_identity_excluded_wingman": wing_x["team_identity_excluded"],
        "main": res,
        "round2_check_main_population": r2,
        "round_profile_main": prof,
        "null_benchmark_fair_rounds": nb,
        "round2_check_context_all_decided_deduped": ctx,
        "control_team_row_logit": ctl,
        "sensitivity": sens_out,
        "comparison_snippet_2_95": comparison,
        "bars": [{"label": lab, "rate": val, "lo": lo, "hi": hi, "n_matches": n}
                 for lab, val, lo, hi, n in bars],
        "assumptions": [
            "Team A starts on T in round 1; side swaps follow bakeoff swap_expected (halftime and every 6 rounds after regulation).",
            "Match winner = team with larger header final count (header t_starters/ct_starters = team A/B totals, verified against the derived totals).",
            "Duplicates = identical round_end sequences (round, tick, winner), lowest key kept.",
            "Pistol 2 = round max_rounds//2 + 1 (round 13 for 24-round matches, round 9 for 16-round wingman).",
            "Team-row odds ratio = square of the match-level odds (one winner row and one loser row per match, complementary outcomes).",
            "Association only; pistol winners may be stronger teams (confounding) and round 13 is itself one of the rounds that decide the match.",
        ],
        "not_checked": [
            "The 1 Hz sample (equipment/money) and the bake-off kept set (1,521 matches) were not used; this population is defined from round_end and header alone.",
            "Player-level roster changes and bot takeovers were not checked per match.",
            "The CS:GO Majors snippet (OR 2.95, 334 matches) was not located; only compared numerically.",
        ],
        "run_seconds": round(time.time() - t0, 1),
    }
    OUT_JSON.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"\nwrote {OUT_JSON}\nwrote {OUT_SVG}\nrun {out['run_seconds']}s")


if __name__ == "__main__":
    main()
