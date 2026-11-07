"""Independent verifier for MA, step 3: headline statistics with a match-cluster bootstrap.

Team A = team that starts on T (round 1). A is on T in rounds 1-12 and on CT in rounds 13-24.
winner_team_code: 2 = T, 3 = CT. Match winner from header final counts (A = t_starters_score_final).
Bootstrap: resample matches with replacement, 1000 resamples, numpy default_rng(0), percentile 95% CI.
"""
import math

import numpy as np
import polars as pl

PATH = r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_MA_dedup.parquet"
B = 1000
SEED = 0
MAX_K = 13

d = pl.read_parquet(PATH)

# ---- Team identity check on all 24-round non-wingman decided matches (regulation and OT) ----
base = d.filter(
    (pl.col("max_rounds") == 24)
    & (~pl.col("is_wingman"))
    & (pl.col("t_starters_score_final") != pl.col("ct_starters_score_final"))
    & (pl.col("t_starters_score_final") + pl.col("ct_starters_score_final") == pl.col("n_round_end"))
)
print("decided 24-round non-wingman (dedup, not tied, sum==rounds):", base.height)

def a_won_regulation(winners):
    """A won round k (k=1..24) for regulation sides: A on T for k<=12, on CT for 13..24."""
    out = []
    for k, w in enumerate(winners, start=1):
        if k <= 12:
            out.append(w == 2)
        else:
            out.append(w == 3)
    return out

def a_won_ot(winners, phase):
    """OT rounds 25+: 3-round halves; phase 0 -> A on CT for rounds 25-27, phase 1 -> A on T."""
    out = []
    for k, w in enumerate(winners, start=1):
        if k <= 24:
            out.append(w == 2 if k <= 12 else w == 3)
        else:
            block = (k - 25) // 3
            a_side_is_t = ((block + phase) % 2) == 1
            out.append((w == 2) if a_side_is_t else (w == 3))
    return out

reg_ok = 0
reg_n = 0
ot_ok0 = 0
ot_ok1 = 0
ot_n = 0
for row in base.iter_rows(named=True):
    w = row["winners"]
    tf = row["t_starters_score_final"]
    cf = row["ct_starters_score_final"]
    if len(w) <= 24:
        reg_n += 1
        if sum(a_won_regulation(w)) == tf and (len(w) - sum(a_won_regulation(w))) == cf:
            reg_ok += 1
    else:
        ot_n += 1
        for phase, counter in ((0, "ot_ok0"), (1, "ot_ok1")):
            aw = a_won_ot(w, phase)
            if sum(aw) == tf and (len(w) - sum(aw)) == cf:
                if phase == 0:
                    ot_ok0 += 1
                else:
                    ot_ok1 += 1
print(f"team identity (regulation <=24 rounds): {reg_ok}/{reg_n} pass")
print(f"team identity (overtime): n={ot_n}, phase0 pass={ot_ok0}, phase1 pass={ot_ok1}")

# ---- Main set: 24-round 5v5 (non-wingman), both pistols (rounds 1 and 13) played ----
main = base.filter(pl.col("n_round_end") >= 13).sort("match_key")
N = main.height
print("MAIN N (24-round non-wingman decided, rounds 1 and 13 present):", N)
print("  of which overtime (>24 rounds):", main.filter(pl.col("n_round_end") > 24).height)
print("  of which regulation (<=24 rounds):", main.filter(pl.col("n_round_end") <= 24).height)
print("  main player_count(max) distribution:", main.group_by("player_count").len().sort("player_count").rows())
print("  matches ending before round 13 among base (24-round):", base.filter(pl.col("n_round_end") < 13).height)

wins = main["winners"].to_list()
maps = main["map_name"].to_list()
map_ids = {m: i for i, m in enumerate(sorted(set(maps)))}
mid = np.array([map_ids[m] for m in maps])
tf = main["t_starters_score_final"].to_numpy()
cf = main["ct_starters_score_final"].to_numpy()

# A-won indicator for rounds 1..13 (regulation mapping, rounds 1..13 are all regulation)
WA = np.zeros((N, MAX_K), dtype=bool)
for i, w in enumerate(wins):
    for k in range(1, MAX_K + 1):
        code = w[k - 1]
        WA[i, k - 1] = (code == 2) if k <= 12 else (code == 3)
MA_ = tf > cf  # A (T starter) wins match
# Check: A wins match from final counts should equal A total wins in regulation for regulation matches
print("  sanity: A-total from round winners equals t_final for all main matches:",
      int(sum(1 for i, w in enumerate(wins)
              if (sum(a_won_regulation(w)) if len(w) <= 24 else sum(a_won_ot(w, 0))) == tf[i]
              or (sum(a_won_regulation(w)) if len(w) <= 24 else sum(a_won_ot(w, 1))) == tf[i])), "of", N)

# Round-k winner wins match: indicator matrix
WIN_K = (WA == MA_[:, None])  # N x 13
R1_WIN = WIN_K[:, 0]
R13_WIN = WIN_K[:, 12]

def team_rows_or(x_won_pre, y_won_match):
    """2x2 team-row table: rows = team x (pre-event won) by y (match won). Each match gives 2 rows."""
    # Team A row
    xa = x_won_pre.astype(int)
    ya = y_won_match.astype(int)
    # Team B row (complement)
    xb = 1 - xa
    yb = 1 - ya
    x = np.concatenate([xa, xb])
    y = np.concatenate([ya, yb])
    n11 = int(np.sum((x == 1) & (y == 1)))
    n10 = int(np.sum((x == 1) & (y == 0)))
    n01 = int(np.sum((x == 0) & (y == 1)))
    n00 = int(np.sum((x == 0) & (y == 0)))
    return n11, n10, n01, n00

def or_from(n11, n10, n01, n00):
    if n10 == 0 or n01 == 0:
        return float("nan")
    return (n11 * n00) / (n10 * n01)

# Point estimates
p1 = float(np.mean(R1_WIN))
p13 = float(np.mean(R13_WIN))
print(f"\nP(R1 winner wins match) = {100*p1:.2f}%  (k={int(R1_WIN.sum())}/{N})")
print(f"P(R13 winner wins match) = {100*p13:.2f}%  (k={int(R13_WIN.sum())}/{N})")
print(f"match-level odds of R1 winner winning = {p1/(1-p1):.4f}  (sqrt of team-row OR)")
n11, n10, n01, n00 = team_rows_or(WA[:, 0], MA_)
print(f"team-row table R1: n11={n11} n10={n10} n01={n01} n00={n00}")
print(f"team-row OR R1 vs match = {or_from(n11, n10, n01, n00):.4f}")
n11, n10, n01, n00 = team_rows_or(WA[:, 12], MA_)
print(f"team-row table R13: n11={n11} n10={n10} n01={n01} n00={n00}")
print(f"team-row OR R13 vs match = {or_from(n11, n10, n01, n00):.4f}")

# Round profile
print("\nRound profile P(round-k winner wins match):")
prof = [float(np.mean(WIN_K[:, k])) for k in range(MAX_K)]
print(" ".join(f"R{k+1}={100*v:.1f}" for k, v in enumerate(prof)))
print(f"R1 - mean(R2..R12) = {100*(prof[0] - np.mean(prof[1:12])):.2f} pts")

# R2 winner = R1 winner (same-side rounds, use code equality)
r1r2 = np.mean([w[0] == w[1] for w in wins])
print(f"P(R2 winner == R1 winner) = {100*r1r2:.2f}%")

# "both / neither" team-level patterns (team rows)
both_mask = np.zeros(N, dtype=bool)
# Team A had R1 win and R13 win (x1 & x13) -> team rows
def pattern_stats(x1, x13, y):
    """Team rows: x1=won R1, x13=won R13, y=won match. Returns dict of counts and rates."""
    rows_x1 = np.concatenate([x1, ~x1]).astype(int)
    rows_x13 = np.concatenate([x13, ~x13]).astype(int)
    rows_y = np.concatenate([y, ~y]).astype(int)
    out = {}
    for name, a, b in (("both", 1, 1), ("R1only", 1, 0), ("R13only", 0, 1), ("neither", 0, 0)):
        sel = (rows_x1 == a) & (rows_x13 == b)
        out[name] = (int(sel.sum()), float(rows_y[sel].mean()) if sel.sum() else float("nan"))
    return out

ps = pattern_stats(WA[:, 0], WA[:, 12], MA_)
for k_, (n_, r_) in ps.items():
    print(f"  pattern {k_}: team rows n={n_}, P(win match)={100*r_:.2f}%")
print("  both-vs-neither team-row OR:", (ps["both"][0] * ps["neither"][0] * 0 + 1) and
      (ps["both"][1] / (1 - ps["both"][1])) / (ps["neither"][1] / (1 - ps["neither"][1])))
same_team = np.mean(WA[:, 0] == WA[:, 12])
print(f"P(same team won R1 and R13) = {100*same_team:.2f}%")
print(f"P(R13 winner wins | won R1) = {100*np.mean(R13_WIN[WA[:, 0]]):.2f}%  "
      f"P(R13 winner wins | lost R1) = {100*np.mean(R13_WIN[~WA[:, 0]]):.2f}%")

# ---- Bootstrap (match clusters) ----
rng = np.random.default_rng(SEED)
boot = {k: [] for k in ["p1", "odds1", "or1", "p13", "or13", "diff_r1_rest", "mh_map", "or_both_neither"]}
for _ in range(B):
    idx = rng.integers(0, N, N)
    w1 = R1_WIN[idx]
    w13 = R13_WIN[idx]
    s1 = w1.mean()
    s13 = w13.mean()
    boot["p1"].append(s1)
    boot["odds1"].append(s1 / (1 - s1))
    n11b = int(w1.sum()); n00b = n11b; n10b = len(w1) - n11b; n01b = n10b
    boot["or1"].append((n11b * n00b) / (n10b * n01b) if n10b * n01b else float("nan"))
    boot["p13"].append(s13)
    n11c = int(w13.sum()); n10c = len(w13) - n11c
    boot["or13"].append((n11c * n11c) / (n10c * n10c) if n10c else float("nan"))
    prof_b = WIN_K[idx].mean(axis=0)
    boot["diff_r1_rest"].append(prof_b[0] - prof_b[1:12].mean())
    # Mantel-Haenszel across maps on team-row tables (match-level counts per stratum)
    mh_num = 0.0
    mh_den = 0.0
    mids = mid[idx]
    wa_b = MA_[idx]
    ww1 = w1
    for s in np.unique(mids):
        sel = mids == s
        Ms = int(sel.sum())
        S = int(ww1[sel].sum())
        # Team-row table in stratum: n11=n00=S, n10=n01=Ms-S (rows = 2*Ms)
        n11s = S
        n00s = S
        n10s = Ms - S
        n01s = Ms - S
        mh_num += n11s * n00s / (2 * Ms)
        mh_den += n10s * n01s / (2 * Ms)
    boot["mh_map"].append(mh_num / mh_den if mh_den else float("nan"))
    # Both vs neither team-row OR on bootstrap sample
    pb = pattern_stats(WA[idx, 0], WA[idx, 12], MA_[idx])
    pb_both = pb["both"][1]
    pb_neither = pb["neither"][1]
    if pb_both < 1 and pb_neither > 0:
        boot["or_both_neither"].append((pb_both / (1 - pb_both)) / (pb_neither / (1 - pb_neither)))
    else:
        boot["or_both_neither"].append(float("nan"))

def ci(v):
    arr = np.array(v, dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

print("\nBootstrap (B=1000, seed 0) percentile 95% CIs:")
for k_ in ["p1", "odds1", "or1", "p13", "or13", "diff_r1_rest", "mh_map", "or_both_neither"]:
    lo, hi = ci(boot[k_])
    est_map = {
        "p1": p1, "odds1": p1 / (1 - p1), "or1": (p1 / (1 - p1)) ** 2 if False else None,
        "p13": p13, "or13": None, "diff_r1_rest": prof[0] - np.mean(prof[1:12]),
        "mh_map": None, "or_both_neither": None,
    }
    print(f"  {k_}: CI [{lo:.4f}, {hi:.4f}]")

# Point estimates for ORs (exact, from full sample)
n11, n10, n01, n00 = team_rows_or(WA[:, 0], MA_)
or1_point = or_from(n11, n10, n01, n00)
n11c, n10c, n01c, n00c = team_rows_or(WA[:, 12], MA_)
or13_point = or_from(n11c, n10c, n01c, n00c)
print(f"\nPoint: team-row OR R1 = {or1_point:.4f}, CI [{ci(boot['or1'])[0]:.4f}, {ci(boot['or1'])[1]:.4f}]")
print(f"Point: team-row OR R13 = {or13_point:.4f}, CI [{ci(boot['or13'])[0]:.4f}, {ci(boot['or13'])[1]:.4f}]")
print(f"Point: match-level odds R1 = {p1/(1-p1):.4f}, CI [{ci(boot['odds1'])[0]:.4f}, {ci(boot['odds1'])[1]:.4f}]")
print(f"Point: match-level odds R13 = {p13/(1-p13):.4f}")
print(f"Point: MH (map-stratified) OR R1 = ", end="")
mh_num = 0.0
mh_den = 0.0
for s in np.unique(mid):
    sel = mid == s
    Ms = int(sel.sum())
    S = int(R1_WIN[sel].sum())
    mh_num += S * S / (2 * Ms)
    mh_den += (Ms - S) * (Ms - S) / (2 * Ms)
print(f"{mh_num/mh_den:.4f}  CI [{ci(boot['mh_map'])[0]:.4f}, {ci(boot['mh_map'])[1]:.4f}]")
print(f"Point: P(R1 winner wins) CI [{100*ci(boot['p1'])[0]:.2f}, {100*ci(boot['p1'])[1]:.2f}]")
print(f"Point: round1 minus mean(R2..R12) = {100*(prof[0]-np.mean(prof[1:12])):.2f} pts, "
      f"CI [{100*ci(boot['diff_r1_rest'])[0]:.2f}, {100*ci(boot['diff_r1_rest'])[1]:.2f}]")

# Overtime share
print(f"\nOvertime matches in main: {main.filter(pl.col('n_round_end') > 24).height} "
      f"({100*main.filter(pl.col('n_round_end') > 24).height/N:.1f}%)")

# Save bootstrap values for reproducibility of CIs
import json
out = {
    "N": N,
    "p1": p1,
    "p13": p13,
    "or1_point": or1_point,
    "or13_point": or13_point,
    "round_profile": prof,
    "ci": {k_: ci(boot[k_]) for k_ in boot},
}
out_path = r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_MA_stats.json"
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1, default=float)
print("wrote", out_path)
