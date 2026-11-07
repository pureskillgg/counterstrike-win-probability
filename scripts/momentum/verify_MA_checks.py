"""Independent verifier for MA, step 4: fair-coin benchmark, map-adjusted logit, sensitivity sets, funnel extras."""
import numpy as np
import polars as pl

DEDUP = r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_MA_dedup.parquet"

d = pl.read_parquet(DEDUP)

# ---- 0. Funnel extras ----
tied24 = d.filter((pl.col("max_rounds") == 24) & (~pl.col("is_wingman"))
                  & (pl.col("t_starters_score_final") == pl.col("ct_starters_score_final")))
print("24-round non-wingman ties (dedup):", tied24.height)
print("24-round non-wingman sum!=rounds (dedup, not tied):",
      d.filter((pl.col("max_rounds") == 24) & (~pl.col("is_wingman"))
               & (pl.col("t_starters_score_final") != pl.col("ct_starters_score_final"))
               & (pl.col("t_starters_score_final") + pl.col("ct_starters_score_final") != pl.col("n_round_end"))).height)
print("all dedup ties:", d.filter(pl.col("t_starters_score_final") == pl.col("ct_starters_score_final")).height)
print("all dedup not-tied:", d.filter(pl.col("t_starters_score_final") != pl.col("ct_starters_score_final")).height)

# Near-duplicate groups (same map, date, final scores) inside the 24-round main population
near = d.group_by(["map_name", "match_date", "t_starters_score_final", "ct_starters_score_final"]).agg(
    pl.len().alias("n"), pl.col("max_rounds").alias("mr"), pl.col("n_round_end").alias("nr"))
near = near.filter(pl.col("n") > 1)
print("near-duplicate groups (map,date,score):", near.height)
print(near.head(20))

# ---- 1. Main set ----
main = d.filter(
    (pl.col("max_rounds") == 24) & (~pl.col("is_wingman"))
    & (pl.col("t_starters_score_final") != pl.col("ct_starters_score_final"))
    & (pl.col("t_starters_score_final") + pl.col("ct_starters_score_final") == pl.col("n_round_end"))
    & (pl.col("n_round_end") >= 13)
).sort("match_key")
N = main.height
print("\nMAIN N =", N)


def build(df):
    w = df["winners"].to_list()
    n = len(w)
    WA = np.zeros((n, 13), dtype=bool)
    for i, ws in enumerate(w):
        for k in range(1, 14):
            WA[i, k - 1] = (ws[k - 1] == 2) if k <= 12 else (ws[k - 1] == 3)
    MA_ = df["t_starters_score_final"].to_numpy() > df["ct_starters_score_final"].to_numpy()
    return WA, MA_, df["map_name"].to_list(), w


WA, MA_, maps, W = build(main)
R1W = (WA[:, 0] == MA_)
R13W = (WA[:, 12] == MA_)


def or_team_rows(x, y):
    """team-row 2x2 from match arrays: x = pistol-winner team indicator for team A, y = A wins match."""
    xa = x.astype(int)
    ya = y.astype(int)
    X = np.concatenate([xa, 1 - xa])
    Y = np.concatenate([ya, 1 - ya])
    n11 = int(((X == 1) & (Y == 1)).sum())
    n10 = int(((X == 1) & (Y == 0)).sum())
    n01 = int(((X == 0) & (Y == 1)).sum())
    n00 = int(((X == 0) & (Y == 0)).sum())
    return n11, n10, n01, n00


# ---- 2. Fair-coin benchmark, simulation with 2,000,000 matches ----
rng = np.random.default_rng(12345)
S = 2_000_000
flips = rng.integers(0, 2, size=(S, 24), dtype=np.int8)  # 1 = team P wins the round
tP = flips.sum(axis=1)
coin = rng.integers(0, 2, size=S, dtype=np.int8)
P_wins = np.where(tP > 12, 1, np.where(tP < 12, 0, coin)).astype(bool)  # team P wins match
# Team P is the team that won round 1 when flips[:,0]==1; define "R1 winner" as the team that won round 1
r1_is_P = flips[:, 0] == 1
r1_winner_wins = (r1_is_P == P_wins)
r13_is_P = flips[:, 12] == 1
r13_winner_wins = (r13_is_P == P_wins)
print("\nFair-coin benchmark (S=2,000,000, coin for 12-12):")
print(f"  P(R1 winner wins match) = {100*r1_winner_wins.mean():.2f}%")
print(f"  P(R13 winner wins match) = {100*r13_winner_wins.mean():.2f}%")
# Team-row patterns (team P and team Q rows)
xr1 = np.concatenate([r1_is_P, ~r1_is_P])
xr13 = np.concatenate([r13_is_P, ~r13_is_P])
yy = np.concatenate([P_wins, ~P_wins])
for name, a, b in (("both", 1, 1), ("R1only", 1, 0), ("R13only", 0, 1), ("neither", 0, 0)):
    sel = (xr1 == a) & (xr13 == b)
    print(f"  pattern {name}: P(win match) = {100*yy[sel].mean():.2f}%")
p_null = r1_winner_wins.mean()
print(f"  null risk ratio = {p_null/(1-p_null):.4f}, null team-row OR = {(p_null/(1-p_null))**2:.4f}")
print("  analytic check: 0.5 + 0.5*C(23,11)/2^23 =",
      0.5 + 0.5 * (np.math.comb(23, 11) / 2**23) if hasattr(np, "math") else
      0.5 + 0.5 * (__import__("math").comb(23, 11) / 2**23))

# ---- 3. Observed headline ratios on main set ----
p1 = R1W.mean()
print(f"\nObserved: P(R1 winner wins)={100*p1:.2f}%, P(win|lost R1)={100*(1-p1):.2f}%")
print(f"  risk ratio (won R1 vs lost R1) = {p1/(1-p1):.4f}")
n11, n10, n01, n00 = or_team_rows(WA[:, 0], MA_)
print(f"  team-row OR = {(n11*n00)/(n10*n01):.4f}  (n11={n11}, n10={n10}, n01={n01}, n00={n00})")

# ---- 4. Map-adjusted team-row logit (unpenalised Newton) ----
maps_u = sorted(set(maps))
mi = {m: i for i, m in enumerate(maps_u)}
N_m = len(maps_u)
# Team rows
X_r1 = np.concatenate([WA[:, 0], ~WA[:, 0]]).astype(float)
Y = np.concatenate([MA_, ~MA_]).astype(float)
mapidx = np.array([mi[m] for m in maps] * 2)
M = np.zeros((2 * N, N_m))
M[np.arange(2 * N), mapidx] = 1.0


def newton_logit(Xmat, y, iters=50):
    beta = np.zeros(Xmat.shape[1])
    for _ in range(iters):
        eta = Xmat @ beta
        pr = 1 / (1 + np.exp(-eta))
        g = Xmat.T @ (y - pr)
        H = Xmat.T @ (Xmat * (pr * (1 - pr))[:, None]) + 1e-10 * np.eye(Xmat.shape[1])
        step = np.linalg.solve(H, g)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return beta

b_un = newton_logit(np.column_stack([X_r1, np.ones(2 * N)]), Y)
print(f"\nTeam-row logit, R1 only + intercept: OR = {np.exp(b_un[0]):.4f}")
Xadj = np.column_stack([X_r1, M])
b_adj = newton_logit(Xadj, Y)
print(f"Team-row logit, R1 + map one-hot (no separate intercept): OR = {np.exp(b_adj[0]):.4f}")
print(f"  max |map coefficient| = {np.max(np.abs(b_adj[1:])):.2e} (map terms near zero?)")

# ---- 5. Sensitivity sets (point estimates only) ----
def sens_or(df_sub, label):
    WA2, MA2, _, _ = build(df_sub)
    p = (WA2[:, 0] == MA2).mean()
    n11_, n10_, n01_, n00_ = or_team_rows(WA2[:, 0], MA2)
    print(f"  {label}: N={df_sub.height}, P(R1 winner wins)={100*p:.2f}%, team-row OR={(n11_*n00_)/(n10_*n01_):.3f}")

print("\nSensitivity (point estimates):")
reg = main.filter(pl.col("n_round_end") <= 24)
sens_or(reg, "regulation only (<=24 rounds)")
pc10 = main.filter((pl.col("player_count").max().over("match_key") == 10) if False else pl.col("player_count") == 10)
print("  main rows with round_end player_count==10 for round-1 row (unique matches):", main.height)
p10 = d.filter(pl.col("max_rounds") == 24).height
# Alternative definition: both pistol rounds have player_count == 10
# (round-level player_count is in the matches parquet only as max/min; use min/max over rounds)
sens_or(main.filter((pl.col("player_count_min") == 10) & (pl.col("player_count") == 10)), "all rounds player_count==10")
sens_or(main.filter(pl.col("player_count") == 10), "max player_count==10")

# Wingman: pistols are round 1 and round 9 (MR8, 16 rounds), sides swap after round 8
wm = d.filter((pl.col("is_wingman")) & (pl.col("t_starters_score_final") != pl.col("ct_starters_score_final"))
              & (pl.col("t_starters_score_final") + pl.col("ct_starters_score_final") == pl.col("n_round_end"))
              & (pl.col("n_round_end") >= 9)).sort("match_key")
wid_ok = 0
wW = []
wA = []
for row in wm.iter_rows(named=True):
    ws = row["winners"]
    tf_, cf_ = row["t_starters_score_final"], row["ct_starters_score_final"]
    aw = [((ws[k - 1] == 2) if k <= 8 else (ws[k - 1] == 3)) for k in range(1, len(ws) + 1)]
    ok = (sum(aw) == tf_) and (len(ws) - sum(aw) == cf_)
    wid_ok += int(ok)
    wW.append((ws, bool(ok)))
print(f"\nWingman candidates (decided, sum==rounds, >=9 rounds): {wm.height}; team-identity pass: {wid_ok}")
wm_ok = wm.filter(pl.Series([ok for _, ok in wW]))
if wm_ok.height:
    r1_wins = []
    r9_wins = []
    mA = []
    for row in wm_ok.iter_rows(named=True):
        ws = row["winners"]
        tf_, cf_ = row["t_starters_score_final"], row["ct_starters_score_final"]
        a1 = ws[0] == 2
        a9 = ws[8] == 3  # round 9: A on CT
        mA.append(tf_ > cf_)
        r1_wins.append(a1)
        r9_wins.append(a9)
    r1_wins = np.array(r1_wins)
    r9_wins = np.array(r9_wins)
    mA = np.array(mA)
    p9 = np.mean(r9_wins == mA)
    n11_, n10_, n01_, n00_ = or_team_rows(r9_wins, mA)
    print(f"  wingman pistol round 9: P(winner wins)={100*p9:.2f}%, team-row OR={(n11_*n00_)/(n10_*n01_) if n10_*n01_ else float('nan'):.3f}, N={len(mA)}")
    print(f"  wingman round 1: P(winner wins)={100*np.mean(r1_wins == mA):.2f}%")

# ---- 6. R2 = R1 winner, population variants ----
def r2_eq(df):
    return float(np.mean([w[0] == w[1] for w in df["winners"].to_list() if len(w) >= 2]))

print("\nR2 winner == R1 winner variants:")
print(f"  main (N={N}): {100*r2_eq(main):.2f}%")
print(f"  all dedup not-tied (N={d.filter(pl.col('t_starters_score_final') != pl.col('ct_starters_score_final')).height}):",
      f"{100*r2_eq(d.filter(pl.col('t_starters_score_final') != pl.col('ct_starters_score_final'))):.2f}%")
print(f"  all dedup (N={d.height}): {100*r2_eq(d):.2f}%")
print(f"  all dedup decided and sum==rounds (N={d.filter((pl.col('t_starters_score_final') != pl.col('ct_starters_score_final')) & (pl.col('t_starters_score_final') + pl.col('ct_starters_score_final') == pl.col('n_round_end'))).height}):",
      f"{100*r2_eq(d.filter((pl.col('t_starters_score_final') != pl.col('ct_starters_score_final')) & (pl.col('t_starters_score_final') + pl.col('ct_starters_score_final') == pl.col('n_round_end')))):.2f}%")
