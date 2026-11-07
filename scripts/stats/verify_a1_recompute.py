"""Independent verifier for A1 (armor analysis).

Recomputes the three headline numbers in the A1 report from the raw 1 Hz
sample, using its own code (no reuse of A1_armor_stats.py):

  H1  Mb (equipment + armor-sum) vs Ma (equipment), in-sample dLL, and bootstrap CI
  H2  Mxb vs Mx (equipment with kevlar/helmet/defuser removed), dLL and CI;
      plus the OLS check that equipment_value_calc already prices armor items
  H3  Halftime reset: armor=0 share and helmet share at R12 vs R13, with
      paired match-cluster bootstrap contrasts

Inputs (read only):
  data/state_sample/players_1hz.parquet
  data/state_sample/rounds.parquet
Outputs: stdout and logs/stats/verify_a1.log
"""
import math
import os

import numpy as np
import polars as pl

ROOT = "G:/pureskill/csds-cs2"
P = f"{ROOT}/data/state_sample/players_1hz.parquet"
R = f"{ROOT}/data/state_sample/rounds.parquet"
LOG = f"{ROOT}/logs/stats/verify_a1.log"
B = 1000
SEED = 0

LINES = []


def out(*args):
    s = " ".join(str(a) for a in args)
    print(s, flush=True)
    LINES.append(s)


# This environment crashes (Windows fatal exception 0xC06D007F) on BLAS
# matrix-matrix products (e.g. (X*W).T @ X) and on numpy.linalg. Matrix-vector
# and elementwise sums run fine, so the helpers below use only those.
def matvec(X, b):
    return np.sum(X * b[None, :], axis=1)


def rmat(X, v):
    """X.T @ v without BLAS."""
    return np.sum(X * v[:, None], axis=0)


def gram(X, W):
    """X.T @ diag(W) @ X without BLAS."""
    k = X.shape[1]
    H = np.empty((k, k))
    for j in range(k):
        H[:, j] = np.sum(X * (W * X[:, j])[:, None], axis=0)
    return H


def solve(A, b):
    """Gaussian elimination with partial pivoting. Avoids numpy.linalg."""
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).copy()
    n = len(b)
    for i in range(n):
        piv = max(range(i, n), key=lambda r: abs(A[r, i]))
        if piv != i:
            A[[i, piv]] = A[[piv, i]]
            b[[i, piv]] = b[[piv, i]]
        for r in range(i + 1, n):
            f = A[r, i] / A[i, i]
            A[r, i:] -= f * A[i, i:]
            b[r] -= f * b[i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i]
    return x


def fit_logit(X, y, w, max_iter=100):
    """Weighted logistic regression by Newton / IRLS.

    Returns beta, mean log loss (nats per round, weighted), and the
    max absolute score at the solution (convergence check).
    """
    k = X.shape[1]
    beta = np.zeros(k)
    pbar = np.sum(w * y) / np.sum(w)
    beta[0] = math.log(pbar / (1 - pbar))
    for _ in range(max_iter):
        eta = matvec(X, beta)
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -35, 35)))
        W = w * p * (1 - p)
        g = rmat(X, w * (y - p))
        H = gram(X, W)
        step = solve(H, g)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    eta = matvec(X, beta)
    p = 1.0 / (1.0 + np.exp(-np.clip(eta, -35, 35)))
    score = np.max(np.abs(rmat(X, w * (y - p))))
    # stable -[y log p + (1-y) log(1-p)]
    ll = np.logaddexp(0.0, eta) - y * eta
    return beta, float(np.sum(w * ll) / np.sum(w)), float(score)


def ols(X, yv):
    beta = solve(gram(X, np.ones(X.shape[0])), rmat(X, yv))
    res = yv - matvec(X, beta)
    r2 = 1.0 - np.sum(res ** 2) / np.sum((yv - yv.mean()) ** 2)
    return beta, float(r2)


def pct(arr):
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    try:
        run()
    finally:
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        with open(LOG, "w", encoding="utf-8") as f:
            f.write("\n".join(LINES) + "\n")


def run():
    out("== verify_a1: independent recomputation of A1 headline numbers ==")
    out("inputs:", P, "|", R)
    out("bootstrap: B =", B, "| seed =", SEED, "| resample matches, weight rounds by draw count")

    # self-test of the solver on a known system (solution [0.8, 1.4])
    st = solve([[2.0, 1.0], [1.0, 3.0]], [3.0, 5.0])
    out("solver self-test x =", np.round(st, 10).tolist(), "(expect [0.8, 1.4])")

    players = (pl.scan_parquet(P)
               .select(["match_key", "round", "tick", "player_id", "team_code", "armor",
                        "has_helmet", "has_defuser", "equipment_value_calc",
                        "seconds_since_live", "inv_primary", "inv_secondary",
                        "inv_flashbang", "inv_smokegrenade", "inv_hegrenade",
                        "inv_molotov", "inv_incgrenade", "inv_decoy", "inv_c4"])
               .collect()
               .with_columns(pl.col("round").cast(pl.Int64)))
    rounds = (pl.scan_parquet(R)
              .select(["match_key", "round", "ct_win", "ct_score", "t_score"])
              .collect()
              .with_columns(pl.col("round").cast(pl.Int64)))
    out("player rows:", players.height, "| round rows:", rounds.height)

    # ---- ct_win cross-check against score deltas on consecutive rounds
    rs = (rounds.sort(["match_key", "round"])
          .with_columns([
              pl.col("ct_score").shift(1).over("match_key").fill_null(0).alias("prev_ct"),
              pl.col("t_score").shift(1).over("match_key").fill_null(0).alias("prev_t"),
              pl.col("round").shift(1).over("match_key").fill_null(0).alias("prev_round"),
              pl.col("ct_score").shift(-1).over("match_key").alias("nxt_ct"),
              pl.col("t_score").shift(-1).over("match_key").alias("nxt_t"),
              pl.col("round").shift(-1).over("match_key").alias("nxt_round"),
          ]))
    cons = rs.filter(pl.col("round") == pl.col("prev_round") + 1).with_columns([
        (pl.col("ct_score") - pl.col("prev_ct")).alias("d_ct"),
        (pl.col("t_score") - pl.col("prev_t")).alias("d_t"),
    ])
    bad = cons.filter(~(((pl.col("ct_win") == 1) & (pl.col("d_ct") == 1) & (pl.col("d_t") == 0))
                        | ((pl.col("ct_win") == 0) & (pl.col("d_t") == 1) & (pl.col("d_ct") == 0))))
    out("ct_win vs score deltas: consecutive rounds =", cons.height,
        "| disagreements =", bad.height, "| ct_win null =", rounds["ct_win"].null_count())
    out("ct_win counts (all rounds):", rounds["ct_win"].value_counts().sort("ct_win").rows())
    # The score columns are pre-round scores. At the 12->13 halftime swap the
    # CT/T columns trade teams, so check R12 with the swapped relation:
    # CT won R12 -> next row ct_score == old t_score and next t_score == old ct_score + 1.
    nxt = rs.filter((pl.col("round") == 12) & (pl.col("nxt_round") == 13))
    sw_ok = nxt.filter(
        ((pl.col("ct_win") == 1) & (pl.col("nxt_t") == pl.col("ct_score") + 1)
         & (pl.col("nxt_ct") == pl.col("t_score")))
        | ((pl.col("ct_win") == 0) & (pl.col("nxt_ct") == pl.col("t_score") + 1)
           & (pl.col("nxt_t") == pl.col("ct_score"))))
    out("R12 ct_win vs swapped score relation (halftime):", sw_ok.height, "agree of", nxt.height)

    # ---- first freeze-end snapshot per (match_key, round) = minimum tick
    mins = players.group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("tick_min"))
    first = (players.join(mins, on=["match_key", "round"])
             .filter(pl.col("tick") == pl.col("tick_min")))
    dup = first.group_by(["match_key", "round", "player_id"]).len().filter(pl.col("len") > 1).height
    out("first-snapshot rows:", first.height, "| groups:", mins.height,
        "| duplicate (match,round,player):", dup,
        "| rows with seconds_since_live != 0:",
        first.filter(pl.col("seconds_since_live") != 0).height)

    first = first.with_columns([
        pl.col("armor").cast(pl.Int64).alias("arm_i"),
        (pl.col("armor") > 0).cast(pl.Int64).alias("armored"),
        pl.col("has_helmet").cast(pl.Int64).alias("helm"),
        pl.col("has_defuser").cast(pl.Int64).alias("defz"),
        pl.col("equipment_value_calc").cast(pl.Int64).alias("eq_i"),
    ]).with_columns(
        (pl.col("eq_i") - 650 * pl.col("armored") - 350 * pl.col("helm")
         - 400 * pl.col("defz")).alias("eqx_i"))

    out("armor values (first snapshot):", first["arm_i"].value_counts().sort("arm_i").rows())
    out("armor 1-50 (first snapshot rows):",
        first.filter(pl.col("arm_i").is_between(1, 50)).height, "of", first.height)
    out("helmet without armor (rows):",
        first.filter((pl.col("helm") == 1) & (pl.col("armored") == 0)).height)

    # ---- side aggregates per round
    stats = ["n", "eq", "arm", "narm", "nhelm", "eqx"]
    agg = first.group_by(["match_key", "round", "team_code"]).agg([
        pl.len().alias("n"),
        pl.col("eq_i").sum().alias("eq"),
        pl.col("arm_i").sum().alias("arm"),
        pl.col("armored").sum().alias("narm"),
        pl.col("helm").sum().alias("nhelm"),
        pl.col("eqx_i").sum().alias("eqx"),
    ])
    ct = agg.filter(pl.col("team_code") == 3).select(
        ["match_key", "round"] + [pl.col(c).alias(c + "_ct") for c in stats])
    tt = agg.filter(pl.col("team_code") == 2).select(
        ["match_key", "round"] + [pl.col(c).alias(c + "_t") for c in stats])
    rd = (rounds.join(ct, on=["match_key", "round"], how="left")
          .join(tt, on=["match_key", "round"], how="left"))
    fill = [c + "_ct" for c in stats] + [c + "_t" for c in stats]
    rd = rd.with_columns([pl.col(c).fill_null(0) for c in fill])
    rd = rd.with_columns(
        (pl.col("round").is_between(2, 12) | pl.col("round").is_between(14, 24)).alias("np"))

    np_rd = rd.filter(pl.col("np"))
    out("rounds by (n_ct, n_t) over all rounds:",
        rd.group_by(["n_ct", "n_t"]).len().sort(["n_ct", "n_t"]).rows())
    out("NP rounds (2-12, 14-24):", np_rd.height,
        "| NP rounds by (n_ct, n_t):", np_rd.group_by(["n_ct", "n_t"]).len().sort(["n_ct", "n_t"]).rows())
    dropped = np_rd.filter(~((pl.col("n_ct") == 5) & (pl.col("n_t") == 5)))
    out("NP rounds not 5v5 (dropped):", dropped.select(["match_key", "round", "n_ct", "n_t"]).rows())

    use = (np_rd.filter((pl.col("n_ct") == 5) & (pl.col("n_t") == 5))
           .sort(["match_key", "round"]))
    out("kept NP rounds:", use.height, "| kept player rows:",
        int(use["n_ct"].sum() + use["n_t"].sum()))
    pistol = rd.filter(pl.col("round").is_in([1, 13]))
    out("pistol rounds present (R1, R13):",
        pistol.group_by("round").len().sort("round").rows())
    out("ct_win mean on kept NP rounds:", round(float(use["ct_win"].mean()), 5))

    matches = sorted(rounds["match_key"].unique().to_list())
    M = len(matches)
    midx = {m: i for i, m in enumerate(matches)}
    out("matches in rounds file:", M)

    mid = np.array([midx[m] for m in use["match_key"].to_list()])
    y = use["ct_win"].cast(pl.Float64).to_numpy()
    eq = (use["eq_ct"] - use["eq_t"]).cast(pl.Float64).to_numpy() / 1000.0
    arm = (use["arm_ct"] - use["arm_t"]).cast(pl.Float64).to_numpy() / 100.0
    eqx = (use["eqx_ct"] - use["eqx_t"]).cast(pl.Float64).to_numpy() / 1000.0
    FEAT = {"eq": eq, "arm": arm, "eqx": eqx}
    SPECS = {
        "Ma": ["eq"],
        "Mb": ["eq", "arm"],
        "Mf": ["arm"],
        "Mx": ["eqx"],
        "Mxb": ["eqx", "arm"],
    }

    def X_of(names):
        return np.column_stack([np.ones(len(y))] + [FEAT[n] for n in names])

    Xs = {name: X_of(f) for name, f in SPECS.items()}

    # ---- full-sample fits (weight 1)
    ones = np.ones(len(y))
    full = {}
    for name, f in SPECS.items():
        beta, l, sc = fit_logit(Xs[name], y, ones)
        full[name] = (beta, l)
        out(f"[full] {name} features={f} LL={l:.5f} max|score|={sc:.1e} "
            f"beta={np.round(beta, 4).tolist()}")

    d_Mb = (full["Ma"][1] - full["Mb"][1]) * 1000.0
    d_Mxb = (full["Mx"][1] - full["Mxb"][1]) * 1000.0
    d_Mf = (full["Ma"][1] - full["Mf"][1]) * 1000.0
    out(f"[full] H1 dLL Mb vs Ma = {d_Mb:+.2f} milli-nats/round")
    out(f"[full] H2 dLL Mxb vs Mx = {d_Mxb:+.2f} milli-nats/round")
    out(f"[full] dLL Mf vs Ma = {d_Mf:+.2f} milli-nats/round")
    out(f"[full] Mb arm coef per 100 pts = {full['Mb'][0][2]:+.4f} ; "
        f"Mxb arm coef per 100 pts = {full['Mxb'][0][2]:+.4f}")

    # ---- halftime reset inputs: player rows in rounds 12 and 13
    pr = first.filter(pl.col("round").is_in([12, 13])).select(
        ["match_key", "round", "team_code", "arm_i", "helm"])
    p_mid = np.array([midx[m] for m in pr["match_key"].to_list()])
    p_round = pr["round"].to_numpy()
    p_team = pr["team_code"].to_numpy()
    p_arm0 = (pr["arm_i"].to_numpy() == 0).astype(float)
    p_helm = pr["helm"].to_numpy().astype(float)
    out("R12 players:", int((p_round == 12).sum()), "R13 players:", int((p_round == 13).sum()),
        "| R12 rounds:", int(pr.filter(pl.col("round") == 12).select(["match_key", "round"]).unique().height),
        "| R13 rounds:", int(pr.filter(pl.col("round") == 13).select(["match_key", "round"]).unique().height))

    def share(mask, val, w):
        den = np.sum(w[mask])
        return float(np.sum(w[mask] * val[mask]) / den) if den > 0 else float("nan")

    def h3_stats(w):
        r12 = p_round == 12
        r13 = p_round == 13
        ct_ = p_team == 3
        t_ = p_team == 2
        return {
            "a0_R12": share(r12, p_arm0, w),
            "a0_R13": share(r13, p_arm0, w),
            "h_R12": share(r12, p_helm, w),
            "h_R13": share(r13, p_helm, w),
            "a0_R12_CT": share(r12 & ct_, p_arm0, w),
            "a0_R12_T": share(r12 & t_, p_arm0, w),
            "a0_R13_CT": share(r13 & ct_, p_arm0, w),
            "a0_R13_T": share(r13 & t_, p_arm0, w),
        }

    pt = h3_stats(np.ones(len(p_round)))
    out("[full] H3 armor=0 share R12 = %.2f%%, R13 = %.2f%%, diff R13-R12 = %+.2f pp"
        % (100 * pt["a0_R12"], 100 * pt["a0_R13"], 100 * (pt["a0_R13"] - pt["a0_R12"])))
    out("[full] H3 helmet share R12 = %.2f%%, R13 = %.2f%%, diff = %+.2f pp"
        % (100 * pt["h_R12"], 100 * pt["h_R13"], 100 * (pt["h_R13"] - pt["h_R12"])))
    out("[full] H3 armor=0 by side: R12 CT %.2f%% T %.2f%% | R13 CT %.2f%% T %.2f%%"
        % (100 * pt["a0_R12_CT"], 100 * pt["a0_R12_T"], 100 * pt["a0_R13_CT"], 100 * pt["a0_R13_T"]))

    # ---- OLS check that equipment_value_calc already prices armor items
    def ols_block(df, label):
        Xo = np.column_stack([np.ones(df.height),
                              df["armored"].to_numpy().astype(float),
                              df["helm"].to_numpy().astype(float),
                              df["defz"].to_numpy().astype(float)])
        yo = df["eq_i"].to_numpy().astype(float)
        beta, r2 = ols(Xo, yo)
        out(f"[ols] equip ~ 1 + armored + helmet + defuser ({label}, n={df.height}): "
            f"const={beta[0]:.1f} armor>0={beta[1]:+.1f} helmet={beta[2]:+.1f} "
            f"defuser={beta[3]:+.1f} R2={r2:.4f}")
    ols_block(first, "all first-snapshot rows, armor items only")
    np_first = first.filter(pl.col("round").is_between(2, 12) | pl.col("round").is_between(14, 24))
    ols_block(np_first, "NP rows, armor items only")

    def ext_block(df, label):
        """Equipment on armor items plus weapon and grenade controls."""
        cols = [np.ones(df.height)]
        names = ["const"]
        for c in ["armored", "helm", "defz", "inv_flashbang", "inv_smokegrenade",
                  "inv_hegrenade", "inv_molotov", "inv_incgrenade", "inv_decoy", "inv_c4"]:
            cols.append(df[c].to_numpy().astype(float))
            names.append(c)
        for c, base in [("inv_primary", 0), ("inv_secondary", 0)]:
            vals = df[c].to_numpy()
            for code in sorted(set(vals.tolist())):
                if code == base:
                    continue
                cols.append((vals == code).astype(float))
                names.append(f"{c}={code}")
        X = np.column_stack(cols)
        yv = df["eq_i"].to_numpy().astype(float)
        beta, r2 = ols(X, yv)
        b = dict(zip(names, beta))
        out(f"[ols-ext] equip ~ armor items + weapon/grenade dummies ({label}, n={df.height}, "
            f"k={X.shape[1]}): armor>0={b['armored']:+.1f} helmet={b['helm']:+.1f} "
            f"defuser={b['defz']:+.1f} R2={r2:.4f}")
        return b, r2

    ext_block(first, "all first-snapshot rows")
    b_np, _ = ext_block(np_first, "NP rows")

    # ---- sensitivity: equipment-ex-items base with fitted (not list) armor/helmet/defuser prices
    fitted_p = np_first.with_columns(
        (pl.col("eq_i") - b_np["armored"] * pl.col("armored") - b_np["helm"] * pl.col("helm")
         - b_np["defz"] * pl.col("defz")).alias("eqf_i"))
    aggf = fitted_p.group_by(["match_key", "round", "team_code"]).agg(
        pl.col("eqf_i").sum().alias("eqf"))
    cf = aggf.filter(pl.col("team_code") == 3).select(
        ["match_key", "round", pl.col("eqf").alias("eqf_ct")])
    tf = aggf.filter(pl.col("team_code") == 2).select(
        ["match_key", "round", pl.col("eqf").alias("eqf_t")])
    use_f = (use.select(["match_key", "round"])
             .join(cf, on=["match_key", "round"], how="left")
             .join(tf, on=["match_key", "round"], how="left"))
    eqf = (use_f["eqf_ct"] - use_f["eqf_t"]).cast(pl.Float64).to_numpy() / 1000.0
    _, llf1, _ = fit_logit(np.column_stack([ones, eqf]), y, ones)
    _, llf2, _ = fit_logit(np.column_stack([ones, eqf, arm]), y, ones)
    out(f"[sens] fitted prices: armor>0={b_np['armored']:.1f} helmet={b_np['helm']:.1f} "
        f"defuser={b_np['defz']:.1f} (list 650/350/400)")
    out(f"[sens] equip-ex-items base LL={llf1:.5f} | +armor LL={llf2:.5f} | "
        f"dLL = {(llf1 - llf2) * 1000:+.2f} milli-nats (list-price version: {d_Mxb:+.2f})")

    # ---- bootstrap: resample matches with replacement, weight rounds by draw count
    rng = np.random.default_rng(SEED)
    keys = ["d_Mb", "d_Mxb", "d_Mf", "arm_Mb", "arm_Mxb", "a0_R12", "a0_R13", "h_R12", "h_R13",
            "d_a0", "d_h", "a0_R12_CT", "a0_R12_T", "a0_R13_CT", "a0_R13_T"]
    boot = {k: [] for k in keys}
    for _ in range(B):
        draws = np.bincount(rng.integers(0, M, size=M), minlength=M).astype(float)
        w = draws[mid]
        ll = {}
        bet = {}
        for name in SPECS:
            beta, l, _ = fit_logit(Xs[name], y, w)
            ll[name] = l
            bet[name] = beta
        boot["d_Mb"].append((ll["Ma"] - ll["Mb"]) * 1000.0)
        boot["d_Mxb"].append((ll["Mx"] - ll["Mxb"]) * 1000.0)
        boot["d_Mf"] = boot.get("d_Mf", []) + [(ll["Ma"] - ll["Mf"]) * 1000.0]
        boot["arm_Mb"].append(bet["Mb"][2])
        boot["arm_Mxb"].append(bet["Mxb"][2])
        wp = draws[p_mid]
        hs = h3_stats(wp)
        boot["a0_R12"].append(hs["a0_R12"])
        boot["a0_R13"] = boot.get("a0_R13", []) + [hs["a0_R13"]]
        boot["h_R12"].append(hs["h_R12"])
        boot["h_R13"] = boot.get("h_R13", []) + [hs["h_R13"]]
        boot["d_a0"] = boot.get("d_a0", []) + [hs["a0_R13"] - hs["a0_R12"]]
        boot["d_h"] = boot.get("d_h", []) + [hs["h_R13"] - hs["h_R12"]]
        boot["a0_R12_CT"].append(hs["a0_R12_CT"])
        boot["a0_R12_T"].append(hs["a0_R12_T"])
        boot["a0_R13_CT"].append(hs["a0_R13_CT"])
        boot["a0_R13_T"].append(hs["a0_R13_T"])

    out("")
    out("== bootstrap 95%% intervals (percentile, B=%d) ==" % B)
    for k in ["d_Mb", "d_Mxb", "d_Mf"]:
        lo, hi = pct(boot[k])
        out(f"{k}: [{lo:+.2f}, {hi:+.2f}]")
    for k in ["arm_Mb", "arm_Mxb"]:
        lo, hi = pct(boot[k])
        out(f"{k} (arm coef per 100 pts): [{lo:+.4f}, {hi:+.4f}]")
    for k in ["a0_R12", "a0_R13", "h_R12", "h_R13", "a0_R12_CT", "a0_R12_T", "a0_R13_CT", "a0_R13_T"]:
        lo, hi = pct(boot[k])
        out(f"{k}: [{100 * lo:.2f}%, {100 * hi:.2f}%]")
    for k in ["d_a0", "d_h"]:
        lo, hi = pct(boot[k])
        out(f"{k} (R13 minus R12, pp): {100 * float(np.mean(boot[k])):+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]")

    # ---- headline summary against the report
    lo1, hi1 = pct(boot["d_Mb"])
    lo2, hi2 = pct(boot["d_Mxb"])
    out("")
    out("== HEADLINE COMPARISON ==")
    out(f"H1 Mb vs Ma dLL: mine {d_Mb:+.2f} [{lo1:+.2f}, {hi1:+.2f}] | report +7.42 [+3.52, +12.94]")
    out(f"   LL Ma mine {full['Ma'][1]:.5f} report 0.63504 | LL Mb mine {full['Mb'][1]:.5f} report 0.62762")
    out(f"H2 Mxb vs Mx dLL: mine {d_Mxb:+.2f} [{lo2:+.2f}, {hi2:+.2f}] | report +11.60 [+6.48, +18.42]")
    out(f"   LL Mx mine {full['Mx'][1]:.5f} report 0.63920 | LL Mxb mine {full['Mxb'][1]:.5f} report 0.62759")
    a0_12, a0_13 = 100 * pt["a0_R12"], 100 * pt["a0_R13"]
    h_12, h_13 = 100 * pt["h_R12"], 100 * pt["h_R13"]
    out(f"H3 armor=0 share R12 {a0_12:.2f}% -> R13 {a0_13:.2f}% | report 7.0% -> 53.5%")
    out(f"   helmet share R12 {h_12:.2f}% -> R13 {h_13:.2f}% | report 85.8% -> 0%")
    out(f"   R13-R12 armor=0 {100 * (pt['a0_R13'] - pt['a0_R12']):+.2f} pp | report +46.4 [+42.6, +50.2]")
    out(f"   R13-R12 helmet {100 * (pt['h_R13'] - pt['h_R12']):+.2f} pp | report -85.8 [-87.6, -84.0]")


if __name__ == "__main__":
    main()
