"""M2 analysis: CT win rate by map per era, and whether the map effect survives the alive
difference and the live-start score lead.

Inputs (scratchpad, written by scripts/swarm2/M2_build.py):
  M2_rounds_<era>.parquet, M2_states_<era>.parquet

Sections
  A. Per-map CT win rate on the bake-off kept set, per era. Cluster bootstrap over matches
     (2000 resamples, seed 0). Per-map CI resamples matches within the map. Contrast
     (map minus all other maps in the era) resamples all matches in the era.
  B. Heterogeneity across maps on rounds (no controls): LR test, cluster-robust Wald.
  C. Logistic regression on states (the model's unit). Controls: ct_alive_minus_t and the
     live-start score lead (ct_score_minus_t_score). Compared with and without map dummies.
     Also a flexible control variant (alive difference and lead as categories).
  D. Out-of-sample: 2026 primary split (bake-off make_splits, seed 0). Fit on train, score
     test, with a match-cluster bootstrap CI of the log-loss difference.
  E. Cross-era transfer: fit on all 2026 kept states, score 2024 kept states. Also excludes
     2024 round 13 (the open side-label defect).
  F. Round-13 defect: 2024 bomb-outcome mismatch by map; per-map 2024 rates with and without
     round 13; 2026 mismatch count (expected 0).
  G. Sensitivity: 2026 per-map rates for 24-round matches only; raw (unfiltered) rates per era.

Assumptions (all marked ASSUMPTION in the report)
  - B = 2000 bootstrap resamples, seed 0 (numpy default_rng(0) for bootstrap; Python random
    seed 0 for the primary split, same as the bake-off).
  - Map level rule: a map has its own level in an era if it has >= 20 kept matches there;
    other maps are pooled into "other". Reference map: de_mirage (the largest map in both eras).
  - Score lead is the freeze-end score difference (CT minus T). Alive difference is the state
    alive count difference (CT minus T). Both are model features, not outcomes.
  - Cluster-robust SEs: CR1 small-sample factor, clusters = match.

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\M2_analysis.py
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.special import expit
from scipy.stats import chi2 as chi2_dist

sys.stdout.reconfigure(encoding="utf-8")

SCRATCH = Path("C:/Users/13144/AppData/Local/Temp/claude/C--Users-13144-Documents-pureskill/"
               "e0a18ad2-aa9e-4a3c-9946-29641b2cde82/scratchpad")
B = 2000
SEED = 0
REF_MAP = "de_mirage"
MIN_MATCHES = 20
ERAS = ["2026", "2024"]
DEFECT_ROUND = 13
T, CT = 2, 3
RESULTS = {}


def hr(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def load(era):
    r = pl.read_parquet(SCRATCH / f"M2_rounds_{era}.parquet")
    s = pl.read_parquet(SCRATCH / f"M2_states_{era}.parquet")
    return r, s


def kept_rounds(r):
    return r.filter(pl.col("kept"))


def match_counts(rounds):
    """Per match and map: rounds (n) and CT wins (c)."""
    return (rounds.group_by(["match_key", "map_name"])
            .agg(pl.len().alias("n"), pl.col("label_ct_win").sum().alias("c"))
            .sort(["match_key"]))


def pct_ci(x):
    return np.nanpercentile(x, [2.5, 97.5])


def boot_map_rate(n, c, B, seed):
    """Within-map cluster bootstrap of the CT win rate. n, c are per-match round and CT-win counts."""
    rng = np.random.default_rng(seed)
    M = len(n)
    idx = rng.integers(0, M, size=(B, M))
    rate = c[idx].sum(1) / n[idx].sum(1)
    return pct_ci(rate)


def boot_contrast(n_all, c_all, g, B, seed):
    """Era-wide cluster bootstrap of (map rate - rest-of-era rate). g = 1 for matches on the map."""
    rng = np.random.default_rng(seed)
    M = len(n_all)
    W = rng.multinomial(M, np.full(M, 1.0 / M), size=B).astype(np.float64)
    num_m = W @ (c_all * g)
    den_m = W @ (n_all * g)
    num_r = W @ (c_all * (1 - g))
    den_r = W @ (n_all * (1 - g))
    diff = num_m / den_m - num_r / den_r
    return pct_ci(diff)


def boot_overall(n_all, c_all, B, seed):
    rng = np.random.default_rng(seed)
    M = len(n_all)
    W = rng.multinomial(M, np.full(M, 1.0 / M), size=B).astype(np.float64)
    return pct_ci((W @ c_all) / (W @ n_all))


# --------------------------------------------------------------------------- A
def section_a(rounds_by_era):
    hr("A. CT win rate by map (kept set), cluster bootstrap, 2000 resamples, seed 0")
    out = {}
    for era in ERAS:
        k = kept_rounds(rounds_by_era[era])
        mc = match_counts(k)
        n_all = mc.group_by("match_key").agg(pl.col("n").sum(), pl.col("c").sum()).sort("match_key")
        n_all_arr = n_all["n"].to_numpy().astype(np.float64)
        c_all_arr = n_all["c"].to_numpy().astype(np.float64)
        tot_r = int(n_all_arr.sum())
        tot_c = int(c_all_arr.sum())
        ov_lo, ov_hi = boot_overall(n_all_arr, c_all_arr, B, SEED)
        print(f"\n[{era}] kept matches {n_all.height}, rounds {tot_r}, CT win rate "
              f"{tot_c / tot_r:.4f} (95% CI {ov_lo:.4f}-{ov_hi:.4f})")
        maps = (mc.group_by("map_name").agg(pl.col("match_key").n_unique().alias("matches"),
                                             pl.col("n").sum().alias("rounds"),
                                             pl.col("c").sum().alias("ct_wins"))
                .sort("rounds", descending=True))
        rows = []
        for m in maps.iter_rows(named=True):
            mm = mc.filter(pl.col("map_name") == m["map_name"]).sort("match_key")
            n_m = mm["n"].to_numpy().astype(np.float64)
            c_m = mm["c"].to_numpy().astype(np.float64)
            rate = m["ct_wins"] / m["rounds"]
            if len(n_m) >= 2:
                lo, hi = boot_map_rate(n_m, c_m, B, SEED)
            else:
                lo, hi = float("nan"), float("nan")
            mk_set = set(mm["match_key"].to_list())
            g = np.array([1.0 if x in mk_set else 0.0 for x in n_all["match_key"].to_list()])
            rest_rate = (tot_c - m["ct_wins"]) / (tot_r - m["rounds"])
            if 0 < g.sum() < len(g):
                clo, chi = boot_contrast(n_all_arr, c_all_arr, g, B, SEED)
            else:
                clo, chi = float("nan"), float("nan")
            rows.append({"map": m["map_name"], "matches": m["matches"], "rounds": m["rounds"],
                         "ct_wins": m["ct_wins"], "ct_rate": rate, "ci_lo": lo, "ci_hi": hi,
                         "rest_rate": rest_rate, "diff_vs_rest": rate - rest_rate,
                         "diff_ci_lo": clo, "diff_ci_hi": chi,
                         "small_map": m["matches"] < MIN_MATCHES})
        print(f"{'map':<14}{'matches':>8}{'rounds':>8}{'CT rate':>9}{'95% CI':>20}"
              f"{'rest':>8}{'diff':>8}{'diff 95% CI':>22}")
        for r in rows:
            flag = " *" if r["small_map"] else ""
            ci_s = "{:.4f}-{:.4f}".format(r["ci_lo"], r["ci_hi"])
            dci_s = "{:+.4f}..{:+.4f}".format(r["diff_ci_lo"], r["diff_ci_hi"])
            print(f"{r['map']:<14}{r['matches']:>8}{r['rounds']:>8}{r['ct_rate']:>9.4f}"
                  f"{ci_s:>20}{r['rest_rate']:>8.4f}"
                  f"{r['diff_vs_rest']:>+8.4f}{dci_s:>22}{flag}")
        print("  * fewer than 20 kept matches: CI shown for completeness, not decision-grade")
        out[era] = {"overall": {"matches": n_all.height, "rounds": tot_r, "ct_rate": tot_c / tot_r,
                                "ci": [ov_lo, ov_hi]}, "maps": rows}
    RESULTS["A_map_rates"] = out
    return out


# --------------------------------------------------------------------------- model helpers
def fit_logit(X, y, clusters):
    n, k = X.shape
    beta = np.zeros(k)
    p0 = float(y.mean())
    beta[0] = np.log(p0 / (1 - p0))
    for _ in range(100):
        p = expit(X @ beta)
        w = p * (1 - p)
        g = X.T @ (y - p)
        H = (X * w[:, None]).T @ X + 1e-9 * np.eye(k)
        step = np.linalg.solve(H, g)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = expit(X @ beta)
    w = p * (1 - p)
    H = (X * w[:, None]).T @ X + 1e-9 * np.eye(k)
    Hinv = np.linalg.inv(H)
    eps = 1e-15
    ll = float(np.sum(y * np.log(np.clip(p, eps, 1)) + (1 - y) * np.log(np.clip(1 - p, eps, 1))))
    s = X * (y - p)[:, None]
    _, inv = np.unique(clusters, return_inverse=True)
    C = int(inv.max()) + 1
    S = np.column_stack([np.bincount(inv, weights=s[:, j], minlength=C) for j in range(k)])
    V = Hinv @ (S.T @ S) @ Hinv * (C / (C - 1)) * ((n - 1) / (n - k))
    return {"beta": beta, "V": V, "ll": ll, "k": k, "n": n, "C": C, "p": p}


def map_levels(matches_map, era_label):
    """matches_map: dict map_name -> kept match count in the training era. Returns the level list."""
    lv = sorted([m for m, n in matches_map.items() if n >= MIN_MATCHES and m != REF_MAP])
    has_other = any(n < MIN_MATCHES for n in matches_map.values())
    levels = [REF_MAP] + lv + (["other"] if has_other else [])
    return levels


def map_level_of(map_name, levels):
    return map_name if map_name in levels else "other"


def design(df, levels, control, include_map, alive_col="ct_alive_minus_t", lead_col="ct_score_minus_t_score"):
    n = df.height
    cols = [np.ones(n)]
    names = ["intercept"]
    alive = lead = None
    if control in ("linear", "flex"):
        alive = df[alive_col].to_numpy().astype(np.float64)
        lead = df[lead_col].to_numpy().astype(np.float64)
    if control == "linear":
        cols += [alive, lead]
        names += ["alive_diff", "score_lead"]
    elif control == "flex":
        for a in range(-5, 6):
            if a == 0:
                continue
            cols.append((alive == a).astype(np.float64))
            names.append(f"alive={a}")
        lead_c = np.clip(lead, -12, 12)
        for s_ in range(-12, 13):
            if s_ == 0:
                continue
            cols.append((lead_c == s_).astype(np.float64))
            names.append(f"lead={s_}")
    elif control == "none":
        pass
    if include_map:
        mlv = df["map_level"].to_numpy()
        for lvl in levels:
            if lvl == REF_MAP:
                continue
            cols.append((mlv == lvl).astype(np.float64))
            names.append(f"map={lvl}")
    X = np.column_stack(cols)
    return X, names


def loss_vec(y, p):
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def wald_block(beta, V, idx):
    b = beta[idx]
    Vb = V[np.ix_(idx, idx)]
    return float(b @ np.linalg.solve(Vb, b)), len(idx)


def boot_diff_matches(d, match_ids, B, seed):
    _, inv = np.unique(match_ids, return_inverse=True)
    C = int(inv.max()) + 1
    sd = np.bincount(inv, weights=d, minlength=C)
    cnt = np.bincount(inv, minlength=C).astype(np.float64)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(C, np.full(C, 1.0 / C), size=B).astype(np.float64)
    vals = (W @ sd) / (W @ cnt)
    return float(d.mean()), pct_ci(vals)


# --------------------------------------------------------------------------- B
def section_b(rounds_by_era):
    hr("B. Heterogeneity across maps on rounds (no controls). LR test and cluster-robust Wald")
    out = {}
    for era in ERAS:
        k = kept_rounds(rounds_by_era[era]).sort("match_key")
        cnt = k.group_by("map_name").agg(pl.col("match_key").n_unique().alias("m"))
        mdict = {r["map_name"]: r["m"] for r in cnt.iter_rows(named=True)}
        levels = map_levels(mdict, era)
        k = k.with_columns(pl.col("map_name").map_elements(lambda x: map_level_of(x, levels),
                                                           return_dtype=pl.String).alias("map_level"))
        y = k["label_ct_win"].to_numpy().astype(np.float64)
        cl = k["match_key"].to_numpy()
        X0, _ = design(k, levels, "none", False)
        X1, names = design(k, levels, "none", True)
        f0 = fit_logit(X0, y, cl)
        f1 = fit_logit(X1, y, cl)
        lr = 2 * (f1["ll"] - f0["ll"])
        dfree = f1["k"] - f0["k"]
        p_lr = float(chi2_dist.sf(lr, dfree))
        idx = list(range(f0["k"], f1["k"]))
        W, dfw = wald_block(f1["beta"], f1["V"], idx)
        p_w = float(chi2_dist.sf(W, dfw))
        print(f"[{era}] rounds {k.height}, levels {levels}")
        print(f"  LR chi2 (map, df={dfree}) = {lr:.2f}, p = {p_lr:.3g}; cluster-robust Wald chi2 = {W:.2f}, "
              f"p = {p_w:.3g}")
        out[era] = {"levels": levels, "lr": lr, "df": dfree, "p_lr": p_lr, "wald": W, "p_wald": p_w,
                    "n_rounds": k.height, "n_matches": k["match_key"].n_unique()}
    RESULTS["B_heterogeneity"] = out
    return out


# --------------------------------------------------------------------------- C
def fit_states_report(era, states, levels_src, label, control, match_col="match_key", extra=""):
    """Fit controls-only (M1) and controls+map (M2) on states for one era. Returns dict."""
    s = states
    y = s["label_ct_win"].to_numpy().astype(np.float64)
    cl = s[match_col].to_numpy()
    s = s.with_columns(pl.col("map_name").map_elements(lambda x: map_level_of(x, levels_src),
                                                       return_dtype=pl.String).alias("map_level"))
    X1, n1 = design(s, levels_src, control, False)
    X2, n2 = design(s, levels_src, control, True)
    f1 = fit_logit(X1, y, cl)
    f2 = fit_logit(X2, y, cl)
    lr = 2 * (f2["ll"] - f1["ll"])
    dfree = f2["k"] - f1["k"]
    p_lr = float(chi2_dist.sf(lr, dfree))
    aic1 = -2 * f1["ll"] + 2 * f1["k"]
    aic2 = -2 * f2["ll"] + 2 * f2["k"]
    bic1 = -2 * f1["ll"] + np.log(f1["n"]) * f1["k"]
    bic2 = -2 * f2["ll"] + np.log(f2["n"]) * f2["k"]
    idx = list(range(f1["k"], f2["k"]))
    W, dfw = wald_block(f2["beta"], f2["V"], idx)
    p_w = float(chi2_dist.sf(W, dfw))
    return {"label": label, "control": control, "n_states": int(f1["n"]), "n_matches": int(len(np.unique(cl))),
            "ll_no_map_per_state": f1["ll"] / f1["n"], "ll_map_per_state": f2["ll"] / f2["n"],
            "delta_ll_per_state": (f2["ll"] - f1["ll"]) / f1["n"],
            "ll_total_no_map": f1["ll"], "ll_total_map": f2["ll"],
            "lr": lr, "df": dfree, "p_lr": p_lr, "wald": W, "p_wald": p_w,
            "aic_no_map": aic1, "aic_map": aic2, "delta_aic": aic2 - aic1,
            "bic_no_map": bic1, "bic_map": bic2, "delta_bic": bic2 - bic1,
            "names_map": n2, "beta_map": f2["beta"].tolist(), "V_map_diag": np.diag(f2["V"]).tolist(),
            "levels": levels_src, "f2_beta": f2["beta"], "f2_V": f2["V"]}


def print_fit(rep):
    print(f"  [{rep['label']}] control={rep['control']} states={rep['n_states']} matches={rep['n_matches']}")
    print(f"    LL/state  no map {rep['ll_no_map_per_state']:.6f}  with map {rep['ll_map_per_state']:.6f}  "
          f"delta {rep['delta_ll_per_state']:+.6f}")
    print(f"    LR chi2 = {rep['lr']:.2f} (df {rep['df']}), p = {rep['p_lr']:.3g};  cluster Wald chi2 = "
          f"{rep['wald']:.2f}, p = {rep['p_wald']:.3g}")
    print(f"    AIC no map {rep['aic_no_map']:.1f}  with map {rep['aic_map']:.1f}  dAIC {rep['delta_aic']:+.1f};  "
          f"dBIC {rep['delta_bic']:+.1f}")


def section_c(rounds_by_era, states_by_era):
    hr("C. Logistic regression on states: with and without map, controlling for alive diff and score lead")
    out = {}
    for era in ERAS:
        mdict = {r["map_name"]: r["m"] for r in
                 kept_rounds(rounds_by_era[era]).group_by("map_name")
                 .agg(pl.col("match_key").n_unique().alias("m")).iter_rows(named=True)}
        levels = map_levels(mdict, era)
        st = states_by_era[era].sort(["match_key", "round", "tick"])
        rep_lin = fit_states_report(era, st, levels, f"{era} states", "linear")
        print_fit(rep_lin)
        rep_flex = fit_states_report(era, st, levels, f"{era} states", "flex")
        print_fit(rep_flex)
        # Coefficients for the linear model with map
        names = rep_lin["names_map"]
        beta = rep_lin["f2_beta"]
        V = rep_lin["f2_V"]
        print(f"    linear controls with map: intercept {beta[0]:.4f}; alive_diff {beta[1]:.4f}; "
              f"score_lead {beta[2]:.4f} (per point)")
        coef_rows = []
        for i, nm in enumerate(names):
            if nm.startswith("map="):
                se = float(np.sqrt(V[i, i]))
                b = float(beta[i])
                lo, hi = b - 1.96 * se, b + 1.96 * se
                # implied CT win difference at the live start (alive 0, lead 0) vs reference map
                p_ref = float(expit(beta[0]))
                p_map = float(expit(beta[0] + b))
                coef_rows.append({"map": nm.replace("map=", ""), "logodds_vs_ref": b, "se": se,
                                  "ci_lo": lo, "ci_hi": hi, "p_live_start_map_minus_ref": p_map - p_ref})
        print(f"    map coefficients vs {REF_MAP} (log-odds, cluster-robust SE; probability difference at a "
              f"tied live start):")
        for c in coef_rows:
            print(f"      {c['map']:<12} b={c['logodds_vs_ref']:+.4f} se={c['se']:.4f} "
                  f"95% [{c['ci_lo']:+.4f},{c['ci_hi']:+.4f}]  p_live {c['p_live_start_map_minus_ref']:+.4f}")
        # round-level version: live-start rows only, lead only
        live = st.filter(pl.col("kind") == "live_start")
        rep_round = fit_states_report(era, live, levels, f"{era} live-start rows", "linear")
        print_fit(rep_round)
        out[era] = {"linear": {k: v for k, v in rep_lin.items() if k not in ("f2_beta", "f2_V", "names_map")},
                    "flex": {k: v for k, v in rep_flex.items() if k not in ("f2_beta", "f2_V", "names_map")},
                    "live_start_linear": {k: v for k, v in rep_round.items() if k not in ("f2_beta", "f2_V", "names_map")},
                    "levels": levels, "coef": coef_rows}
        # keep the per-era linear coefficients for the replication check
        out[era]["_coef_by_map"] = {c["map"]: (c["logodds_vs_ref"], c["se"]) for c in coef_rows}
    RESULTS["C_state_models"] = out
    return out


# --------------------------------------------------------------------------- D
def section_d(rounds_by_era, states_by_era):
    hr("D. 2026 primary split (bake-off make_splits, seed 0): fit on train, score test")
    r26 = kept_rounds(rounds_by_era["2026"])
    kept_m = sorted(r26["match_key"].unique().to_list())
    shuffled = list(kept_m)
    random.Random(SEED).shuffle(shuffled)
    n_test = round(0.20 * len(kept_m))
    test = set(shuffled[:n_test])
    pool = shuffled[n_test:]
    n_val = round(0.10 * len(pool))
    val = set(pool[:n_val])
    train = set(pool[n_val:])
    print(f"matches: train {len(train)}, val {len(val)}, test {len(test)} (kept set {len(kept_m)})")
    st = states_by_era["2026"].sort(["match_key", "round", "tick"])
    tr_m = {r["map_name"]: r["m"] for r in kept_rounds(rounds_by_era["2026"]).filter(
        pl.col("match_key").is_in(list(train))).group_by("map_name")
        .agg(pl.col("match_key").n_unique().alias("m")).iter_rows(named=True)}
    levels = map_levels(tr_m, "2026-train")
    print(f"map levels (train matches): {levels}")
    s_tr = st.filter(pl.col("match_key").is_in(list(train)))
    s_te = st.filter(pl.col("match_key").is_in(list(test)))
    s_tr = s_tr.with_columns(pl.col("map_name").map_elements(lambda x: map_level_of(x, levels), return_dtype=pl.String).alias("map_level"))
    s_te = s_te.with_columns(pl.col("map_name").map_elements(lambda x: map_level_of(x, levels), return_dtype=pl.String).alias("map_level"))
    res = {}
    for control in ("linear", "flex"):
        X1tr, _ = design(s_tr, levels, control, False)
        X2tr, n2 = design(s_tr, levels, control, True)
        X1te, _ = design(s_te, levels, control, False)
        X2te, _ = design(s_te, levels, control, True)
        ytr = s_tr["label_ct_win"].to_numpy().astype(np.float64)
        yte = s_te["label_ct_win"].to_numpy().astype(np.float64)
        f1 = fit_logit(X1tr, ytr, s_tr["match_key"].to_numpy())
        f2 = fit_logit(X2tr, ytr, s_tr["match_key"].to_numpy())
        p1 = expit(X1te @ f1["beta"])
        p2 = expit(X2te @ f2["beta"])
        l1 = loss_vec(yte, p1)
        l2 = loss_vec(yte, p2)
        est, ci = boot_diff_matches(l1 - l2, s_te["match_key"].to_numpy(), B, SEED)
        print(f"  control={control}: test states {len(yte)}, test LL no map {l1.mean():.6f}, "
              f"with map {l2.mean():.6f}; gain from map {est:+.6f} (match-cluster 95% CI {ci[0]:+.6f} to {ci[1]:+.6f})")
        res[control] = {"test_ll_no_map": float(l1.mean()), "test_ll_map": float(l2.mean()),
                        "gain_map": est, "gain_ci": [float(ci[0]), float(ci[1])], "n_test_states": int(len(yte)),
                        "n_test_matches": int(s_te["match_key"].n_unique())}
    res["levels"] = levels
    RESULTS["D_primary_split"] = res
    return res


# --------------------------------------------------------------------------- E
def section_e(rounds_by_era, states_by_era):
    hr("E. Cross-era transfer: fit on all 2026 kept states, score 2024 kept states")
    r26 = kept_rounds(rounds_by_era["2026"])
    mdict26 = {r["map_name"]: r["m"] for r in r26.group_by("map_name").agg(
        pl.col("match_key").n_unique().alias("m")).iter_rows(named=True)}
    levels26 = map_levels(mdict26, "2026")
    print(f"2026 levels: {levels26}")
    s26 = states_by_era["2026"].sort(["match_key", "round", "tick"]).with_columns(
        pl.col("map_name").map_elements(lambda x: map_level_of(x, levels26), return_dtype=pl.String).alias("map_level"))
    s24 = states_by_era["2024"].sort(["match_key", "round", "tick"]).with_columns(
        pl.col("map_name").map_elements(lambda x: map_level_of(x, levels26), return_dtype=pl.String).alias("map_level"))
    s24_no13 = s24.filter(pl.col("round") != DEFECT_ROUND)
    res = {}
    for control in ("linear", "flex"):
        X1, _ = design(s26, levels26, control, False)
        X2, n2 = design(s26, levels26, control, True)
        y26 = s26["label_ct_win"].to_numpy().astype(np.float64)
        f1 = fit_logit(X1, y26, s26["match_key"].to_numpy())
        f2 = fit_logit(X2, y26, s26["match_key"].to_numpy())
        for tag, s_eval in (("2024 all", s24), ("2024 excl round 13", s24_no13)):
            Xe1, _ = design(s_eval, levels26, control, False)
            Xe2, _ = design(s_eval, levels26, control, True)
            ye = s_eval["label_ct_win"].to_numpy().astype(np.float64)
            p1 = expit(Xe1 @ f1["beta"])
            p2 = expit(Xe2 @ f2["beta"])
            l1 = loss_vec(ye, p1)
            l2 = loss_vec(ye, p2)
            est, ci = boot_diff_matches(l1 - l2, s_eval["match_key"].to_numpy(), B, SEED)
            unseen = float((s_eval["map_level"] == "other").mean())
            print(f"  control={control} eval={tag}: states {len(ye)}, matches {s_eval['match_key'].n_unique()}, "
                  f"LL no map {l1.mean():.6f}, with map {l2.mean():.6f}, gain {est:+.6f} "
                  f"(95% CI {ci[0]:+.6f} to {ci[1]:+.6f}); share of states on 'other' level {unseen:.3f}")
            res[f"{control}|{tag}"] = {"ll_no_map": float(l1.mean()), "ll_map": float(l2.mean()),
                                        "gain_map": est, "gain_ci": [float(ci[0]), float(ci[1])],
                                        "states": int(len(ye)), "matches": int(s_eval["match_key"].n_unique()),
                                        "share_other": unseen}
    res["levels26"] = levels26
    RESULTS["E_cross_era"] = res
    return res


# --------------------------------------------------------------------------- F
def section_f(rounds_by_era):
    hr("F. Round-13 side defect: bomb-outcome mismatch by map (defect check)")
    out = {}
    for era in ERAS:
        r = rounds_by_era[era].filter(pl.col("bomb_mismatch").is_not_null())
        r13 = r.filter(pl.col("round") == DEFECT_ROUND)
        print(f"[{era}] rounds with a bomb outcome: {r.height}; mismatches: {int(r['bomb_mismatch'].sum())}; "
              f"round 13 with outcome: {r13.height}, mismatches: {int(r13['bomb_mismatch'].sum())}")
        by_map = (r13.group_by("map_name").agg(pl.len().alias("r13_outcome_rounds"),
                                                pl.col("bomb_mismatch").sum().alias("r13_mismatch"))
                  .with_columns((pl.col("r13_mismatch") / pl.col("r13_outcome_rounds")).alias("rate"))
                  .sort("r13_outcome_rounds", descending=True))
        if era == "2024":
            print(by_map)
            non13 = r.filter(pl.col("round") != DEFECT_ROUND)
            print(f"  non-round-13 mismatches: {int(non13['bomb_mismatch'].sum())} of {non13.height}")
        out[era] = {"outcome_rounds": r.height, "mismatches": int(r["bomb_mismatch"].sum()),
                    "r13_outcome_rounds": r13.height, "r13_mismatches": int(r13["bomb_mismatch"].sum()),
                    "by_map_r13": by_map.to_dicts()}
    # per map CT rate in 2024 kept set with and without round 13
    k24 = kept_rounds(rounds_by_era["2024"])
    k24_no13 = k24.filter(pl.col("round") != DEFECT_ROUND)
    a = (k24.group_by("map_name").agg(pl.len().alias("rounds_all"), pl.col("label_ct_win").mean().alias("ct_all"))
         .join(k24_no13.group_by("map_name").agg(pl.len().alias("rounds_no13"),
                                                  pl.col("label_ct_win").mean().alias("ct_no13")),
               on="map_name", how="left").sort("rounds_all", descending=True))
    print("\n2024 kept set, CT win rate by map: all rounds vs excluding round 13")
    print(a)
    out["2024_kept_all_vs_no13"] = a.to_dicts()
    RESULTS["F_defect"] = out
    return out


# --------------------------------------------------------------------------- G
def section_g(rounds_by_era):
    hr("G. Sensitivity: 2026 kept set with 24-round matches only; raw (unfiltered) rates per era")
    k = kept_rounds(rounds_by_era["2026"])
    k24 = k.filter(pl.col("max_rounds") == 24)
    a = (k24.group_by("map_name").agg(pl.col("match_key").n_unique().alias("matches"), pl.len().alias("rounds"),
                                       pl.col("label_ct_win").mean().alias("ct_rate"))
         .sort("rounds", descending=True))
    print("2026 kept set, 24-round matches only:")
    print(a)
    out = {"2026_24round_only": a.to_dicts()}
    for era in ERAS:
        raw = rounds_by_era[era].filter(pl.col("winner").is_in([T, CT]))
        rr = (raw.group_by("map_name").agg(pl.col("match_key").n_unique().alias("matches"), pl.len().alias("rounds"),
                                            pl.col("label_ct_win").mean().alias("ct_rate"))
              .sort("rounds", descending=True))
        print(f"\n{era} raw (all rounds with a winner, no filters):")
        print(rr)
        out[f"{era}_raw"] = rr.to_dicts()
    RESULTS["G_sensitivity"] = out
    return out


def main():
    t0 = time.time()
    rounds_by_era, states_by_era = {}, {}
    for era in ERAS:
        rounds_by_era[era], states_by_era[era] = load(era)
        print(f"loaded {era}: rounds {rounds_by_era[era].height}, states {states_by_era[era].height}")
    section_a(rounds_by_era)
    section_b(rounds_by_era)
    section_c(rounds_by_era, states_by_era)
    section_d(rounds_by_era, states_by_era)
    section_e(rounds_by_era, states_by_era)
    section_f(rounds_by_era)
    section_g(rounds_by_era)
    # replication: per-map linear coefficients across eras on shared levels
    hr("H. Replication: per-map adjusted log-odds (linear controls) across eras")
    c26 = RESULTS["C_state_models"]["2026"]["_coef_by_map"]
    c24 = RESULTS["C_state_models"]["2024"]["_coef_by_map"]
    shared = sorted(set(c26) & set(c24))
    rows = []
    for m in shared:
        b26, s26 = c26[m]
        b24, s24 = c24[m]
        z = (b26 - b24) / np.sqrt(s26 ** 2 + s24 ** 2)
        rows.append({"map": m, "b2026": b26, "se2026": s26, "b2024": b24, "se2024": s24, "z_diff": float(z)})
        print(f"  {m:<12} 2026 b={b26:+.4f} (se {s26:.4f})   2024 b={b24:+.4f} (se {s24:.4f})   z_diff={z:+.2f}")
    if len(shared) >= 3:
        x = np.array([r["b2026"] for r in rows])
        y = np.array([r["b2024"] for r in rows])
        r_ = float(np.corrcoef(x, y)[0, 1])
        print(f"  shared levels {len(shared)}: Pearson r of per-map log-odds across eras = {r_:+.3f}")
    else:
        r_ = None
    RESULTS["H_replication"] = {"rows": rows, "pearson_r": r_, "shared": shared}
    print(f"\nrun time {time.time() - t0:.1f}s")

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items() if not str(k).startswith("_")}
        if isinstance(o, list):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, float) and (o != o):
            return None
        return o

    (SCRATCH / "M2_results.json").write_text(json.dumps(clean(RESULTS), indent=1, default=str), encoding="utf-8")
    print(f"wrote {SCRATCH / 'M2_results.json'}")


if __name__ == "__main__":
    main()
