"""M5: is the 2026-10-08 map-level CT shift significant, and does it survive controlling for batch?

Input: scratchpad M5_rounds.parquet (written by scripts\\swarm2\\M5_load.py) plus header fields
(server_name, build_num) read from the tome. Clean 5v5 non-wingman rounds, duplicates removed.

Batch = header match_date day: 'b1008' (2026-10-08) vs 'b1001_1007' (2026-10-01..07). The 3 matches
dated outside both are excluded from batch models (ASSUMPTION, see notes in M5_load.py).
Source = header server_name starts with 'FACEIT' (ASSUMPTION: this is the hosting-source marker that
co-varies with batch; see M5_source_crosstab.py).

Core maps = the 7 maps with >= 40 matches in both batches. Other maps are pooled into 'rare' for the
main-effect models and excluded from the per-map interaction models.

Outcome: ct_win (side-based, 1 = CT side won the round). Clusters = match_key.

Sections
  1. Per-map batch differences: round-level CT rate, match-cluster bootstrap CI (resample matches
     within map x batch), permutation p (batch labels permuted across matches within map), Holm across
     core maps, and a match-level Welch / Mann-Whitney on per-match CT share.
  2. Cluster-robust logistic models (CR1 sandwich by match): round-bucket controls throughout.
       M0 round only; M1 + map; M2 + batch (additive, core maps); M3 + batch x map (interaction);
       M4 + faceit (additive); M5 + faceit x map; M6 batch + faceit together; M7 M3 + score lead.
     Joint Wald tests for map effects and interactions.
  3. Source-fixed batch contrast: non-FACEIT matches only (batch varies, source fixed).
  4. Held-out-batch validation of pre-round map priors: random match split vs forward (train
     1001-1007, test 1008) vs reverse (train 1008, test 1001-1007). Metrics: log loss, Brier, AUC.

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\M5_map_batch.py
"""
import glob
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
SCRATCH = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\M5")
RPQ = SCRATCH / "M5_rounds.parquet"

CORE = ["de_ancient", "de_anubis", "de_cache", "de_dust2", "de_inferno", "de_mirage", "de_nuke"]
B_BOOT = 5000
B_PERM = 5000
RNG = np.random.default_rng(20261013)
MAIN = ["b1001_1007", "b1008"]


def hdr(msg):
    print("\n" + "=" * 100)
    print(msg)
    print("=" * 100)


# ----------------------------------------------------------------------------- data
def load():
    r = pl.read_parquet(RPQ)
    fs = sorted(glob.glob(str(TOME / "header" / "dataframe_*")))
    h = (pl.scan_parquet(fs).select(["key", "server_name", "build_num"]).collect()
         .rename({"key": "match_key"})
         .with_columns(pl.col("server_name").str.starts_with("FACEIT").alias("faceit")))
    d = r.join(h.select(["match_key", "faceit", "build_num"]), on="match_key", how="left")
    assert d.filter(pl.col("faceit").is_null()).height == 0, "header join miss"
    d = d.filter(pl.col("batch").is_in(MAIN)).with_columns(
        (pl.col("batch") == "b1008").cast(pl.Int8).alias("b1008"),
        pl.col("faceit").cast(pl.Int8).alias("faceit_i"),
        pl.when(pl.col("map_name").is_in(CORE)).then(pl.col("map_name")).otherwise(pl.lit("rare")).alias("map_pool"),
        pl.when(pl.col("round_no") > 24).then(pl.lit(25)).otherwise(pl.col("round_no")).alias("rb"),
        pl.col("score_lead_ct").cast(pl.Int64).alias("lead"),
    )
    return d


# ----------------------------------------------------------------------------- logistic, cluster-robust
def fit_logit(X, y, cl, max_iter=60, tol=1e-9):
    n, k = X.shape
    beta = np.zeros(k)
    for _ in range(max_iter):
        eta = X @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        w = p * (1 - p)
        g = X.T @ (y - p)
        H = (X * w[:, None]).T @ X
        step = np.linalg.solve(H + 1e-10 * np.eye(k), g)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    eta = X @ beta
    p = 1.0 / (1.0 + np.exp(-eta))
    w = p * (1 - p)
    H = (X * w[:, None]).T @ X
    bread = np.linalg.inv(H)
    s = X * (y - p)[:, None]
    G = cl.max() + 1
    S = np.zeros((G, k))
    np.add.at(S, cl, s)
    meat = S.T @ S
    c = (G / (G - 1)) * ((n - 1) / (n - k))
    V = c * bread @ meat @ bread
    ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return beta, V, ll, G


def wald(beta, V, idx):
    idx = list(idx)
    b = beta[idx]
    Vs = V[np.ix_(idx, idx)]
    W = float(b @ np.linalg.solve(Vs, b))
    return W, len(idx), float(stats.chi2.sf(W, len(idx)))


def design(d, parts):
    cols, names = [np.ones(d.height)], ["const"]
    idx = {}
    for part in parts:
        if part == "round":
            rb = d["rb"].to_numpy()
            start = len(cols)
            for lvl in range(2, 26):
                cols.append((rb == lvl).astype(float))
                names.append(f"rb{lvl}")
            idx["round"] = list(range(start, len(cols)))
        elif part == "map":
            mp = d["map_pool"].to_numpy()
            present = set(np.unique(mp).tolist())
            levels = [m for m in CORE if m != "de_mirage" and m in present] + (["rare"] if "rare" in present else [])
            start = len(cols)
            for lvl in levels:
                cols.append((mp == lvl).astype(float))
                names.append(f"map_{lvl}")
            idx["map"] = list(range(start, len(cols)))
        elif part == "batch":
            idx["batch"] = [len(cols)]
            cols.append(d["b1008"].to_numpy().astype(float))
            names.append("b1008")
        elif part == "faceit":
            idx["faceit"] = [len(cols)]
            cols.append(d["faceit_i"].to_numpy().astype(float))
            names.append("faceit")
        elif part == "batch_x_map":
            mp = d["map_pool"].to_numpy()
            b = d["b1008"].to_numpy().astype(float)
            present = set(np.unique(mp).tolist())
            start = len(cols)
            for lvl in [m for m in CORE if m != "de_mirage" and m in present]:
                cols.append(b * (mp == lvl))
                names.append(f"b1008x_{lvl}")
            idx["batch_x_map"] = list(range(start, len(cols)))
        elif part == "faceit_x_map":
            mp = d["map_pool"].to_numpy()
            f = d["faceit_i"].to_numpy().astype(float)
            present = set(np.unique(mp).tolist())
            start = len(cols)
            for lvl in [m for m in CORE if m != "de_mirage" and m in present]:
                cols.append(f * (mp == lvl))
                names.append(f"faceitx_{lvl}")
            idx["faceit_x_map"] = list(range(start, len(cols)))
        elif part == "lead":
            ld = d["lead"].to_numpy()
            start = len(cols)
            for lvl in [-4, -3, -2, -1, 1, 2, 3, 4]:
                cols.append((ld == lvl).astype(float))
                names.append(f"lead{lvl}")
            idx["lead"] = list(range(start, len(cols)))
    return np.column_stack(cols), names, idx


def run_model(d, parts, label):
    X, names, idx = design(d, parts)
    y = d["ct_win"].to_numpy().astype(float)
    cl = np.unique(d["match_key"].to_numpy(), return_inverse=True)[1].astype(np.int64)
    t0 = time.time()
    beta, V, ll, G = fit_logit(X, y, cl)
    res = {"label": label, "ll": ll, "k": X.shape[1], "G": G, "beta": beta, "V": V, "names": names, "idx": idx,
           "n": len(y)}
    print(f"{label}: n={len(y)} clusters={G} params={X.shape[1]} loglik={ll:.2f} ({time.time()-t0:.1f}s)")
    return res


def coef_row(res, key):
    i = res["idx"][key][0]
    b, se = res["beta"][i], np.sqrt(res["V"][i, i])
    return b, se, np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se)


# ----------------------------------------------------------------------------- per-map tests
def per_map(d):
    hdr("1. Per-map batch difference in CT rate (core maps, 2 batches)")
    print("Diff = CT% (b1008) minus CT% (b1001_1007), round-level pooled rates.")
    print(f"Bootstrap: {B_BOOT} resamples of matches within map x batch (match-cluster). "
          f"Permutation: {B_PERM} draws of batch labels across matches within map.")
    rows = []
    z_obs, perm_store = {}, {}
    maps = CORE
    per_map_data = {}
    for m in maps:
        sub = d.filter(pl.col("map_name") == m)
        mm = (sub.group_by(["match_key", "batch"])
              .agg(pl.len().alias("r"), pl.col("ct_win").sum().alias("w"), pl.col("ct_win").mean().alias("share"))
              .sort(["batch", "match_key"]))
        a = mm.filter(pl.col("batch") == "b1001_1007")
        bb = mm.filter(pl.col("batch") == "b1008")
        per_map_data[m] = (a, bb)
    for m in maps:
        a, bb = per_map_data[m]
        ra, wa = a["r"].to_numpy(), a["w"].to_numpy()
        rb_, wb = bb["r"].to_numpy(), bb["w"].to_numpy()
        pa, pb = wa.sum() / ra.sum(), wb.sum() / rb_.sum()
        diff = pb - pa
        # bootstrap
        ia = RNG.integers(0, len(ra), size=(B_BOOT, len(ra)))
        ib = RNG.integers(0, len(rb_), size=(B_BOOT, len(rb_)))
        boot_a = wa[ia].sum(1) / ra[ia].sum(1)
        boot_b = wb[ib].sum(1) / rb_[ib].sum(1)
        bd = boot_b - boot_a
        lo, hi = np.percentile(bd, [2.5, 97.5])
        se = bd.std(ddof=1)
        # match-level tests on per-match CT share
        sa = a["share"].to_numpy()
        sb = bb["share"].to_numpy()
        t_p = stats.ttest_ind(sb, sa, equal_var=False).pvalue
        try:
            mw_p = stats.mannwhitneyu(sb, sa, alternative="two-sided").pvalue
        except ValueError:
            mw_p = float("nan")
        z_obs[m] = diff / se
        rows.append({"map": m, "matches_1001_1007": len(ra), "rounds_1001_1007": int(ra.sum()),
                     "ct_pct_1001_1007": 100 * pa, "matches_1008": len(rb_), "rounds_1008": int(rb_.sum()),
                     "ct_pct_1008": 100 * pb, "diff_pp": 100 * diff, "boot_se_pp": 100 * se,
                     "ci95_lo_pp": 100 * lo, "ci95_hi_pp": 100 * hi, "match_welch_p": t_p, "match_mw_p": mw_p})
    # permutation: pool matches within map, permute batch labels
    perm_diff = {m: [] for m in maps}
    perm_z = {m: [] for m in maps}
    for m in maps:
        a, bb = per_map_data[m]
        allr = np.concatenate([a["r"].to_numpy(), bb["r"].to_numpy()])
        allw = np.concatenate([a["w"].to_numpy(), bb["w"].to_numpy()])
        n_a = len(a)
        n_tot = len(allr)
        se_obs = [r for r in rows if r["map"] == m][0]["boot_se_pp"] / 100.0
        for _ in range(B_PERM):
            perm = RNG.permutation(n_tot)
            ga, gb = perm[:n_a], perm[n_a:]
            pa = allw[ga].sum() / allr[ga].sum()
            pb = allw[gb].sum() / allr[gb].sum()
            dd = pb - pa
            perm_diff[m].append(dd)
            perm_z[m].append(dd / se_obs if se_obs > 0 else 0.0)
    for r in rows:
        m = r["map"]
        pd_ = np.array(perm_diff[m])
        r["perm_p"] = float(np.mean(np.abs(pd_) >= abs(r["diff_pp"] / 100.0) - 1e-15))
    # Holm across core maps on permutation p
    order = sorted(range(len(rows)), key=lambda i: rows[i]["perm_p"])
    m_ = len(rows)
    running = 0.0
    holm = [0.0] * m_
    for rank, i in enumerate(order):
        adj = (m_ - rank) * rows[i]["perm_p"]
        running = max(running, min(1.0, adj))
        holm[i] = running
    for i, r in enumerate(rows):
        r["perm_p_holm"] = holm[i]
    # global max-|z| permutation test
    z_perm_max = np.max(np.abs(np.column_stack([np.array(perm_z[m]) for m in maps])), axis=1)
    z_obs_max = max(abs(z_obs[m]) for m in maps)
    p_global = float(np.mean(z_perm_max >= z_obs_max))
    rows_df = pl.DataFrame(rows)
    print(rows_df.select(["map", "matches_1001_1007", "rounds_1001_1007", "ct_pct_1001_1007",
                          "matches_1008", "rounds_1008", "ct_pct_1008"]))
    print(rows_df.select(["map", "diff_pp", "boot_se_pp", "ci95_lo_pp", "ci95_hi_pp",
                          "perm_p", "perm_p_holm", "match_welch_p", "match_mw_p"]).with_columns(
        pl.col(["diff_pp", "boot_se_pp", "ci95_lo_pp", "ci95_hi_pp"]).round(2),
        pl.col(["perm_p", "perm_p_holm", "match_welch_p", "match_mw_p"]).round(4)))
    print(f"\nGlobal max-|z| permutation p (any core map shifts between batches): {p_global:.4f}")
    print(f"Observed max |z| across core maps: {z_obs_max:.3f} (z = diff / bootstrap SE)")
    return rows_df, p_global


def overall_batch(d):
    hdr("1b. Overall batch CT rate (no map control) and map mix by batch")
    g = d.group_by("batch").agg(pl.len().alias("rounds"), pl.col("match_key").n_unique().alias("matches"),
                                (pl.col("ct_win").mean() * 100).round(2).alias("ct_pct"))
    print(g.sort("batch"))
    mix = (d.group_by(["map_pool", "batch"]).agg(pl.len().alias("rounds"))
           .with_columns((pl.col("rounds") / pl.col("rounds").sum().over("batch") * 100).round(2).alias("pct_of_batch"))
           .sort(["map_pool", "batch"]))
    print(mix)
    # direct standardisation: CT rate of b1008 if it had the b1001_1007 map mix, and vice versa
    rates = d.group_by(["map_pool", "batch"]).agg(pl.col("ct_win").mean().alias("p"), pl.len().alias("n"))
    mixp = d.group_by(["map_pool", "batch"]).agg(pl.len().alias("n")).with_columns(
        (pl.col("n") / pl.col("n").sum().over("batch")).alias("w"))
    m_all = rates.join(mixp.select(["map_pool", "batch", "w"]), on=["map_pool", "batch"])
    r1 = m_all.filter(pl.col("batch") == "b1008").select(["map_pool", "p"]).rename({"p": "p1008"})
    r0 = m_all.filter(pl.col("batch") == "b1001_1007").select(["map_pool", "p"]).rename({"p": "p1001"})
    w0 = mixp.filter(pl.col("batch") == "b1001_1007").select(["map_pool", "w"]).rename({"w": "w1001"})
    w1 = mixp.filter(pl.col("batch") == "b1008").select(["map_pool", "w"]).rename({"w": "w1008"})
    j = r1.join(r0, on="map_pool").join(w0, on="map_pool").join(w1, on="map_pool")
    cr1_mix0 = float((j["p1008"] * j["w1001"]).sum())
    cr0_mix1 = float((j["p1001"] * j["w1008"]).sum())
    print(f"\nDirect standardisation (map mix): b1008 rates on b1001 map mix = {100*cr1_mix0:.2f}%; "
          f"b1001 rates on b1008 map mix = {100*cr0_mix1:.2f}%")
    print(f"Raw: b1008 {100*float(d.filter(pl.col('batch')=='b1008')['ct_win'].mean()):.2f}%, "
          f"b1001_1007 {100*float(d.filter(pl.col('batch')=='b1001_1007')['ct_win'].mean()):.2f}%")


def models(d):
    hdr("2. Cluster-robust logistic models (clusters = matches, CR1). Round-bucket controls throughout.")
    print("Batch = b1008 indicator. Faceit = FACEIT-hosted server indicator. Core-map sample "
          "(7 core maps, both batches) for models with map terms.")
    dc = d.filter(pl.col("map_pool") != "rare")
    print(f"core-map sample: rounds {dc.height}, matches {dc['match_key'].n_unique()}; "
          f"all-map sample (for M0 and M_all): rounds {d.height}, matches {d['match_key'].n_unique()}")
    out = {}
    out["M0"] = run_model(dc, ["round"], "M0 round only")
    out["M1"] = run_model(dc, ["round", "map"], "M1 round + map")
    out["M2"] = run_model(dc, ["round", "map", "batch"], "M2 round + map + batch (additive)")
    out["M3"] = run_model(dc, ["round", "map", "batch", "batch_x_map"], "M3 round + map + batch + batch x map")
    out["M4"] = run_model(dc, ["round", "map", "faceit"], "M4 round + map + faceit (additive)")
    out["M5"] = run_model(dc, ["round", "map", "faceit", "faceit_x_map"], "M5 round + map + faceit + faceit x map")
    out["M6"] = run_model(dc, ["round", "map", "batch", "faceit"], "M6 round + map + batch + faceit")
    out["M7"] = run_model(dc, ["round", "map", "batch", "batch_x_map", "lead"],
                          "M7 M3 + score lead (sensitivity; lead is a live-state variable)")
    out["M8"] = run_model(dc, ["round", "map", "faceit", "batch", "faceit_x_map"],
                          "M8 round + map + faceit + batch + faceit x map")

    print("\n-- coefficients (OR = exp(b); CI from CR1 sandwich) --")
    for key, lab in [("M2", "batch"), ("M4", "faceit"), ("M6", "batch"), ("M6", "faceit")]:
        b, se, orr, lo, hi = coef_row(out[key], lab)
        print(f"{key} {lab:<8}: b={b:+.4f} se={se:.4f} OR={orr:.3f} [{lo:.3f}, {hi:.3f}] "
              f"p={2*stats.norm.sf(abs(b/se)):.4f}")

    print("\n-- joint Wald tests --")
    def wt(res, key, lab):
        W, df, p = wald(res["beta"], res["V"], res["idx"][key])
        print(f"{lab:<60}: chi2={W:.2f} df={df} p={p:.4g}")
        return W, df, p
    wt(out["M1"], "map", "map main effects | round (M1)")
    wt(out["M2"], "map", "map main effects | round + batch (M2)")
    wt(out["M4"], "map", "map main effects | round + faceit (M4)")
    wt(out["M3"], "batch_x_map", "batch x map interaction | round, map, batch (M3)")
    wt(out["M5"], "faceit_x_map", "faceit x map interaction | round, map, faceit (M5)")
    wt(out["M8"], "faceit_x_map", "faceit x map interaction | round, map, faceit, batch (M8)")
    wt(out["M7"], "batch_x_map", "batch x map interaction | M7 incl. score lead")
    # LR-like comparison (not cluster-valid, indicative only)
    print("\n-- indicative log-likelihood differences (not cluster-valid; for direction only) --")
    for a, b_ in [("M1", "M2"), ("M2", "M3"), ("M1", "M4"), ("M4", "M5"), ("M2", "M6"), ("M4", "M6")]:
        print(f"{a} -> {b_}: dLL = {out[b_]['ll'] - out[a]['ll']:+.2f} (params {out[a]['k']} -> {out[b_]['k']})")
    return out


def source_fixed(d):
    hdr("3. Source-fixed batch contrast: non-FACEIT matches only (batch varies, source fixed)")
    nf = d.filter((pl.col("faceit") == False) & (pl.col("map_pool") != "rare"))
    print(f"non-FACEIT core-map rounds {nf.height}, matches {nf['match_key'].n_unique()} "
          f"by batch: {nf.group_by('batch').agg(pl.col('match_key').n_unique()).sort('batch').rows()}")
    res0 = run_model(nf, ["round", "map"], "S0 non-FACEIT round + map")
    res1 = run_model(nf, ["round", "map", "batch"], "S1 non-FACEIT round + map + batch")
    b, se, orr, lo, hi = coef_row(res1, "batch")
    print(f"S1 batch (b1008 vs b1001_1007, non-FACEIT only): b={b:+.4f} se={se:.4f} "
          f"OR={orr:.3f} [{lo:.3f}, {hi:.3f}] p={2*stats.norm.sf(abs(b/se)):.4f}")
    print("CT% by batch in non-FACEIT core maps:")
    print(nf.group_by(["map_pool", "batch"]).agg(pl.len().alias("rounds"), pl.col("match_key").n_unique().alias("matches"),
                                              (pl.col("ct_win").mean()*100).round(2).alias("ct_pct")).sort(["map_pool", "batch"]))
    print("Within non-FACEIT, the de_cache / de_dust2 rows are small (see matches column). "
          "Map-level interaction is not estimable with reliable power here, so only the overall batch term is tested.")


# ----------------------------------------------------------------------------- held-out batch validation
def onehot(d, use_map, use_round=True):
    feats = []
    rb = d["rb"].to_numpy()
    feats.append(np.eye(26)[rb][:, 1:])
    if use_map:
        mp = d["map_pool"].to_numpy()
        levels = CORE + ["rare"]
        feats.append(np.column_stack([(mp == lv).astype(float) for lv in levels]))
    return np.column_stack(feats)


def eval_pred(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return dict(logloss=float(log_loss(y, p, labels=[0, 1])), brier=float(brier_score_loss(y, p)),
                auc=float(roc_auc_score(y, p)))


def validation(d):
    hdr("4. Held-out-batch validation of pre-round priors (round, map, round + map)")
    print("Models: constant (train CT rate); round bucket only; round + map (logistic, C=1, one-hot).")
    print("Designs: random match split 80/20 (5 seeds, averaged); forward (train 1001-1007, test 1008); "
          "reverse (train 1008, test 1001-1007).")
    keys = np.array(sorted(d["match_key"].unique().to_list()))
    y_all = d["ct_win"].to_numpy()
    rows = []
    def fit_eval(tr, te, name, design_name):
        y_tr = tr["ct_win"].to_numpy()
        y_te = te["ct_win"].to_numpy()
        const = np.full(len(y_te), y_tr.mean())
        rows.append({"design": design_name, "model": "constant", **eval_pred(y_te, const), "n_test": len(y_te)})
        for use_map, lab in [(False, "round only"), (True, "round + map")]:
            Xtr = onehot(tr, use_map)
            Xte = onehot(te, use_map)
            clf = LogisticRegression(C=1.0, max_iter=2000)
            clf.fit(Xtr, y_tr)
            p = clf.predict_proba(Xte)[:, 1]
            rows.append({"design": design_name, "model": lab, **eval_pred(y_te, p), "n_test": len(y_te)})
    seeds = [0, 1, 2, 3, 4]
    agg = {}
    for s in seeds:
        rng = np.random.default_rng(s)
        test_keys = set(rng.choice(keys, size=int(0.2 * len(keys)), replace=False).tolist())
        mk = d["match_key"].to_numpy()
        is_te = np.array([k in test_keys for k in mk])
        tr = d.filter(pl.Series(~is_te))
        te = d.filter(pl.Series(is_te))
        before = len(rows)
        fit_eval(tr, te, "", f"random_seed{s}")
    rand_rows = [r for r in rows if r["design"].startswith("random")]
    for model in ["constant", "round only", "round + map"]:
        vals = [r for r in rand_rows if r["model"] == model]
        ll = np.mean([v["logloss"] for v in vals])
        br = np.mean([v["brier"] for v in vals])
        au = np.mean([v["auc"] for v in vals])
        rows.append({"design": "random_mean5", "model": model, "logloss": ll, "brier": br, "auc": au,
                     "n_test": vals[0]["n_test"]})
    rows = [r for r in rows if not r["design"].startswith("random_seed")]
    b1001 = d.filter(pl.col("batch") == "b1001_1007")
    b1008 = d.filter(pl.col("batch") == "b1008")
    fit_eval(b1001, b1008, "", "forward_train1001_test1008")
    fit_eval(b1008, b1001, "", "reverse_train1008_test1001")
    out = pl.DataFrame(rows)
    print(out.with_columns(pl.col(["logloss", "brier", "auc"]).round(5)))
    # skill vs constant for each design
    print("\nLog-loss skill vs constant (positive = better than constant), per design:")
    for dsg in ["random_mean5", "forward_train1001_test1008", "reverse_train1008_test1001"]:
        sub = out.filter(pl.col("design") == dsg)
        c = float(sub.filter(pl.col("model") == "constant")["logloss"][0])
        for mdl in ["round only", "round + map"]:
            v = float(sub.filter(pl.col("model") == mdl)["logloss"][0])
            print(f"  {dsg:<30} {mdl:<12} skill {100*(c-v)/c:+.3f}%  (delta log loss {v-c:+.5f})")
    # map-feature gain comparison
    print("\nMap feature gain (round + map minus round only; negative = map helps):")
    for dsg in ["random_mean5", "forward_train1001_test1008", "reverse_train1008_test1001"]:
        sub = out.filter(pl.col("design") == dsg)
        r0 = float(sub.filter(pl.col("model") == "round only")["logloss"][0])
        r1 = float(sub.filter(pl.col("model") == "round + map")["logloss"][0])
        print(f"  {dsg:<30} delta log loss = {r1-r0:+.5f}")
    return out


def main():
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_fmt_str_lengths(30)
    d = load()
    print(f"loaded clean rounds (1001_1007 + 1008): {d.height}, matches {d['match_key'].n_unique()}")
    overall_batch(d)
    rows_df, p_global = per_map(d)
    rows_df.write_csv(SCRATCH / "M5_per_map.csv")
    models(d)
    source_fixed(d)
    validation(d)


if __name__ == "__main__":
    main()
