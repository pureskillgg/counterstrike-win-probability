"""Ablation analysis: validation log loss (1,168 matches) and GDT (297-match population), per arm, seed means, paired differences,
and the selection rule. Uses score_gdt.py's own functions (logloss, wboot, situations, draw, read_rows) and its table; the
match-cluster bootstrap is score_gdt's wboot with 1,000 resamples and seed 0.

Arms: base (59 inputs), base-GEO (75), full (83). Pairs: 83 minus 75, 75 minus 59, 83 minus 59.
Log loss differences are on the validation rows (1,515,228 rows over 1,168 matches). GDT differences are on the 297-match
population's datapoints (score_gdt draw, seed 0).
"""
import json
import os
import sys

import numpy as np
import polars as pl

SCRATCH = "C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/ablation_ll_gdt"
GDT_DIR = "C:/Users/13144/Documents/pureskill/counterstrike-win-probability/evaluation/intern_gdt"
TABLE = GDT_DIR + "/table.csv"
sys.path.insert(0, SCRATCH + "/code")
sys.path.insert(0, GDT_DIR)
import build_gdt_rows as B  # noqa: E402
import score_gdt as G  # noqa: E402

ARMS = {"base": ("p_base", 59), "base-GEO": ("p_geo", 75), "full": ("p_full", 83)}
PAIRS = [("full", "base-GEO"), ("base-GEO", "base"), ("full", "base")]  # "83 minus 75", "75 minus 59", "83 minus 59"
SEEDS = [0, 1, 2]
MODELS = ["p_base", "p_geo", "p_full"]
REPS, BOOT_SEED = 1000, 0
PINNED = {"base": 0.45381}  # GPU reproduction target, docs/sprint/validation_gpu_fast/report.md


def ll_worker(p, y):
    """The sprint worker's log loss (sprint_worker.fit_cell_prepared), row by row."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return -(y * np.log(np.clip(p, 1e-15, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-15, 1)))


def boot_mean(per_match_sum, per_match_n, reps, seed):
    """Match-cluster bootstrap of a row-weighted mean, as score_gdt.wboot does it: resample matches, ratio of sums."""
    rng = np.random.default_rng(seed)
    m = len(per_match_sum)
    stats = []
    for _ in range(reps):
        pick = rng.integers(0, m, m)
        stats.append(per_match_sum[pick].sum() / per_match_n[pick].sum())
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(per_match_sum.sum() / per_match_n.sum()), [float(lo), float(hi)]


def fit_results():
    out = {}
    for arm in ARMS:
        for s in SEEDS:
            path = f"{SCRATCH}/out/fits/results/{arm.replace('+', '-')}_d4_e0.03_m20_s{s}.json"
            r = json.load(open(path, encoding="utf-8"))
            out[(arm, s)] = r
    return out


def logloss_section(wide, fits):
    y = wide["label_ct_win"].to_numpy()
    matches = wide["match_key"].to_numpy()
    uniq, inv = np.unique(matches, return_inverse=True)
    n_rows = np.bincount(inv).astype(np.float64)
    n_matches = len(uniq)
    rows_loss = {}
    seed_ll = {}
    for arm, (col, _) in ARMS.items():
        per_seed = []
        for s in SEEDS:
            l = ll_worker(wide[f"{col}_s{s}"].to_numpy(), y)
            per_seed.append(l)
            seed_ll[(arm, s)] = float(l.mean())
        rows_loss[arm] = np.mean(per_seed, axis=0)  # seed mean of the row loss
    seed_mean_ll = {arm: float(np.mean([seed_ll[(arm, s)] for s in SEEDS])) for arm in ARMS}
    match_sum = {arm: np.bincount(inv, weights=rows_loss[arm]) for arm in ARMS}
    paired = {}
    for a, b in PAIRS:
        d_sum = match_sum[a] - match_sum[b]
        point, ci = boot_mean(d_sum, n_rows, REPS, BOOT_SEED)
        per_seed = {}
        for s in SEEDS:
            la = ll_worker(wide[f"{ARMS[a][0]}_s{s}"].to_numpy(), y)
            lb = ll_worker(wide[f"{ARMS[b][0]}_s{s}"].to_numpy(), y)
            per_seed[s] = float((la - lb).mean())
        paired[f"{a} minus {b}"] = {"point": point, "ci95": ci, "per_seed_point": per_seed,
                                    "unit": "matches", "n_matches": n_matches, "rows": int(len(y))}
    return {"seed_ll": {f"{a}|{s}": v for (a, s), v in seed_ll.items()}, "seed_mean_ll": seed_mean_ll, "paired": paired,
            "n_matches": n_matches, "n_rows": int(len(y))}


def gdt_section(fits):
    table = pl.read_csv(TABLE, schema_overrides={"tb": pl.Utf8, "econ_ct": pl.Utf8, "econ_t": pl.Utf8})
    table = table.select(G.CELL + [pl.col("p_ct").alias("p_table")])
    pts_by_seed, counts = {}, {}
    for s in SEEDS:
        df = G.read_rows(f"{B.OUT_DIR}/rows_s{s}.parquet")
        df = df.select(G.NEED + MODELS)
        df = df.with_columns(pl.when(pl.col("planted") == 0).then(pl.col("since_plant").fill_null(0))
                             .otherwise(pl.col("since_plant")).alias("since_plant"))
        df = df.drop_nulls()
        rows_after_drop = df.height
        df = G.situations(df)
        pts = G.draw(df, MODELS, 0).join(table, on=G.CELL, how="left")
        lost = pts.filter(pl.col("p_table").is_null()).height
        pts = pts.filter(pl.col("p_table").is_not_null())
        pts = pts.join(pts.group_by(G.CELL).agg(pl.len().alias("n_s")), on=G.CELL, how="left")
        pts_by_seed[s] = pts
        counts[s] = {"rows": rows_after_drop, "lost_datapoints": lost, "matches": int(pts["match"].n_unique()),
                     "rounds": int(pts.select("match", "round").n_unique()), "datapoints": pts.height,
                     "situations": int(pts.select(G.CELL).n_unique()),
                     "situations_with_500_or_more": int(pts.filter(pl.col("n_s") >= G.SIT_MIN).select(G.CELL).n_unique())}
    base = pts_by_seed[0]
    key_cols = ["match", "round", "second"] + G.CELL
    for s in SEEDS[1:]:
        other = pts_by_seed[s]
        assert other.height == base.height, "datapoint counts differ across seeds"
        assert other.select(key_cols).equals(base.select(key_cols)), "datapoints differ across seeds"
        assert (other["y"] == base["y"]).all(), "labels differ across seeds"
    mids = base["match"].to_numpy()
    y = base["y"].to_numpy()
    lt = G.logloss(base["p_table"].to_numpy(), y)
    ones = np.ones(len(y))
    per_seed_loss, seed_gdt = {}, {}
    for arm, (col, _) in ARMS.items():
        losses = []
        for s in SEEDS:
            col_s = pts_by_seed[s][col].to_numpy()
            losses.append(G.logloss(col_s, y))
            seed_gdt[(arm, s)] = float((lt - losses[-1]).mean())
        per_seed_loss[arm] = losses
    seed_mean_loss = {arm: np.mean(per_seed_loss[arm], axis=0) for arm in ARMS}
    gdt_mean, gdt_ci = {}, {}
    for arm in ARMS:
        res = G.wboot(mids, lt, seed_mean_loss[arm], ones, REPS, BOOT_SEED)
        gdt_mean[arm], gdt_ci[arm] = res["diff"], res["ci"]
    paired = {}
    for a, b in PAIRS:
        # GDT(x) = mean(table loss - loss_x), so mean(loss_b - loss_a) = GDT(a) - GDT(b): the "a minus b" GDT difference
        res = G.wboot(mids, seed_mean_loss[b], seed_mean_loss[a], ones, REPS, BOOT_SEED)
        per_seed = {s: float((per_seed_loss[b][i] - per_seed_loss[a][i]).mean()) for i, s in enumerate(SEEDS)}
        paired[f"{a} minus {b}"] = {"point_GDT_diff": res["diff"], "ci95": res["ci"], "per_seed_point": per_seed}
    return {"counts": counts[0], "counts_by_seed": counts, "seed_gdt": {f"{a}|{s}": v for (a, s), v in seed_gdt.items()},
            "seed_mean_gdt": gdt_mean, "seed_mean_gdt_ci95": gdt_ci, "paired": paired}


def selection(gdt_mean, gdt_ci):
    top = max(gdt_mean, key=lambda a: gdt_mean[a])
    lo_t, hi_t = gdt_ci[top]
    overlapping = [a for a in ARMS if gdt_ci[a][0] <= hi_t and lo_t <= gdt_ci[a][1]]
    pick = min(overlapping, key=lambda a: ARMS[a][1])
    return {"highest_GDT_arm": top, "highest_GDT_interval": gdt_ci[top], "arms_overlapping_it": overlapping,
            "picked_fewest_inputs": pick, "rule": "fewest inputs among arms whose GDT 95% interval overlaps the highest-GDT arm's"}


def main():
    fits = fit_results()
    wide = B.load_preds()
    print(f"validation rows {wide.height:,}; matches {wide['match_key'].n_unique():,}", flush=True)
    ll = logloss_section(wide, fits)
    gd = gdt_section(fits)
    sel = selection(gd["seed_mean_gdt"], gd["seed_mean_gdt_ci95"])
    out = {"log_loss": ll, "gdt": gd, "selection": sel,
           "fit_results": {f"{a}|{s}": {k: fits[(a, s)][k] for k in ("val_logloss", "best_iteration", "fit_seconds", "n_features",
                                                                    "n_validation", "device", "id")} for (a, s) in fits}}
    with open(f"{SCRATCH}/out/analysis.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
