"""Paired analysis for the A77 and A77+DM arms against the existing full (83) arm: validation log loss (1,168 matches) and GDT
(297-match population). Adapted from scratchpad/ablation_ll_gdt/code/analyse.py. Changes: the three arms and the three pairs the
task names, the fit results of a77 and a77dm read from this folder and full from the ablation folder, GDT_sit counts per arm,
and no selection rule (the intervals are reported, nothing is picked).

Match-cluster bootstrap, 1,000 resamples, seed 0: log loss by boot_mean (ratio of sums over resampled matches, as in analyse.py),
GDT by score_gdt.wboot (the same resamples for every call). Writes analysis.json under the scratch folder only.
"""
import json
import os
import sys

import numpy as np
import polars as pl

SCRATCH = "C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/a77dm"
ABLATION = "C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/ablation_ll_gdt"
GDT_DIR = "C:/Users/<user>/Documents/pureskill/counterstrike-win-probability/evaluation/intern_gdt"
TABLE = GDT_DIR + "/table.csv"
sys.path.insert(0, SCRATCH)
sys.path.insert(0, GDT_DIR)
import build_gdt_rows_a77dm as B  # noqa: E402
import score_gdt as G  # noqa: E402

ARMS = {"a77": ("p_a77", 77), "a77dm": ("p_a77dm", 78), "full": ("p_full", 83)}
PAIRS = [("a77", "full"), ("a77dm", "a77"), ("a77dm", "full")]  # the three pairs the task names
SEEDS = [0, 1, 2]
MODELS = ["p_a77", "p_a77dm", "p_full"]
REPS, BOOT_SEED = 1000, 0


def ll_worker(p, y):
    """The sprint worker's log loss (sprint_worker.fit_cell_prepared), row by row."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return -(y * np.log(np.clip(p, 1e-15, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-15, 1)))


def boot_mean(per_match_sum, per_match_n, reps, seed):
    """Match-cluster bootstrap of a row-weighted mean, as analyse.py does it: resample matches, ratio of sums."""
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
            folder = ABLATION if arm == "full" else SCRATCH
            path = f"{folder}/out/fits/results/{arm}_d4_e0.03_m20_s{s}.json"
            out[(arm, s)] = json.load(open(path, encoding="utf-8"))
    return out


def logloss_section(wide):
    y = wide["label_ct_win"].to_numpy()
    matches = wide["match_key"].to_numpy()
    uniq, inv = np.unique(matches, return_inverse=True)
    n_rows = np.bincount(inv).astype(np.float64)
    n_matches = len(uniq)
    rows_loss, seed_ll = {}, {}
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


def gdt_section():
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
                     f"situations_with_{G.SIT_MIN}_or_more": int(pts.filter(pl.col("n_s") >= G.SIT_MIN).select(G.CELL).n_unique())}
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
    return {"counts": counts[0], "counts_by_seed": counts,
            "seed_gdt": {f"{a}|{s}": v for (a, s), v in seed_gdt.items()},
            "seed_mean_gdt": gdt_mean, "seed_mean_gdt_ci95": gdt_ci, "paired": paired,
            "gdt_sit_counts_note": "the datapoints and their situations depend on the rows and the table, not on any model, so the "
                                   "situation counts below hold for every arm (checked: counts_by_seed agree across the seeds)"}


def main():
    fits = fit_results()
    wide = B.load_preds()
    print(f"validation rows {wide.height:,}; matches {wide['match_key'].n_unique():,}", flush=True)
    ll = logloss_section(wide)
    gd = gdt_section()
    out = {"log_loss": ll, "gdt": gd,
           "fit_results": {f"{a}|{s}": {k: fits[(a, s)][k] for k in ("val_logloss", "best_iteration", "fit_seconds", "n_features",
                                                                    "n_validation", "device", "id")} for (a, s) in fits},
           "settings": {"reps": REPS, "boot_seed": BOOT_SEED, "pairs": [f"{a} minus {b}" for a, b in PAIRS],
                        "selection_rule": "none: intervals reported only"}}
    out_path = f"{SCRATCH}/out/analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1), flush=True)
    print(f"written {out_path}", flush=True)


if __name__ == "__main__":
    main()
