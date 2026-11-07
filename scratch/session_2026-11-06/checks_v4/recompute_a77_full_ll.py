import json, subprocess
import numpy as np
import polars as pl

SCR = "C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/a77dm"
ABL = "C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/ablation_ll_gdt"
ANALYSIS = SCR + "/out/analysis.json"
KEY = ["match_key", "round", "second"]
SEEDS = [0, 1, 2]
PATHS = {
    "a77": {s: f"{SCR}/out/fits/preds/a77_d4_e0.03_m20_s{s}.parquet" for s in SEEDS},
    "full": {s: f"{ABL}/out/fits/preds/full_d4_e0.03_m20_s{s}.parquet" for s in SEEDS},
}

# 1. key membership only (no label column read yet)
mk = pl.read_parquet(PATHS["a77"][0], columns=KEY)
mk = mk.with_columns(pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int32),
                     pl.col("match_key").str.extract(r"csds/(\d{4}/\d{2}/\d{2})", 1).alias("day"))
day_min, day_max = mk["day"].min(), mk["day"].max()
oct26_31 = [f"2026/10/{d}" for d in range(26, 32)]
in_oct26_31 = mk["day"].is_in(oct26_31)
# holdout B overlap was counted on the space (hb_membership_out.json): 0 of 1,168 validation keys in the 653-key list
hb = set()
in_hb = mk["match_key"].is_in(list(hb)) if hb else pl.Series([False] * mk.height)
print(json.dumps({"prediction_rows_all": mk.height, "matches_all": int(mk["match_key"].n_unique()),
                  "day_min": day_min, "day_max": day_max,
                  "rows_with_key_date_2026_10_26_to_31": int(in_oct26_31.sum()),
                  "rows_with_key_in_holdout_b_list": int(in_hb.sum())}, indent=1))

keep_mask = ~in_oct26_31 & ~in_hb
keep = mk.filter(keep_mask).select(KEY)
print(json.dumps({"kept_rows": keep.height, "kept_matches": int(keep["match_key"].n_unique())}))


# 2. labels and predictions, only for the kept validation rows
def load(path):
    df = pl.read_parquet(path, columns=KEY + ["label_ct_win", "p"])
    df = df.with_columns(pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int32))
    j = keep.join(df, on=KEY, how="inner")
    assert j.height == keep.height, (path, j.height, keep.height)
    assert j.select(KEY).is_duplicated().sum() == 0, path
    return j.sort(KEY)


data = {arm: {s: load(PATHS[arm][s]) for s in SEEDS} for arm in PATHS}
ref = data["a77"][0]
y = ref["label_ct_win"].to_numpy().astype(np.float64)
for arm in PATHS:
    for s in SEEDS:
        d = data[arm][s]
        assert d.height == ref.height
        assert np.array_equal(d.select(KEY).to_numpy(), ref.select(KEY).to_numpy()), (arm, s)
        assert np.array_equal(d["label_ct_win"].to_numpy().astype(np.float64), y), (arm, s)
print(json.dumps({"aligned_rows": int(ref.height), "labels_agree_across_all_6_files": True}))


def ll(p, yy):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-15, 1 - 1e-15)
    return -(yy * np.log(p) + (1 - yy) * np.log(1 - p))


row_ll = {arm: {s: ll(data[arm][s]["p"].to_numpy(), y) for s in SEEDS} for arm in PATHS}
seed_ll = {f"{arm}|{s}": float(row_ll[arm][s].mean()) for arm in PATHS for s in SEEDS}
seed_mean = {arm: float(np.mean([row_ll[arm][s].mean() for s in SEEDS])) for arm in PATHS}
paired_point = seed_mean["a77"] - seed_mean["full"]
per_seed_pair = {str(s): float((row_ll["a77"][s] - row_ll["full"][s]).mean()) for s in SEEDS}

an = json.load(open(ANALYSIS, encoding="utf-8"))["log_loss"]
stored_point = an["paired"]["a77 minus full"]["point"]
stored_per_seed = an["paired"]["a77 minus full"]["per_seed_point"]
out = {"recomputed_seed_ll": seed_ll,
       "stored_seed_ll_max_abs_diff": max(abs(seed_ll[k] - an["seed_ll"][k]) for k in seed_ll),
       "recomputed_paired_point_a77_minus_full": paired_point,
       "stored_paired_point": stored_point,
       "abs_diff_point": abs(paired_point - stored_point),
       "within_1e_6": abs(paired_point - stored_point) < 1e-6,
       "recomputed_per_seed_pair": per_seed_pair,
       "stored_per_seed_pair": stored_per_seed,
       "max_abs_diff_per_seed": max(abs(per_seed_pair[str(s)] - stored_per_seed[str(s)]) for s in SEEDS)}
print(json.dumps(out, indent=1))
