"""Validation-only checks: row keys, population, and overlap with holdout B, the Oct 26-31 holdout and the frozen test.
Holdout and test files are read for KEY COLUMNS ONLY (no features, labels, predictions or scores). Prints counts, no key values."""
import json, os, collections
import polars as pl

OUT = "C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/gdt57v83"
VAL = "G:/pureskill/csds-cs2/data/sprint/matrix_trainctr/val_keys.parquet"
PRED83 = "G:/pureskill/csds-cs2/data/sprint/run/gpu_fast/preds/full_d4_e0.03_m20_s0.parquet"
RES83 = "G:/pureskill/csds-cs2/data/sprint/run/gpu_fast/results/full_d4_e0.03_m20_s0.json"
HB = "G:/pureskill/csds-cs2/data/combined/holdout_22_25_test_keys.json"
FZ = "G:/pureskill/csds-cs2/data/benchmark/frozen_test_keys.json"
OCT_S = "C:/Users/13144/sprint/holdout_oct26_31_keys.parquet"
OCT_B = "C:/Users/13144/sprint/holdout_oct26_31/keys.parquet"
MAN = "G:/pureskill/csds-cs2/data/combined/split_manifest.json"
EPS_WORKER = 1e-15
res = {}

V = pl.read_parquet(VAL)
assert V.columns == ["match_key", "round", "second", "label_ct_win"], V.columns
res["val_keys"] = {"path": VAL, "columns": V.columns, "rows": V.height, "matches": V["match_key"].n_unique(),
                   "round_min": int(V["round"].min()), "round_max": int(V["round"].max()),
                   "second_min": int(V["second"].min()), "second_max": int(V["second"].max()),
                   "duplicate_row_keys": int(V.select(["match_key", "round", "second"]).is_duplicated().sum()),
                   "rows_per_match_min": int(V.group_by("match_key").len()["len"].min()),
                   "rows_per_match_max": int(V.group_by("match_key").len()["len"].max()),
                   "label_mean": float(V["label_ct_win"].mean())}
val_matches = set(V["match_key"].unique().to_list())
print("val_keys", res["val_keys"], flush=True)

P = pl.read_parquet(PRED83)
assert P.columns == ["match_key", "round", "second", "label_ct_win", "p"], P.columns
assert P.height == V.height
aligned = bool(P.select(["match_key", "round", "second", "label_ct_win"]).equals(V))
p = P["p"].cast(pl.Float64).to_numpy()
y = P["label_ct_win"].cast(pl.Float64).to_numpy()
import numpy as np
pc = np.clip(p, EPS_WORKER, 1 - EPS_WORKER)
ll = float(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean())
rec = json.load(open(RES83, encoding="utf-8"))
res["pred83"] = {"path": PRED83, "columns": P.columns, "rows": P.height, "matches": P["match_key"].n_unique(),
                 "keys_aligned_row_by_row_with_val_keys": aligned, "p_nan": int(np.isnan(p).sum()),
                 "p_min": float(p.min()), "p_max": float(p.max()),
                 "validation_logloss_recomputed": ll, "validation_logloss_recorded_in_result_json": rec["val_logloss"],
                 "result_json_n_validation": rec["n_validation"], "result_json_best_iteration": rec["best_iteration"],
                 "result_json_status": rec["status"]}
print("pred83", res["pred83"], flush=True)

hb = json.load(open(HB, encoding="utf-8"))
hb_keys = set(hb["match_keys"])
assert len(hb["match_keys"]) == len(hb_keys)
fz_list = json.load(open(FZ, encoding="utf-8"))
fz_keys = set(fz_list)
assert len(fz_list) == len(fz_keys)
oct_s = pl.read_parquet(OCT_S, columns=["match_key"])
oct_s_keys = set(oct_s["match_key"].unique().to_list())
oct_b = pl.read_parquet(OCT_B, columns=["match_key", "round", "second"])
oct_b_match = set(oct_b["match_key"].unique().to_list())
oct_b_rows = oct_b.unique(["match_key", "round", "second"])
print("key-file sizes: holdoutB matches", len(hb_keys), "frozen matches", len(fz_keys), "oct small rows", oct_s.height,
      "oct small matches", len(oct_s_keys), "oct big rows", oct_b.height, "oct big matches", len(oct_b_match), flush=True)
res["key_sets"] = {"holdout_B_matches": len(hb_keys), "frozen_test_matches": len(fz_keys),
                   "oct26_31_small_rows": oct_s.height, "oct26_31_small_matches": len(oct_s_keys),
                   "oct26_31_space_rows": oct_b.height, "oct26_31_space_matches": len(oct_b_match)}

ov = {
    "val_matches_in_holdout_B": len(val_matches & hb_keys),
    "val_matches_in_frozen_test": len(val_matches & fz_keys),
    "val_matches_in_oct26_31_small": len(val_matches & oct_s_keys),
    "val_matches_in_oct26_31_space": len(val_matches & oct_b_match),
    "val_rows_match_key_in_holdout_B": int(V.filter(pl.col("match_key").is_in(list(hb_keys))).height),
    "val_rows_match_key_in_frozen_test": int(V.filter(pl.col("match_key").is_in(list(fz_keys))).height),
    "val_rows_match_key_in_oct26_31_small": int(V.filter(pl.col("match_key").is_in(list(oct_s_keys))).height),
    "val_rows_match_key_in_oct26_31_space": int(V.filter(pl.col("match_key").is_in(list(oct_b_match))).height),
    "val_rows_exact_row_key_in_oct26_31_space": int(V.join(oct_b_rows, on=["match_key", "round", "second"], how="semi").height),
}
res["overlap_counts_val_matrix"] = ov
print("overlap", ov, flush=True)

man = json.load(open(MAN, encoding="utf-8"))["assign"]
tag_all = collections.Counter(man.values())
res["manifest_tag_counts_all_keys"] = dict(tag_all)
res["manifest_tags_for_val_matches"] = dict(collections.Counter(man.get(k, "<absent>") for k in val_matches))
res["manifest_val_tag_total"] = tag_all.get("val", 0)
res["manifest_val_set_equals_matrix_val"] = bool({k for k, v in man.items() if v == "val"} == val_matches)
res["manifest_tags_for_holdout_B_keys"] = dict(collections.Counter(man.get(k, "<absent>") for k in hb_keys))
res["manifest_tags_for_frozen_keys"] = dict(collections.Counter(man.get(k, "<absent>") for k in fz_keys))
res["manifest_tags_for_oct26_31_small_keys"] = dict(collections.Counter(man.get(k, "<absent>") for k in oct_s_keys))
res["manifest_tags_for_oct26_31_space_keys"] = dict(collections.Counter(man.get(k, "<absent>") for k in oct_b_match))
print("manifest", {k: res[k] for k in res if k.startswith("manifest")}, flush=True)

# sample of one validation key only, to see whether dates are encoded in the key string
res["sample_val_match_key"] = sorted(val_matches)[0]
json.dump(res, open(os.path.join(OUT, "overlap_checks.json"), "w", encoding="utf-8"), indent=2)
print("wrote", os.path.join(OUT, "overlap_checks.json"), flush=True)
