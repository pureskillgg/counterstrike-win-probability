"""A13 audit: match counts, tree counts and state-sample counts vs the docs.

Read-only. Reads tome header parquet pages (projected columns), the saved XGBoost JSON,
and the 1 Hz sample. Writes nothing except stdout (captured to logs/swarm2/A13_counts.log).
"""
import glob
import json
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path("G:/pureskill/csds-cs2")


def header_info(tome_csds):
    files = sorted(glob.glob(str(tome_csds / "header" / "dataframe_*")))
    schema = pl.read_parquet_schema(files[0])
    print(f"  header files: {len(files)}; columns ({len(schema)}): {list(schema)[:12]}")
    cols = [c for c in ["key", "match_id", "match_date", "max_rounds"] if c in schema]
    df = pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")
    out = {"rows": df.height}
    if "key" in df.columns:
        out["distinct_key"] = df["key"].n_unique()
    if "match_id" in df.columns:
        out["distinct_match_id"] = df["match_id"].n_unique()
    return out, df


def basic_folder_count(tome_csds):
    return len(glob.glob(str(tome_csds / "basic_*")))


print("== A13 counts ==")
tomes = {
    "2026 canonical": ROOT / "tomes/2026-10-01_to_08/tome/csds",
    "2024 canonical (2024-02-g)": ROOT / "tomes/2024-02-g/tome/csds",
    "2024 superseded (2024-02)": ROOT / "tomes/2024-02/tome/csds",
}
for name, path in tomes.items():
    print(f"\n[{name}] {path}")
    print(f"  basic_* channel folders: {basic_folder_count(path)}")
    info, df = header_info(path)
    print(f"  header: {info}")
    if "match_date" in df.columns:
        vc = df["match_date"].value_counts().sort("match_date")
        print(f"  match_date distinct: {df['match_date'].n_unique()}; min {df['match_date'].min()}; max {df['match_date'].max()}")
        top = vc.sort("count", descending=True).head(3)
        print(f"  top match_date counts: {top.rows()}")

print("\n== tree counts (XGBoost JSON, no xgboost load) ==")
mdl_path = ROOT / "models/xgb_primary_seed0.json"
doc = json.loads(mdl_path.read_text(encoding="utf-8"))
learner = doc["learner"]
gb = learner["gradient_booster"]
model = gb["model"]
n_trees_param = int(model["gbtree_model_param"]["num_trees"])
n_trees_list = len(model["trees"])
print(f"  gbtree_model_param.num_trees: {n_trees_param}; len(trees): {n_trees_list}")
print(f"  iteration_indptr length: {len(model.get('iteration_indptr', []))}")
print(f"  learner attributes: {learner.get('attributes', {})}")
print(f"  learner feature_names: {learner.get('feature_names')}")
print(f"  learner feature_types: {learner.get('feature_types')}")
mc = learner.get("learner_model_param", {})
print(f"  learner_model_param: {mc}")

meta = json.loads((ROOT / "models/xgb_primary_seed0_meta.json").read_text(encoding="utf-8"))
print(f"  meta best_iteration={meta['best_iteration']} predict_iteration_range={meta['predict_iteration_range']}")
print(f"  meta data: {meta['data']}")
print(f"  meta split counts: train={meta['split']['train_matches']} val={meta['split']['val_matches']} test={meta['split']['test_matches']}")

print("\n== 1 Hz state sample ==")
for name in ["players_1hz.parquet", "rounds.parquet"]:
    p = ROOT / "data/state_sample" / name
    if not p.exists():
        print(f"  missing: {p}")
        continue
    sch = pl.read_parquet_schema(p)
    print(f"  {name}: columns {list(sch)[:15]}")
    key_cols = [c for c in ["match_key", "match_id", "key"] if c in sch]
    if key_cols:
        df = pl.read_parquet(p, columns=key_cols)
        print(f"  {name}: rows {df.height}; distinct {key_cols[0]} {df[key_cols[0]].n_unique()}")
    else:
        print(f"  {name}: no match key column; rows {pl.read_parquet(p).height}")
