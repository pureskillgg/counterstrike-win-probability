"""Pre-check for the Oct 26-31 holdout. Reads keys, dates, columns and coverage only. No labels, no predictions, no scores.
Prints counts so the protocol can be pinned before any holdout row is built."""
import json
import polars as pl

ROOT = "G:/pureskill/csds-cs2"
OCT = f"{ROOT}/data/combined/cache_snapshot_2026_oct23_31.parquet"
TRAIN = f"{ROOT}/data/combined/cache_snapshot_2026.parquet"
POS = f"{ROOT}/data/combined/cache_pos_2026_trainctr.parquet"
MATCHES = f"{ROOT}/data/combined/matches.parquet"
HB = f"{ROOT}/data/combined/holdout_22_25_test_keys.json"
FROZEN = None  # frozen test keys: path taken from the matrix meta below if needed
meta = json.load(open("C:/Users/<user>/sprint/matrix_trainctr/meta.json"))
base, feats = meta["base"], meta["feats_all"]

oct_cols = pl.scan_parquet(OCT).collect_schema().names()
print("base columns:", len(base), "| missing from oct23_31:", [f for f in base if f not in oct_cols])
print("pos columns missing from trainctr:", [f for f in feats[len(base):] if f not in pl.scan_parquet(POS).collect_schema().names()])

m = pl.scan_parquet(MATCHES).select(["match_key", "upload_day", "in_scope", "era", "map_name", "dup_keeper"]).collect()
print("upload_day sample:", sorted(m["upload_day"].drop_nulls().unique().to_list())[:3], "...", sorted(m["upload_day"].drop_nulls().unique().to_list())[-3:])
hold_days = [f"2026/10/{d:02d}" for d in range(26, 32)]
cand = m.filter(pl.col("upload_day").is_in(hold_days))
print("matches uploaded Oct 26-31 (all):", cand.height, "| in_scope:", int(cand["in_scope"].sum()), "| dup_keeper:", int(cand["dup_keeper"].sum()))
print("upload_day values in window:", sorted(cand["upload_day"].unique().to_list()))

oct_keys = pl.scan_parquet(OCT).select(["match_key"]).unique().collect()["match_key"]
oct_set = set(oct_keys.to_list())
cand_set = set(cand["match_key"].to_list())
in_oct_cache = cand_set & oct_set
print("Oct 26-31 matches present in oct23_31 cache:", len(in_oct_cache), "of", len(cand_set))

train_keys = set(pl.scan_parquet(TRAIN).select(["match_key"]).unique().collect()["match_key"].to_list())
print("overlap of Oct 26-31 keys with cache_snapshot_2026 (training/val/test pool):", len(cand_set & train_keys))

hb = set(json.load(open(HB))["match_keys"])
print("overlap of Oct 26-31 keys with holdout B (22-25 Oct):", len(cand_set & hb), "| holdout B keys inside the 26-31 window:", len(hb & cand_set))

pos_keys = set(pl.scan_parquet(POS).select(["match_key"]).unique().collect()["match_key"].to_list())
print("Oct 26-31 keys with a positions row:", len(in_oct_cache & pos_keys), "of", len(in_oct_cache))
