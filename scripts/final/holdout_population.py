"""Population count for the Oct 26-31 holdout, under the pinned rule. Counts and keys only: no labels, no predictions, no scores."""
import json
import polars as pl

ROOT = "G:/pureskill/csds-cs2"
OCT = f"{ROOT}/data/combined/cache_snapshot_2026_oct23_31.parquet"
TRAIN = f"{ROOT}/data/combined/cache_snapshot_2026.parquet"
POS = f"{ROOT}/data/combined/cache_pos_2026_trainctr.parquet"
MATCHES = f"{ROOT}/data/combined/matches.parquet"
HB = f"{ROOT}/data/combined/holdout_22_25_test_keys.json"

m = pl.scan_parquet(MATCHES).select(["match_key", "upload_day", "in_scope", "dup_keeper"]).unique(subset=["match_key"]).collect()

# Does the cache's own population already use the same filters? (training pool, validation population)
tk = set(pl.scan_parquet(TRAIN).select("match_key").unique().collect()["match_key"].to_list())
tm = m.filter(pl.col("match_key").is_in(list(tk)))
print("training-pool matches:", len(tk), "| in_scope:", int(tm["in_scope"].sum()), "| dup_keeper:", int(tm["dup_keeper"].sum()))

days = [f"2026/10/{d:02d}" for d in range(26, 32)]
H = m.filter(pl.col("upload_day").is_in(days) & pl.col("in_scope") & pl.col("dup_keeper"))
oct_keys = set(pl.scan_parquet(OCT).select("match_key").unique().collect()["match_key"].to_list())
H = H.filter(pl.col("match_key").is_in(list(oct_keys)))
hb = set(json.load(open(HB))["match_keys"])
assert not (set(H["match_key"].to_list()) & hb), "holdout B key in the Oct 26-31 population"
assert not (set(H["match_key"].to_list()) & tk), "training-pool key in the Oct 26-31 population"
print("Oct 26-31 holdout matches (in_scope, dup_keeper, in cache):", H.height)

pos_keys = set(pl.scan_parquet(POS).select("match_key").unique().collect()["match_key"].to_list())
print("of which with positions rows:", len(set(H["match_key"].to_list()) & pos_keys))

rows = (pl.scan_parquet(OCT).select(["match_key", "round", "tick", "label_ct_win"])
        .filter(pl.col("match_key").is_in(H["match_key"].to_list()) & pl.col("label_ct_win").is_not_null())
        .join(pl.scan_parquet(POS).select(["match_key", "round", "tick"]), on=["match_key", "round", "tick"], how="inner")
        .select(pl.len()).collect().item())
print("holdout rows (labelled, with positions row):", rows)
H.select("match_key").write_parquet("C:/Users/<user>/sprint/holdout_oct26_31_keys.parquet")
print("wrote key list")
