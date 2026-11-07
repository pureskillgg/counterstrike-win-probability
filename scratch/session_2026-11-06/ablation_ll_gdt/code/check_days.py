"""Upload-day range of the training cache and of the validation keys in the sprint matrix, from match_key text only.
Does NOT open holdout B, the Oct 26-31 holdout or the frozen test. Used as a date-range proxy for overlap with Oct 26-31."""
import json
import polars as pl

ROOT = "G:/pureskill/csds-cs2"
CACHE = ROOT + "/data/combined/cache_snapshot_2026.parquet"
VAL = ROOT + "/data/sprint/matrix_trainctr/val_keys.parquet"
OUT = "C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/ablation_ll_gdt/out/check_days.json"
DAY_RE = r"csds/(\d{4}/\d{2}/\d{2})"
OCT_LO, OCT_HI = "2026/10/26", "2026/10/31"


def summarise(keys, label):
    df = pl.DataFrame({"match_key": keys}).unique().with_columns(pl.col("match_key").str.extract(DAY_RE, 1).alias("day"))
    n = df.height
    nulls = df["day"].null_count()
    days = df.drop_nulls("day")
    in_oct = days.filter((pl.col("day") >= OCT_LO) & (pl.col("day") <= OCT_HI)).height
    return {"label": label, "matches": n, "null_day": nulls, "min_day": str(days["day"].min()), "max_day": str(days["day"].max()),
            "matches_on_2026_10_26_to_31": in_oct}


cache_keys = pl.scan_parquet(CACHE).select("match_key").unique().collect()["match_key"].to_list()
val_keys = pl.read_parquet(VAL, columns=["match_key"])["match_key"].to_list()
res = {"cache": summarise(cache_keys, "cache_snapshot_2026 (training cache, all matches)"),
       "validation": summarise(val_keys, "sprint matrix validation keys (1,168 matches)")}
res["validation_subset_of_cache"] = len(set(val_keys) - set(cache_keys)) == 0
res["validation_match_rows"] = len(val_keys)
print(json.dumps(res, indent=1))
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1)
