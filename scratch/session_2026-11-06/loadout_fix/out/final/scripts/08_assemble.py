import json, sys, os
import numpy as np
import polars as pl
S = "C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/loadout_fix"
OUT = S + "/out/final"
os.makedirs(OUT, exist_ok=True)
mi = pl.read_parquet(S + "/out/match_index.parquet")
keys = json.load(open(S + "/match_keys.json"))
assert mi.height == len(keys)
c1 = pl.read_csv(S + "/out/s3_cov.csv").filter(pl.col("set") == "target")
c2 = pl.read_csv(S + "/out/s3_cov2.csv").filter(pl.col("set") == "target")
cl = pl.read_parquet(S + "/out/space_coverage_local.parquet")  # copied from the space local stage
print("job1 target matches with S3 rows", c1.height, "job2 found", c2.height, "local stage matches", cl["m"].n_unique())
s1 = set(c1["match_key"].to_list())
s2 = set(c2["match_key"].to_list())
assert not (s1 & s2), "overlap between job1 and job2 keys"
loc_m = set(cl["match_key"].to_list())
assert not (loc_m & (s1 | s2)), "local and channel overlap"
status = []
reason = []
for r in mi.iter_rows(named=True):
    mk = r["match_key"]
    if r["split"] == 2:
        status.append("not_built"); reason.append("test_split_coverage_only")
    elif r["date"] >= "2026-10-21":
        status.append("not_built"); reason.append("protected_date_2026-10-21_onward")
    elif mk in s1:
        status.append("s3_channel"); reason.append("bucket_key_channel")
    elif mk in s2:
        status.append("s3_channel"); reason.append("local_covered_key_in_channel")
    elif mk in loc_m:
        status.append("local_1hz"); reason.append("local_1hz_only_not_in_channel")
    else:
        status.append("not_built"); reason.append("in_scope_no_source")
mi = mi.with_columns(pl.Series("build_status", status), pl.Series("reason", reason))
print(mi.group_by(["build_status", "reason"]).agg(pl.len().alias("matches")).sort(["build_status", "reason"]))
print(mi.group_by(["split", "build_status"]).agg(pl.len().alias("matches")).sort(["split", "build_status"]))
print("split x protected-date (counts)")
print(mi.with_columns((pl.col("date") >= "2026-10-21").alias("protected_date")).group_by(["split", "protected_date"]).agg(pl.len().alias("matches")).sort(["split", "protected_date"]))
# per-match coverage columns: S3 coverage for job1/job2, local for local stage
COVC = ["rows", "rounds_with_rows", "steps_with_rows", "freeze_rows"]
cov_s3 = pl.concat([c1.select(["match_key"] + [pl.col(c).cast(pl.Int64) for c in COVC]),
                    c2.select(["match_key"] + [pl.col(c).cast(pl.Int64) for c in COVC])])
cov_loc = cl.select(["match_key"] + [pl.col(c).cast(pl.Int64) for c in COVC])
cov = pl.concat([cov_s3, cov_loc])
mi = mi.join(cov, on="match_key", how="left", suffix="_cov")
mi = mi.select(["m", "match_key", "split", "date", "in_scope", "src", "export_rounds", "build_status", "reason",
                "rounds_with_rows", "steps_with_rows", "freeze_rows", "rows"]).sort("m")
mi.write_csv(OUT + "/match_loadout_status.csv")
built = mi.filter((pl.col("build_status") == "s3_channel") | (pl.col("build_status") == "local_1hz"))
json.dump(built["match_key"].to_list(), open(OUT + "/built_match_keys.json", "w"))
json.dump(keys, open(OUT + "/match_keys.json", "w"))
print("built matches", built.height)
mi.write_parquet(S + "/out/final/match_status.parquet")
