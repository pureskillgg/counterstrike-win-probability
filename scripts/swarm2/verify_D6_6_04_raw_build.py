"""verify D6-6, step 04: (a) server / timestamp-format confounds of build_num in the 2024 tome header;
(b) read-only raw E: check: does the raw header carry the same build_num, match_date, server as the tome, and do the
raw parquet writers differ between build 9842 and 9961 matches (parser/export build question)?
Only reads: E: is read-only here. Footer and header reads of a small sample of matches.
Run: "G:/pureskill/csds-cs2/.pixi/envs/default/python.exe" -I -X utf8 scripts/swarm2/verify_D6_6_04_raw_build.py
"""
import glob
import re
import sys
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(220)

ROOT = Path(r"G:\pureskill\csds-cs2")
B24 = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
RAW = Path(r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02")

H = pl.read_parquet(glob.glob(str(B24 / "header" / "dataframe_*"))[0],
                    columns=["key", "match_id", "match_date", "build_num", "server_name", "map_name"])
H = H.with_columns(pl.col("key").str.extract(r"csds/20\d\d/\d\d/(\d\d)/", 1).alias("keyday"))

print("== (a) server_name by build (2024 tome header) ==")
srv = H.with_columns(pl.when(pl.col("server_name").str.contains("FACEIT")).then(pl.lit("FACEIT"))
                     .when(pl.col("server_name").str.contains("Valve")).then(pl.lit("Valve")).otherwise(pl.lit("other"))
                     .alias("srvgrp"))
print(srv.group_by(["build_num", "srvgrp"]).len().sort(["build_num", "srvgrp"]))
print("server_name values not containing 'Valve' or 'FACEIT' (top 10):")
print(H.filter(~pl.col("server_name").str.contains("Valve|FACEIT")).group_by(["build_num", "server_name"]).len().head(10))
print("distinct build_num among FACEIT servers:", srv.filter(pl.col("srvgrp") == "FACEIT")["build_num"].value_counts().rows())

print("\n== (b) match_date timestamp format (nanosecond or non-.000Z) ==")
fmt = H.with_columns(pl.when(pl.col("match_date").str.contains(r"\.000Z$")).then(pl.lit("ms_000Z")).otherwise(pl.lit("other")).alias("fmt"))
print(fmt.group_by(["build_num", "fmt"]).len().sort(["build_num", "fmt"]))
print("non-standard match_date rows:")
print(fmt.filter(pl.col("fmt") == "other").select(["match_date", "build_num", "keyday", "server_name", "map_name"]))

# ---------------------------------------------------------------- raw sample
samp_def = [
    ("9842_d01-07", H.filter((pl.col("build_num") == 9842) & (pl.col("keyday") <= "07")).sample(8, seed=1)),
    ("9842_d12-25", H.filter((pl.col("build_num") == 9842) & (pl.col("keyday") >= "12")).sample(8, seed=2)),
    ("9961_d12", H.filter((pl.col("build_num") == 9961) & (pl.col("keyday") == "12")).sample(8, seed=3)),
    ("9961_d13-25", H.filter((pl.col("build_num") == 9961) & (pl.col("keyday") >= "13")).sample(8, seed=4)),
]
# include the single 9842 match dated on/after 2024-02-07 01:01 (FACEIT outlier), if present
outl = H.filter((pl.col("build_num") == 9842) & (pl.col("match_date") >= "2024-02-07T01:01")).select(
    ["key", "match_id", "match_date", "build_num", "server_name", "map_name", "keyday"])
print("\n9842 matches dated on/after 2024-02-07T01:01 (outliers):", outl.height)
print(outl)

print("\n== raw E: sample: header and writer metadata (read-only) ==")
rows = []
for label, df in samp_def + [("9842_outlier_after_cutoff", outl)]:
    for r in df.iter_rows(named=True):
        mid = r["match_id"]
        mdir = RAW / r["keyday"] / mid
        hd = mdir / "header"
        re_f = mdir / "round_end"
        if not mdir.exists() or not hd.exists():
            rows.append((label, r["keyday"], mid[:8], "no_raw_dir", None, None, None, None, None))
            continue
        # raw channels are single parquet files (no extension) directly under the match folder
        raw_h = pl.read_parquet(str(hd), columns=["build_num", "match_date", "server_name", "map_name"]).row(0)
        md = pq.ParquetFile(str(hd)).metadata
        re_md = pq.ParquetFile(str(re_f)).metadata if re_f.exists() else None
        same = (raw_h[0] == r["build_num"]) and (raw_h[1] == r["match_date"]) and (raw_h[2] == r["server_name"])
        rows.append((label, r["keyday"], mid[:8], "ok", r["build_num"], raw_h[0], same,
                     md.created_by, re_md.created_by if re_md else None))
out = pl.DataFrame(rows, schema=["sample", "keyday", "match", "status", "tome_build", "raw_build", "raw_eq_tome",
                                 "header_created_by", "round_end_created_by"], orient="row")
print(out)
ok = out.filter(pl.col("status") == "ok")
print("\nraw header equal to tome (build, match_date, server) per sample:", ok.group_by(["sample", "raw_eq_tome"]).len().sort("sample").rows())
print("distinct writer metadata (header, round_end) across all sampled raw files:",
      ok.select(["header_created_by", "round_end_created_by"]).unique().rows())
print("status counts:", out.group_by("status").len().rows())
