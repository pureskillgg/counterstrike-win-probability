"""D7-1 verifier, part A: header-only checks (no round data).
Recomputes: path-day vs UTC match_date disagreement; build_num by time and by path day; first/last build times.
"""
import glob, sys, datetime as dt
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(200)
ROOT = Path(r"G:\pureskill\csds-cs2")
SCRATCH = Path("C:/Users/<user>/AppData/Local/Temp/claude/C--Users-<user>-Documents-pureskill/e0a18ad2-aa9e-4a3c-9946-29641b2cde82/scratchpad")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
def files(fd): return sorted(glob.glob(str(fd / "dataframe_*")))

H = pl.concat([pl.read_parquet(f, columns=["key", "match_id", "match_date", "build_num", "map_name",
                                           "t_starters_score_final", "ct_starters_score_final", "demo_id", "sharecode"])
               for f in files(BASE / "header")], how="vertical_relaxed")
H = H.with_columns(
    # all match_date values end in Z (UTC); keep first 19 chars (YYYY-MM-DDTHH:MM:SS)
    pl.col("match_date").str.slice(0, 19).str.to_datetime("%Y-%m-%dT%H:%M:%S", strict=True).alias("mdt"),
    pl.col("key").str.extract(r"^csds/(\d{4}/\d\d/\d\d)/", 1).alias("key_day"),
).with_columns(
    pl.col("mdt").dt.strftime("%Y/%m/%d").alias("md_day"),
    pl.col("mdt").dt.strftime("%Y-%m-%d %H:%M:%S").alias("mdt_s"),
)
print("A0 header rows:", H.height, "| key_day null:", H["key_day"].null_count(), "| mdt null:", H["mdt"].null_count())
print("A0 all match_date end in Z:", H["match_date"].str.ends_with("Z").all(), "| all match_id unique:", H["match_id"].n_unique() == H.height)
print("A0 sharecode unique:", H["sharecode"].n_unique(), "| null/empty sharecode:", H.filter(pl.col("sharecode").is_null() | (pl.col("sharecode") == "")).height)

# A1: path day vs UTC match_date day
diff = H.filter(pl.col("md_day") != pl.col("key_day"))
print(f"\nA1 matches where UTC match_date day != key path day: {diff.height} of {H.height}")
print("A1 path-day vs UTC-day offset (days), first 10 rows:", diff.select(
    ((pl.col("mdt").dt.date().cast(pl.Date) - pl.col("key_day").str.replace_all("/", "-").str.to_date("%Y-%m-%d")).dt.total_days()).alias("off")
    )["off"].value_counts().sort("off").rows()[:15])

# A2: build_num by time
print("\nA2 build_num counts:", H["build_num"].value_counts().sort("build_num").rows())
for b in [9842, 9961]:
    s = H.filter(pl.col("build_num") == b)
    print(f"A2 build {b}: n={s.height} | first mdt={s['mdt'].min()} | last mdt={s['mdt'].max()} | key_day range={s['key_day'].min()}..{s['key_day'].max()}")
first9961 = H.filter(pl.col("build_num") == 9961)["mdt"].min()
last9842 = H.filter(pl.col("build_num") == 9842)["mdt"].max()
print("A2 first build-9961 match (UTC):", first9961)
n9842_after = H.filter((pl.col("build_num") == 9842) & (pl.col("mdt") > first9961))
print(f"A2 build-9842 matches dated AFTER first build-9961 match: {n9842_after.height}")
print("A2   their md_day counts:", n9842_after.group_by("md_day").len().sort("md_day").rows())
print("A2   their key_day counts:", n9842_after.group_by("key_day").len().sort("key_day").rows())
n9961_before = H.filter((pl.col("build_num") == 9961) & (pl.col("mdt") < pl.datetime(2024, 2, 7, 1, 0, 0)))
print("A2 build-9961 matches before 2024-02-07 01:00 UTC:", n9961_before.height)

# A3: crosstab of build by UTC day within the window 2024-02-06..2024-02-09, and by key day
W = H.filter((pl.col("mdt") >= pl.datetime(2024, 2, 6)) & (pl.col("mdt") < pl.datetime(2024, 2, 10)))
print("\nA3 build x UTC day (2024-02-06..09):")
print(W.group_by(["md_day", "build_num"]).len().sort(["md_day", "build_num"]).rows())
print("\nA3 build x key path day (all 2024 matches):")
print(H.group_by(["key_day", "build_num"]).len().sort(["key_day", "build_num"]).rows())

# A4: duplicate-looking matches (same UTC start time and map)
dup = H.group_by(["mdt_s", "map_name"]).agg(pl.len().alias("n"), pl.col("key").alias("keys"), pl.col("build_num").alias("builds"), pl.col("sharecode").n_unique().alias("n_share"))
dd = dup.filter(pl.col("n") > 1)
print(f"\nA4 (mdt_s, map) groups with >1 match: {dd.height} groups covering {dd['n'].sum()} matches")
print("A4 sharecode-identical groups:", dd.filter(pl.col("n_share") == 1).height, "| distinct-sharecode groups:", dd.filter(pl.col("n_share") > 1).height)
print("A4 dup groups by builds:", dd.with_columns(pl.col("builds").list.unique().list.len().alias("nb")).group_by("nb").len().rows())
print("A4 dup group first 6 (mdt_s, map, n, n_share, builds):")
print(dd.select(["mdt_s", "map_name", "n", "n_share", "builds"]).head(6))
H.select(["key", "mdt", "md_day", "key_day", "build_num", "map_name", "match_id"]).write_parquet(str(SCRATCH / "verify_D7_1_header.parquet"))
print("\nA done. wrote logs/swarm2/verify_D7_1_header.parquet (own scratch)")
