"""verify D7-2 (03): is the single build-9842 exception in the 2024-02-12 window explained by its server (FACEIT) rather than
build? Server family is a coarse prefix bucket: FACEIT / Valve / other. ASSUMPTION: server_name prefix is a usable family label.
"""
import sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(200)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
LOG = ROOT / "logs" / "swarm2"
H = pl.read_parquet(str(BASE / "header" / "dataframe_00000"), columns=["key", "match_date", "build_num", "server_name"]) \
    .rename({"key": "match_key"}) \
    .with_columns(pl.col("match_date").str.replace("Z$", "").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", strict=True).alias("mdt_h"),
                  pl.when(pl.col("server_name").str.contains("(?i)faceit")).then(pl.lit("FACEIT"))
                    .when(pl.col("server_name").str.starts_with("Valve")).then(pl.lit("Valve"))
                    .otherwise(pl.lit("other")).alias("family"))
print("distinct server_name:", H["server_name"].n_unique())
print("family x build (all 4,435):", H.group_by(["family", "build_num"]).len().sort(["family", "build_num"]).rows())
print("other-family server_name examples:", H.filter(pl.col("family") == "other")["server_name"].unique().head(8).to_list())
print("FACEIT matches by match_date (first/last):", H.filter(pl.col("family") == "FACEIT").sort("mdt_h").select(["match_key", "mdt_h", "build_num"]).with_columns(pl.col("match_key").str.slice(0, 40)).head(3))
print("FACEIT matches with match_date >= 2024-02-07 01:01:18:", H.filter((pl.col("family") == "FACEIT") & (pl.col("mdt_h") >= __import__("datetime").datetime(2024, 2, 7, 1, 1, 18))).height)
S = pl.read_parquet(str(Path("C:/Users/13144/AppData/Local/Temp/claude/C--Users-13144-Documents-pureskill/e0a18ad2-aa9e-4a3c-9946-29641b2cde82/scratchpad") / "verify_D7_2_S.parquet")).select(["match_key", "round", "t_match", "build_num", "flip"]) \
    .join(H.select(["match_key", "family"]), on="match_key", how="left")
print("\nS (3,287 matches) family x build x flip:")
print(S.group_by(["family", "build_num", "flip"]).len().sort(["family", "build_num", "flip"]).rows())
