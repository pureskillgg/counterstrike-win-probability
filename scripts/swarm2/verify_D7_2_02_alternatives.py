"""verify D7-2 (02): identify the single match_date error, test build_num as an alternative clock, check the claim's
first-build-9961 / first-lag(-429) identities, and check map and server confounds.
ASSUMPTION: lag = out_tick - tick (bomb-tick minus round-end tick), as in D7_12/D7_14; checked against out_minus_end_tick.
ASSUMPTION: 'switch' = 2024-02-07 01:01:18 UTC (claim's first flip); the build boundary is taken from header build_num only.
"""
import sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(240)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
LOG = ROOT / "logs" / "swarm2"
SWITCH = __import__("datetime").datetime(2024, 2, 7, 1, 1, 18)

S = pl.read_parquet(str(Path("C:/Users/13144/AppData/Local/Temp/claude/C--Users-13144-Documents-pureskill/e0a18ad2-aa9e-4a3c-9946-29641b2cde82/scratchpad") / "verify_D7_2_S.parquet"))
H = (pl.read_parquet(str(BASE / "header" / "dataframe_00000"), columns=["key", "match_date", "build_num", "map_name", "server_name"])
     .rename({"key": "match_key"})
     .with_columns(pl.col("match_date").str.replace("Z$", "").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", strict=True).alias("mdt_h"),
                   pl.col("match_key").str.extract(r"^csds/(\d{4}/\d{2}/\d{2})/", 1).alias("path_ymd")))

print("== A. header build_num across the whole 2024 tome (4,435 matches) ==")
print("build counts:", H.group_by("build_num").len().sort("build_num").rows())
print("build 9961 matches before switch:", H.filter((pl.col("build_num") == 9961) & (pl.col("mdt_h") < SWITCH)).height)
print("build 9842 matches at/after switch:", H.filter((pl.col("build_num") == 9842) & (pl.col("mdt_h") >= SWITCH)).height)
print("first build 9961 match by match_date:")
print(H.filter(pl.col("build_num") == 9961).sort("mdt_h").head(3).select(["match_key", "mdt_h", "path_ymd", "build_num", "map_name"]))
late9842 = H.filter((pl.col("build_num") == 9842) & (pl.col("mdt_h") >= SWITCH)).sort("mdt_h")
print("build 9842 matches at/after switch (all):")
print(late9842.select(["match_key", "mdt_h", "path_ymd", "build_num", "map_name", "server_name"]))
print("\n2x2: time side of switch x build (all 4,435 matches):")
print(H.with_columns((pl.col("mdt_h") >= SWITCH).alias("after")).group_by(["after", "build_num"]).len().sort(["after", "build_num"]).rows())

print("\n== B. the single match_date error on S: which row(s) break the time threshold at 00:24:22 / 01:01:18 ==")
T_LO, T_HI = __import__("datetime").datetime(2024, 2, 7, 0, 24, 22), SWITCH
err_rows = S.filter(((pl.col("t_match") >= T_HI) & (~pl.col("flip"))) | ((pl.col("t_match") <= T_LO) & (pl.col("flip"))))
print("error rows at the claimed match_date threshold:", err_rows.height)
print(err_rows.select(["match_key", "round", "t_match", "t_path", "build_num", "w_ok", "map_name", "winner_team_code", "ledger_side", "bomb_side"]).with_columns(pl.col("match_key").str.slice(0, 50)))
print("S rows with t_match >= switch and ok:", S.filter((pl.col("t_match") >= T_HI) & (~pl.col("flip"))).height)
print("S rows with t_match <= 00:24:22 and flip:", S.filter((pl.col("t_match") <= T_LO) & (pl.col("flip"))).height)
S_wo = S.filter(~((pl.col("build_num") == 9842) & (pl.col("t_match") >= T_HI)))
print("S rows dropping build-9842 at/after switch:", S_wo.height)
print("S_wo flip rows:", S_wo.filter(pl.col("flip")).height, "| S_wo time threshold errors (flip after):",
      S_wo.filter(((pl.col("t_match") >= T_HI) & (~pl.col("flip"))) | ((pl.col("t_match") < T_HI) & (pl.col("flip")))).height)

print("\n== C. lag -429 identity (bomb tick minus round-end tick) ==")
X = pl.read_parquet(str(LOG / "D7_rounds_with_ledger_dp.parquet"), columns=["match_key", "round", "tick", "out_tick", "out_minus_end_tick", "bomb_side", "mdt"])
print("out_tick - tick == out_minus_end_tick on all rows:", X.filter(pl.col("out_tick").is_not_null()).with_columns((pl.col("out_tick") - pl.col("tick")).alias("lagc")).filter(pl.col("lagc") != pl.col("out_minus_end_tick")).height == 0)
B = X.filter(pl.col("bomb_side").is_not_null()).with_columns((pl.col("out_tick") - pl.col("tick")).alias("lag"))
print("bomb rounds with lag == -429:", B.filter(pl.col("lag") == -429).height, "| matches:", B.filter(pl.col("lag") == -429)["match_key"].n_unique())
first429 = B.filter(pl.col("lag") == -429).sort("mdt").head(3).select(["match_key", "round", "mdt"])
print("first lag -429 rows:"); print(first429.with_columns(pl.col("match_key").str.slice(0, 50)))
lag_by_match = B.group_by("match_key").agg((pl.col("lag") == -429).any().alias("has429"))
LM = H.join(lag_by_match, on="match_key", how="inner")
print("build x has lag -429 (matches with bomb rounds):", LM.group_by(["build_num", "has429"]).len().sort(["build_num", "has429"]).rows())
late_keys = [k for k in late9842["match_key"].to_list()]
print("lag -429 flag for build-9842 matches at/after switch:", LM.filter(pl.col("match_key").is_in(late_keys)).select(["match_key", "mdt_h", "has429"]).with_columns(pl.col("match_key").str.slice(0, 50)))

print("\n== D. confound checks inside S: map and server against build ==")
print("S by map x build (n, flip count):")
print(S.group_by(["map_name", "build_num"]).agg(pl.len().alias("n"), pl.col("flip").sum().alias("flips")).sort(["map_name", "build_num"]))
print("S distinct server_name:", S["server_name"].n_unique(), "(full server table omitted; family check is in script 03)")

print("\n== E. path day vs header match_date (storage lag) over all 4,435 matches ==")
H2 = H.with_columns(pl.col("mdt_h").dt.strftime("%Y/%m/%d").alias("mdate_utc"))
H2 = H2.with_columns((pl.col("path_ymd").str.replace_all("/", "-").str.to_date("%Y-%m-%d") - pl.col("mdt_h").dt.date()).dt.total_days().alias("lag_days"))
print("path day == UTC match date:", H2.filter(pl.col("lag_days") == 0).height, "of", H2.height)
print("lag_days distribution:", H2.group_by("lag_days").len().sort("lag_days").rows())
print("lag_days for build 9961 vs 9842:", H2.group_by(["build_num"]).agg(pl.col("lag_days").min().alias("min"), pl.col("lag_days").max().alias("max"), pl.col("lag_days").mean().alias("mean")).sort("build_num").rows())
