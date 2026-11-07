"""Verify D7-3 (independent): do other 2024 header fields (game_directory, network_protocol, platform, providence,
ppp_version, rushb_version) separate the same matches as build_num? Cross-tab by build, and the bomb-anchored round-13
mismatch by each field value. Reads the 2024 header and round_end/bomb_state (projected). No files written.
"""
import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(240)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
def files(fd): return sorted(glob.glob(str(fd / "dataframe_*")))
H = pl.read_parquet(str(BASE / "header" / "dataframe_00000"), columns=["key", "match_date", "build_num", "game_directory", "network_protocol", "platform", "providence", "ppp_version", "rushb_version", "map_name"]).rename({"key": "match_key"})
H = H.with_columns(pl.col("match_date").str.replace("Z$", "").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", strict=False).alias("mdt"))
for c in ["game_directory", "network_protocol", "platform", "providence", "ppp_version", "rushb_version"]:
    print(f"\n== {c} x build_num (matches) ==")
    print(H.group_by([c, "build_num"]).len().sort([c, "build_num"]).rows())
print("\n== game_directory values with first/last mdt ==")
print(H.group_by("game_directory").agg(pl.col("mdt").min().alias("first"), pl.col("mdt").max().alias("last"), pl.len().alias("n"), pl.col("build_num").n_unique().alias("n_builds")).sort("first").with_columns(pl.col("game_directory").str.slice(0, 60)).rows())
print("\n9842 matches dated >= 2024-02-07 00:30 (match_key, game_directory, network_protocol, platform, providence, ppp_version):")
print(H.filter((pl.col("build_num") == 9842) & (pl.col("mdt") >= pl.datetime(2024, 2, 7, 0, 30))).select(["mdt", "game_directory", "network_protocol", "platform", "providence", "ppp_version", "rushb_version"]).with_columns(pl.col("mdt").cast(pl.Utf8), pl.col("game_directory").str.slice(0, 60)).rows())
print("\n9961 matches: game_directory x network_protocol x platform x ppp_version counts:")
print(H.filter(pl.col("build_num") == 9961).group_by(["game_directory", "network_protocol", "platform", "providence", "ppp_version"]).len().sort("len", descending=True).with_columns(pl.col("game_directory").str.slice(0, 60)).rows())

# bomb-anchored round-13 mismatch by game_directory x build
RE = pl.concat([pl.scan_parquet(f).select(["match_key", "round", "winner_team_code", "player_count"]) for f in files(BASE / f"basic_round_end{SUF}")]).filter((pl.col("round") == 13) & (pl.col("player_count") == 20)).collect()
BS = pl.concat([pl.scan_parquet(f).select(["match_key", "round", "event_type"]) for f in files(BASE / f"basic_bomb_state{SUF}")]).filter(
    pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]) & (pl.col("round") == 13)).collect()
ev = BS.group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")).with_columns(
    pl.when(pl.col("evs").list.contains("bomb_defused")).then(3).otherwise(2).alias("bomb_side"))
J = RE.join(ev, on=["match_key", "round"], how="inner").join(H.select(["match_key", "build_num", "game_directory", "mdt"]), on="match_key", how="left")
J = J.with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mm"), pl.col("game_directory").str.slice(0, 50).alias("gd"))
print("\n== round 13 bomb rows: mismatch by game_directory x build (n, mm) ==")
print(J.group_by(["gd", "build_num"]).agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort(["gd", "build_num"]).rows())
