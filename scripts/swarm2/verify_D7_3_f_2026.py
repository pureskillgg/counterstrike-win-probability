"""Verify D7-3 (independent): 2026 tome bomb-based winner mismatch by header build_num (10896 vs 10924),
plus the 2026 build-change date. Same definition as the 2024 check (bomb_side 3 if defused else 2; mismatch if != winner).
Reads header, round_end, bomb_state of the 2026 tome only. Prints counts.
"""
import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(240)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
SUF = ".2026-09-18,2026-10-08"
def files(fd): return sorted(glob.glob(str(fd / "dataframe_*")))

H = pl.read_parquet(str(BASE / "header" / "dataframe_00000"), columns=["key", "match_date", "build_num", "map_name"]).rename({"key": "match_key"})
H = H.with_columns(pl.col("match_date").str.replace("Z$", "").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", strict=False).alias("mdt"))
print("2026 build by first/last match_date:")
print(H.group_by("build_num").agg(pl.col("mdt").min().alias("first"), pl.col("mdt").max().alias("last"), pl.len().alias("n")).sort("build_num"))
print("2026 map counts:", H.group_by("map_name").len().sort("len", descending=True).rows())

RE_schema = pl.read_parquet_schema(files(BASE / f"basic_round_end{SUF}")[0])
print("2026 round_end columns:", list(RE_schema.keys()))
RE = pl.concat([pl.scan_parquet(f).select(["match_key", "round", "winner_team_code", "player_count"]) for f in files(BASE / f"basic_round_end{SUF}")]).collect()
print("2026 round_end rows:", RE.height, "| player_count dist:", RE["player_count"].value_counts().sort("player_count").rows())
BS = pl.concat([pl.scan_parquet(f).select(["match_key", "round", "event_type"]) for f in files(BASE / f"basic_bomb_state{SUF}")]).filter(
    pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"])).collect()
ev = (BS.group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs"))
        .with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(3).otherwise(2).alias("bomb_side")))
print("2026 bomb-outcome rounds:", ev.height, "| with both events:", ev.filter(pl.col("evs").list.len() > 1).height)
J = (RE.join(ev, on=["match_key", "round"], how="inner")
       .join(H.select(["match_key", "build_num"]), on="match_key", how="left")
       .with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mm")))
print("2026 all-player-count bomb rounds:", J.height, "| mismatches:", int(J["mm"].sum()))
print("2026 mismatch by build:", J.group_by("build_num").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort("build_num").rows())
print("2026 winner codes in all 20p rounds:", RE.filter(pl.col("player_count") == 20)["winner_team_code"].value_counts().sort("winner_team_code").rows())
print("2026 20p rounds total:", RE.filter(pl.col("player_count") == 20).height)
print("\n2026 mismatch by player_count x build (n, mm):")
print(J.group_by(["player_count", "build_num"]).agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort(["player_count", "build_num"]).rows())
J10 = J.filter(pl.col("player_count") == 10)
print("\n2026 10-player bomb rounds by build (n, mm):", J10.group_by("build_num").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort("build_num").rows())
print("2026 10-player round 13 bomb rows by build (n, mm):", J10.filter(pl.col("round") == 13).group_by("build_num").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort("build_num").rows())
print("2026 winner code dist (10p rounds):", RE.filter(pl.col("player_count") == 10)["winner_team_code"].value_counts().sort("winner_team_code").rows())
