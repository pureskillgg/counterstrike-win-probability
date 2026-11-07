# verify_A2_1 2026 control: header build_num distribution in the 2026 tome, and the same switch-round test
# (score delta vs round_end winner, bomb outcome vs round_end winner) by build for rounds 9, 13, 25, 27, 28.
import glob
import os
import sys

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(30)

BASE = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"


def chan(ch):
    fs = sorted(glob.glob(os.path.join(BASE, f"basic_{ch}.*", "dataframe_*")))
    assert len(fs) >= 1, (ch, fs)
    return pl.concat([pl.scan_parquet(f).select(None) if False else pl.scan_parquet(f) for f in fs], how="vertical_relaxed")


print("2026 header files:", sorted(glob.glob(os.path.join(BASE, "header", "dataframe_*"))), flush=True)
hdr = pl.scan_parquet(os.path.join(BASE, "header", "dataframe_00000")).select(
    ["key", "build_num", "network_protocol", "game_directory", "ppp_version", "map_name", "match_date", "server_name"]).collect()
print("2026 header rows:", hdr.height, "| build_num:", hdr.group_by("build_num").len().sort("build_num").rows(), flush=True)
print("2026 ppp_version:", hdr.group_by("ppp_version").len().rows(), flush=True)
print("2026 network_protocol:", hdr.group_by("network_protocol").len().sort("network_protocol").rows(), flush=True)
print("2026 game_directory (first 6):", hdr.group_by("game_directory").len().sort("len", descending=True).head(6).rows(), flush=True)
print("2026 match_date range:", hdr["match_date"].min(), hdr["match_date"].max(), flush=True)
hdr = hdr.rename({"key": "match_key"})

re_ = chan("round_end").select(["match_key", "round", "tick", "winner_team_code", "win_reason_code"]).collect()
re_u = re_.sort(["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first")
fe = (chan("round_state").select(["match_key", "round", "tick", "event_type", "t_score", "ct_score", "phase"])
      .filter(pl.col("event_type") == "round_freeze_end").collect()
      .sort(["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first"))
bo = (chan("bomb_state").select(["match_key", "round", "event_type"])
      .filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"])).collect())
bsum = bo.group_by(["match_key", "round"]).agg(
    pl.col("event_type").eq("bomb_exploded").any().alias("exp"),
    pl.col("event_type").eq("bomb_defused").any().alias("dfs")).with_columns(
    pl.when(pl.col("exp") & ~pl.col("dfs")).then(2).when(pl.col("dfs") & ~pl.col("exp")).then(3).otherwise(None).alias("bomb_side"))

nx = (fe.sort(["match_key", "round"])
      .with_columns(pl.col("round").shift(-1).over("match_key").alias("r_next"),
                    pl.col("t_score").shift(-1).over("match_key").alias("t_next"),
                    pl.col("ct_score").shift(-1).over("match_key").alias("ct_next"))
      .filter(pl.col("r_next") == pl.col("round") + 1)
      .with_columns((pl.col("t_next") - pl.col("t_score")).alias("dT"),
                    (pl.col("ct_next") - pl.col("ct_score")).alias("dCT"))
      .with_columns(pl.when(pl.col("dT") == 1).then(2).when(pl.col("dCT") == 1).then(3).otherwise(0).alias("w_score"))
      .select(["match_key", "round", "w_score"]))

T = (re_u.join(bsum.select(["match_key", "round", "bomb_side"]), on=["match_key", "round"], how="left")
     .join(nx, on=["match_key", "round"], how="left")
     .join(hdr.select(["match_key", "build_num", "map_name"]), on="match_key", how="left"))
print("2026 round_end rows:", re_u.height, "| with bomb outcome:", T.filter(pl.col("bomb_side").is_not_null()).height, flush=True)
print("2026 winner vs bomb (all rounds, bad/n):", T.filter(pl.col("bomb_side").is_not_null()).select(
    [pl.len().alias("n"), (pl.col("bomb_side") != pl.col("winner_team_code")).sum().alias("bad")]).rows(), flush=True)
print("2026 score vs end (w!=0, bad/n):", T.filter(pl.col("w_score") != 0).select(
    [pl.len().alias("n"), (pl.col("w_score") != pl.col("winner_team_code")).sum().alias("bad")]).rows(), flush=True)
print("2026 by build x k (k in 9,12,13,25,27,28) score bad/n | bomb bad/n:")
for k in [9, 12, 13, 25, 27, 28]:
    s = T.filter(pl.col("round") == k)
    sc = s.filter(pl.col("w_score") != 0)
    bm = s.filter(pl.col("bomb_side").is_not_null())
    print(f"  k={k}: score {int((sc['w_score'] != sc['winner_team_code']).sum())}/{sc.height} | bomb {int((bm['bomb_side'] != bm['winner_team_code']).sum())}/{bm.height}  builds:",
          s.group_by("build_num").len().rows(), flush=True)
print("DONE 2026", flush=True)
