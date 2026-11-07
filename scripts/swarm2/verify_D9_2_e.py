"""verify_D9_2_e: which round indices show a recorded-winner inversion, using bomb-free 5v5 rounds only.

Population: pc20 rounds with NO bomb_planted and NO bomb outcome, and exactly 5 T and 5 CT players in
player_info for that round (ASSUMPTION: 5v5 from player_info distinct players per side).
Decisive round: exactly one side fully dead at round end (CT_all_dead xor T_all_dead).
  consistent: recorded winner is the side that was NOT wiped out (T=2 wins iff CT_all_dead).
  inverted:   recorded winner is the side that WAS wiped out.
No bomb event is used for this test, so it cannot be affected by bomb-event attribution.
Output: per build and round index, decisive / consistent / inverted counts.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(20)

ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"


def files_of(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def scan(name, cols):
    fd = BASE / "header" if name == "header" else BASE / f"basic_{name}{SUF}"
    lf = pl.scan_parquet(files_of(fd))
    have = set(lf.collect_schema().names())
    print(f"[load] {name}: missing={[c for c in cols if c not in have]}")
    return lf.select([c for c in cols if c in have]).collect()


H = scan("header", ["key", "build_num"]).with_columns(pl.col("key").alias("match_key"))
RE = scan("round_end", ["match_key", "round", "winner_team_code", "player_count"]).group_by(
    ["match_key", "round"]).agg(pl.col("winner_team_code").first().alias("winner"),
                                pl.col("player_count").first().alias("pc"))
BS = scan("bomb_state", ["match_key", "round", "event_type"])
EV = BS.filter(pl.col("event_type").is_in(["bomb_planted", "bomb_defused", "bomb_exploded"])).group_by(
    ["match_key", "round"]).agg(pl.len().alias("n_ev"))
DE = scan("player_death", ["match_key", "round", "player_id", "player_team_code"]).unique(
    subset=["match_key", "round", "player_id"], keep="first")
DEAD = DE.group_by(["match_key", "round"]).agg((pl.col("player_team_code") == 2).sum().alias("T_dead"),
                                                (pl.col("player_team_code") == 3).sum().alias("CT_dead"))
PI = scan("player_info", ["match_key", "round", "player_id", "team_code"]).unique(
    subset=["match_key", "round", "player_id"], keep="last")
SIDES = PI.group_by(["match_key", "round"]).agg((pl.col("team_code") == 2).sum().alias("n_T"),
                                                 (pl.col("team_code") == 3).sum().alias("n_CT"))

X = (RE.join(EV, on=["match_key", "round"], how="left")
       .join(DEAD, on=["match_key", "round"], how="left")
       .join(SIDES, on=["match_key", "round"], how="left")
       .join(H.select(["match_key", "build_num"]), on="match_key", how="left")
       .with_columns(pl.col("n_ev").fill_null(0), pl.col("T_dead").fill_null(0),
                     pl.col("CT_dead").fill_null(0), pl.col("n_T").fill_null(0), pl.col("n_CT").fill_null(0))
       .filter((pl.col("pc") == 20) & (pl.col("n_ev") == 0) & (pl.col("n_T") == 5) & (pl.col("n_CT") == 5)))
X = X.with_columns(
    ((pl.col("CT_dead") >= 5) & (pl.col("T_dead") < 5)).alias("ct_wiped"),
    ((pl.col("T_dead") >= 5) & (pl.col("CT_dead") < 5)).alias("t_wiped"),
)
X = X.with_columns(
    (((pl.col("winner") == 2) & pl.col("ct_wiped")) | ((pl.col("winner") == 3) & pl.col("t_wiped"))).alias("consistent"),
    (((pl.col("winner") == 2) & pl.col("t_wiped")) | ((pl.col("winner") == 3) & pl.col("ct_wiped"))).alias("inverted"),
).with_columns((pl.col("consistent") | pl.col("inverted")).alias("decisive"))

print("\n== bomb-free pc20 5v5 rounds: total by build")
print(X.group_by("build_num").agg(pl.len().alias("n"), pl.col("decisive").sum().alias("decisive")).sort("build_num"))

T = (X.filter(pl.col("decisive"))
       .group_by(["build_num", "round"])
       .agg(pl.len().alias("decisive"), pl.col("consistent").sum().alias("consistent"),
            pl.col("inverted").sum().alias("inverted"))
       .sort(["build_num", "round"]))
print("\n== decisive bomb-free 5v5 rounds per round index (build 9842 and 9961)")
print(T.filter(pl.col("round") <= 30))

print("\n== totals over round indices 1-30, by build")
print(T.filter(pl.col("round") <= 30).group_by("build_num").agg(
    pl.col("decisive").sum().alias("decisive"), pl.col("consistent").sum().alias("consistent"),
    pl.col("inverted").sum().alias("inverted")).sort("build_num"))
print("\n== 9961 rounds with ANY inversion, and inverted share, for rounds 1-30")
w = T.filter((pl.col("build_num") == 9961) & (pl.col("round") <= 30)).with_columns(
    (pl.col("inverted") / pl.col("decisive")).round(3).alias("inv_share"))
print(w.filter(pl.col("inverted") > 0))
print("\nDONE part e")
