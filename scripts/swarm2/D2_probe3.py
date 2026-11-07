"""D2 probe 3: which round-13 source is right in 2024 when the bomb outcome disagrees with winner_team_code?

For each (match, round) with a valid 5v5 round and a reason-based side:
  - timing: bomb outcome tick inside the round window [round_start tick, round_end tick]?
  - reason code, recorded winner, bomb outcome side, planter side
  - elimination side from player_death: side whose five players are all dead (team membership from player_info)
  - round_state score side (team-tracked, side-attached assumption) from D2_test.py
Output counts for: all 2024 round-13 rows, bomb-mismatch rows, and rows where winner == reason.

Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\D2_probe3.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(40)

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = {"2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
        "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"}


def read(label, ch, cols):
    fd = sorted(glob.glob(str(TOME[label] / f"basic_{ch}.*")))[0]
    return pl.scan_parquet(sorted(glob.glob(fd + "/dataframe_*"))).select(cols).collect()


label = "2024"
base = pl.read_parquet(ROOT / "logs" / "swarm2" / f"D2_rounds_{label}.parquet")
k = base.filter(pl.col("s_reason").is_not_null() & pl.col("sx").is_not_null())

rs_tick = read(label, "round_start", ["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first")
rs_tick = rs_tick.rename({"tick": "tick_start"})
bs = read(label, "bomb_state", ["match_key", "round", "tick", "event_type"])
bo_ev = (bs.filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]))
         .select(["match_key", "round", "tick", "event_type"]))
bo_ev = bo_ev.sort("tick").unique(["match_key", "round"], keep="first", maintain_order=True).rename({"tick": "tick_bomb", "event_type": "bomb_ev"})
pl_ev = bs.filter(pl.col("event_type") == "bomb_planted").sort("tick").unique(["match_key", "round"], keep="first", maintain_order=True).select(
    ["match_key", "round", pl.col("tick").alias("tick_plant")])

k = k.join(rs_tick, on=["match_key", "round"], how="left").join(bo_ev, on=["match_key", "round"], how="left").join(pl_ev, on=["match_key", "round"], how="left")
# round_end tick is not kept in the table; read it
re_ = read(label, "round_end", ["match_key", "round", "tick"]).rename({"tick": "tick_end"})
k = k.join(re_.unique(["match_key", "round"], keep="first"), on=["match_key", "round"], how="left")

k = k.with_columns(
    ((pl.col("tick_bomb") >= pl.col("tick_start")) & (pl.col("tick_bomb") <= pl.col("tick_end"))).alias("bomb_in_window"),
    (pl.col("tick_bomb") > pl.col("tick_end")).alias("bomb_after_end"),
    (pl.col("tick_bomb") < pl.col("tick_start")).alias("bomb_before_start"),
    pl.when(pl.col("bomb_ev") == "bomb_exploded").then(pl.lit(2)).when(pl.col("bomb_ev") == "bomb_defused").then(pl.lit(3)).otherwise(pl.lit(None)).cast(pl.Int64).alias("s_bomb2"),
)

# death-based elimination side. Team membership: X side = sx at the round; victim side = sx if victim in X else 5 - sx.
pi = (read(label, "player_info", ["match_key", "round", "player_id", "team_code"])
      .unique(["match_key", "round", "player_id"], keep="first"))
dth = read(label, "player_death", ["match_key", "round", "tick", "player_id"])
# membership: map each player_id to X/Y via the first valid round of the match (team code 2 at F = X)
vf = base.select(["match_key", "round", "sx", "sx_prev"]).filter(pl.col("sx").is_not_null())
# X membership from round-1-style identity: team_code at the first round where the player is listed in base's first valid round
first_valid = base.filter(pl.col("sx").is_not_null()).group_by("match_key").agg(pl.col("round").min().alias("F"))
pim = pi.join(first_valid, on="match_key", how="inner").filter(pl.col("round") == pl.col("F"))
mem = pim.select(["match_key", "player_id", pl.col("team_code").alias("tc_F")]).unique(["match_key", "player_id"])
mem = mem.with_columns((pl.col("tc_F") == 2).alias("isX"))
d = dth.join(mem.select(["match_key", "player_id", "isX"]), on=["match_key", "player_id"], how="left")
d = d.join(vf, on=["match_key", "round"], how="left")
d = d.with_columns(
    pl.when(pl.col("isX")).then(pl.col("sx")).otherwise(5 - pl.col("sx")).alias("victim_side"))
elim = d.group_by(["match_key", "round"]).agg(
    (pl.col("victim_side") == 2).sum().alias("t_deaths"),
    (pl.col("victim_side") == 3).sum().alias("ct_deaths"),
    pl.col("victim_side").is_null().sum().alias("unk_deaths"))
k = k.join(elim, on=["match_key", "round"], how="left")
k = k.with_columns(
    pl.when((pl.col("ct_deaths") == 5) & (pl.col("t_deaths") < 5)).then(pl.lit(2))
    .when((pl.col("t_deaths") == 5) & (pl.col("ct_deaths") < 5)).then(pl.lit(3))
    .otherwise(pl.lit(None)).cast(pl.Int64).alias("s_elim"))

mm = k.filter(pl.col("round") == 13).filter(pl.col("s_bomb2").is_not_null() & (pl.col("s_bomb2") != pl.col("winner")))
ok = k.filter(pl.col("round") == 13).filter(pl.col("s_bomb2").is_not_null() & (pl.col("s_bomb2") == pl.col("winner")))
print("== round 13 (2024): rows with bomb outcome, split by bomb==winner")
print("   bomb != winner rows:", mm.height, " bomb == winner rows:", ok.height)

print("\n== timing of the bomb outcome relative to round window (round 13, all bomb rows)")
r13 = k.filter(pl.col("round") == 13).filter(pl.col("bomb_ev").is_not_null())
print(r13.group_by(["bomb_in_window", "bomb_after_end", "bomb_before_start", pl.col("s_bomb2") != pl.col("winner")]).len().sort(["len"], descending=True))

print("\n== bomb mismatch rows (round 13): reason code, recorded winner, bomb event, planter side")
print(mm.group_by(["win_reason_code", "winner", "bomb_ev", "planter"]).len().sort("len", descending=True).head(30))

print("\n== round 13 all rows: elimination side (from deaths) vs winner vs reason vs bomb")
r13all = k.filter(pl.col("round") == 13)
print("rows:", r13all.height, " with s_elim:", r13all.filter(pl.col("s_elim").is_not_null()).height)
e13 = r13all.filter(pl.col("s_elim").is_not_null())
print(e13.group_by(["s_elim", "winner", "s_reason"]).len().sort(["s_elim", "winner", "s_reason"]))
print("bomb-mismatch rows with s_elim:")
print(mm.filter(pl.col("s_elim").is_not_null()).group_by(["s_elim", "winner", "s_bomb2"]).len().sort(["s_elim", "winner", "s_bomb2"]))

print("\n== round 12 and 14 sanity: bomb mismatch vs winner in neighbouring rounds (all 2024)")
for rnd in [11, 12, 13, 14, 15]:
    kk = k.filter(pl.col("round") == rnd).filter(pl.col("s_bomb2").is_not_null())
    print(f"  round {rnd}: bomb rows {kk.height}, bomb != winner {kk.filter(pl.col('s_bomb2') != pl.col('winner')).height}")

print("\n== round 13 bomb-mismatch rows: score-based side (k.s_score) agreement")
print(mm.filter(pl.col("s_score").is_not_null()).group_by(["s_score", "s_bomb2", "winner"]).len().sort(["s_score", "s_bomb2", "winner"]))
print("== round 13 rows where bomb agrees with winner: s_score agreement")
print(ok.filter(pl.col("s_score").is_not_null()).group_by(["s_score", "s_bomb2", "winner"]).len().sort(["s_score", "s_bomb2", "winner"]))

print("\n== round 13 score-based winner vs reason by day (team-tracked score winner)")
r13s = k.filter(pl.col("round") == 13).filter(pl.col("s_score").is_not_null())
print(r13s.group_by(["day", "s_score", "s_reason"]).len().sort(["day", "s_score", "s_reason"]).filter(pl.col("day").is_in(["01", "07", "12", "17", "25"])))
