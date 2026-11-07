"""D4 extra check: in bomb-ended rounds, does the death record support round_end's winner or the bomb event?

For each full 2024 round (player_count 20) with a bomb outcome (label-based, as in e2_side_check_r13.py):
  - bomb_side from the outcome event (defused -> 3 CT, exploded -> 2 T), mismatch = bomb_side != winner_team_code.
  - last death inside the live window: victim side, and gap = round_end.tick - last death tick.
  - deaths of the losing side after the bomb event (count), and the side of the last victim.
  - predicted winner from the last victim (5 - victim side) vs winner_team_code, for the mismatched and matched
    groups, split by round 13 and other rounds.
ASSUMPTIONS: A7 (player_death.player_team_code = victim side); A10 (last death within 192 ticks of round_end
counts as the round-ending death; otherwise the round is a timer or late-event round).
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\D4_bomb_round_deaths.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(14)

ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
OUTCOMES = ["bomb_defused", "bomb_exploded"]
T, CT = 2, 3
MAX_GAP = 192


def read(ch, cols):
    fd = sorted(BASE.glob(f"basic_{ch}.*"))[0]
    fs = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


re_ = (read("round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"])
       .unique(["match_key", "round"], keep="first").filter(pl.col("player_count") == 20)
       .rename({"tick": "end"}))
live = (read("round_state", ["match_key", "round", "tick", "event_type"])
        .filter(pl.col("event_type") == "round_freeze_end")
        .select(["match_key", "round", pl.col("tick").alias("live")]).unique(["match_key", "round"], keep="first"))
bs = read("bomb_state", ["match_key", "round", "tick", "event_type"])
outs = bs.filter(pl.col("event_type").is_in(OUTCOMES))
g = (outs.group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs"),
                                                pl.col("tick").min().alias("bomb_tick"))
     .join(re_.select(["match_key", "round", "end", "winner_team_code"]), on=["match_key", "round"], how="inner")
     .join(live, on=["match_key", "round"], how="inner")
     .with_columns(
         pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(CT)).otherwise(pl.lit(T)).alias("bomb_side"),
         pl.when(pl.col("round") == 13).then(pl.lit("r13")).otherwise(pl.lit("other")).alias("rbin"),
     )
     .with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mm")))
print(f"bomb-outcome full rounds with live start: {g.height}; mismatches: {int(g['mm'].sum())}")

de = read("player_death", ["match_key", "round", "tick", "player_team_code"]).filter(
    pl.col("player_team_code").is_in([T, CT]))
D = de.join(g.select(["match_key", "round", "live", "end", "bomb_tick"]), on=["match_key", "round"], how="inner")
D = D.filter((pl.col("tick") >= pl.col("live")) & (pl.col("tick") <= pl.col("end")))
last = (D.sort("tick").group_by(["match_key", "round"], maintain_order=True)
        .agg(pl.col("tick").last().alias("last_tick"), pl.col("player_team_code").last().alias("last_victim"),
             pl.col("tick").n_unique().alias("n_deaths_in_window"),
             (pl.col("tick") > pl.col("bomb_tick")).sum().alias("deaths_after_bomb"),
             pl.col("player_team_code").filter(pl.col("tick") > pl.col("bomb_tick")).n_unique().alias("sides_died_after_bomb")))
J = g.join(last, on=["match_key", "round"], how="left").with_columns(
    (pl.col("end") - pl.col("last_tick")).alias("gap_last_death"),
    (5 - pl.col("last_victim")).alias("pred_from_last_victim"),
)
J = J.with_columns(
    pl.when(pl.col("last_tick").is_null()).then(pl.lit("no_death"))
    .when(pl.col("gap_last_death") <= MAX_GAP).then(pl.lit("death_at_end"))
    .otherwise(pl.lit("death_long_before_end")).alias("death_class"),
    (pl.col("pred_from_last_victim") == pl.col("winner_team_code")).alias("pred_agrees_round_end"),
    (pl.col("pred_from_last_victim") == pl.col("bomb_side")).alias("pred_agrees_bomb"),
)
print("\nby group (mm) and round bin: death class, and whether the last-victim prediction matches round_end or the bomb")
print(J.group_by(["rbin", "mm", "death_class"]).agg(
    pl.len().alias("n"),
    pl.col("pred_agrees_round_end").sum().alias("pred_eq_round_end"),
    pl.col("pred_agrees_bomb").sum().alias("pred_eq_bomb_side"),
    pl.col("deaths_after_bomb").median().alias("median_deaths_after_bomb"),
    pl.col("sides_died_after_bomb").median().alias("median_sides_died_after_bomb"),
).sort(["rbin", "mm", "death_class"]))
print("\nmismatched rounds: winner_team_code, bomb side, last victim side (counts)")
print(J.filter(pl.col("mm")).group_by(["rbin", "evs", "winner_team_code", "last_victim"]).len()
      .sort(["rbin", "evs", "winner_team_code", "last_victim"]))
print("\nmismatched r13 rounds: gap_last_death quantiles (ticks)")
mm13 = J.filter(pl.col("mm") & (pl.col("rbin") == "r13"))
print(mm13.select(pl.col("gap_last_death").quantile(0.1).alias("p10"), pl.col("gap_last_death").median().alias("med"),
                  pl.col("gap_last_death").quantile(0.9).alias("p90")))
