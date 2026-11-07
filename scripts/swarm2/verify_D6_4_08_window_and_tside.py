"""D6-4 verifier, step 8: (a) are the affected round-13 bomb events inside round 13's own time window
(freeze_end(13) .. freeze_end(14)), or do they belong to a neighbour round? (b) T survivors at the explosion
in affected exploded rounds (round_end says CTs killed every T, reason 8).
Also (c) does round 12 or round 14 carry a bomb row with the same tick (duplicate-copy test).
Prints counts only.
"""
import sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
base = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"

def ch(name):
    return [p for p in base.iterdir() if p.name.startswith(f"basic_{name}.")][0] / "dataframe_*"

fe = (pl.scan_parquet(str(ch("round_state"))).select(["match_key", "round", "tick", "event_type"])
      .filter(pl.col("event_type") == "round_freeze_end").select(["match_key", "round", "tick"]).collect()
      .group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("fe_tick")))
fe_next = fe.with_columns((pl.col("round") - 1).alias("round")).rename({"fe_tick": "fe_next_tick"})
bs = (pl.scan_parquet(str(ch("bomb_state"))).select(["match_key", "round", "tick", "event_type", "player_team_code"]).collect())
bomb_rows = bs.filter(pl.col("event_type").is_in(["bomb_planted", "bomb_defused", "bomb_exploded"]))
pdeath = pl.scan_parquet(str(ch("player_death"))).select(["match_key", "round", "tick", "player_id", "player_team_code"]).collect()
pspawn = pl.scan_parquet(str(ch("player_spawn"))).select(["match_key", "round", "tick", "player_id", "player_team_code"]).collect()

x = pl.read_parquet(str(SCR / "verify_D6_4_physical_2024_r13.parquet"))
# (a) window membership of plant/defuse/explode ticks for round 13
w = x.join(fe, on=["match_key", "round"], how="left").join(fe_next.select(["match_key", "round", "fe_next_tick"]), on=["match_key", "round"], how="left")
w = w.with_columns(
    ((pl.col("plant_tick") >= pl.col("fe_tick")) & (pl.col("plant_tick") < pl.col("fe_next_tick"))).alias("plant_in_r13_window"),
    (pl.col("defuse_tick").is_null() | ((pl.col("defuse_tick") >= pl.col("fe_tick")) & (pl.col("defuse_tick") < pl.col("fe_next_tick")))).alias("defuse_in_r13_window"),
    (pl.col("explode_tick").is_null() | ((pl.col("explode_tick") >= pl.col("fe_tick")) & (pl.col("explode_tick") < pl.col("fe_next_tick")))).alias("explode_in_r13_window"))
print("(a) affected d12-25 round 13, bomb events inside round-13 freeze window:")
for label, sub in [("affected", w.filter(pl.col("aff") & (pl.col("win") == "d12-25"))),
                   ("control d12-25", w.filter((~pl.col("aff")) & (pl.col("win") == "d12-25"))),
                   ("control d01-07", w.filter((~pl.col("aff")) & (pl.col("win") == "d01-07")))]:
    print(f"  {label}: n={sub.height} plant_in={sub['plant_in_r13_window'].sum()} defuse_in={sub['defuse_in_r13_window'].sum()} explode_in={sub['explode_in_r13_window'].sum()}")

# (c) duplicate-copy test: does a bomb row in round 12 or 14 of the same match share the same tick?
nb = bomb_rows.select(["match_key", "round", "tick", "event_type"])
aff_keys = w.filter(pl.col("aff") & (pl.col("win") == "d12-25")).select(["match_key"])
for nbr in [12, 14]:
    other = nb.filter(pl.col("round") == nbr).join(aff_keys, on="match_key", how="inner")
    same = (w.filter(pl.col("aff") & (pl.col("win") == "d12-25")).select(["match_key", "plant_tick", "defuse_tick", "explode_tick"])
            .join(other.select(["match_key", "tick"]), on="match_key", how="inner")
            .filter((pl.col("tick") == pl.col("plant_tick")) | (pl.col("tick") == pl.col("defuse_tick")) | (pl.col("tick") == pl.col("explode_tick"))))
    print(f"(c) affected matches: bomb tick in round {nbr} equal to round-13 bomb tick: {same.height}")
print(f"(c) affected matches with a bomb row in round 12: {nb.filter(pl.col('round')==12).join(aff_keys, on='match_key', how='inner').select('match_key').unique().height} of {aff_keys.height}")

# (b) T survivors at explosion, affected exploded rounds (round_end says CTs killed all Ts: reason 8)
t_deaths = pdeath.filter(pl.col("player_team_code") == 2).select(["match_key", "round", "tick", "player_id"])
# roster of Ts in the round (spawn, side code 2)
t_roster = (pspawn.filter(pl.col("player_team_code") == 2).group_by(["match_key", "round"])
            .agg(pl.col("player_id").n_unique().alias("t_roster_n")))
e = w.filter(pl.col("aff") & (pl.col("win") == "d12-25") & pl.col("explode_tick").is_not_null()).select(["match_key", "round", "explode_tick"])
te = (e.join(t_deaths, on=["match_key", "round"], how="left")
        .filter(pl.col("tick").is_null() | (pl.col("tick") < pl.col("explode_tick")))
        .group_by(["match_key", "round", "explode_tick"]).agg(pl.col("player_id").n_unique().alias("t_dead_before_explode")))
te = te.join(t_roster, on=["match_key", "round"], how="left").with_columns(
    (pl.col("t_roster_n") - pl.col("t_dead_before_explode")).alias("t_alive_at_explode"))
print(f"(b) affected exploded rounds: n={te.height}; T alive at explosion (roster - dead):")
print(te.group_by("t_alive_at_explode").len().sort("t_alive_at_explode", nulls_last=True).rows())
print("    T roster distribution:", te.group_by("t_roster_n").len().sort("t_roster_n", nulls_last=True).rows())
