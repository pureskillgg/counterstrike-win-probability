"""D6-4 verifier, step 9: are player_team_code values internally consistent in affected rounds?
Tests (a) the planter's and defuser's spawn-row code equals the code on their bomb row, (b) a victim's code on
death rows equals the code on their spawn rows in the same round, (c) the defuser is not a death victim before the
defuse tick. Run for affected round 13 (d12-25), control round 13 (d12-25) and control round 13 (d01-07).
Prints counts only.
"""
import sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
base = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"

def ch(name):
    return [p for p in base.iterdir() if p.name.startswith(f"basic_{name}.")][0] / "dataframe_*"

pspawn = pl.scan_parquet(str(ch("player_spawn"))).select(["match_key", "round", "player_id", "player_team_code"]).collect()
spawn_code = (pspawn.group_by(["match_key", "round", "player_id"]).agg(pl.col("player_team_code").mode().first().alias("spawn_code"),
                                                                       pl.col("player_team_code").n_unique().alias("n_codes")))
pdeath = pl.scan_parquet(str(ch("player_death"))).select(["match_key", "round", "tick", "player_id", "player_team_code"]).collect()
bs = pl.scan_parquet(str(ch("bomb_state"))).select(["match_key", "round", "tick", "event_type", "player_id", "player_team_code"]).collect()
x = pl.read_parquet(str(SCR / "verify_D6_4_physical_2024_r13.parquet"))
x = x.select(["match_key", "round", "win", "aff", "planter_id", "planter_code", "defuser_id", "defuser_code", "defuse_tick"])

# (a) bomb-row code vs spawn-row code for the same player, same round
pl_chk = x.join(spawn_code.select(["match_key", "round", "player_id", pl.col("spawn_code").alias("planter_spawn_code")]),
                left_on=["match_key", "round", "planter_id"], right_on=["match_key", "round", "player_id"], how="left")
df_chk = pl_chk.join(spawn_code.select(["match_key", "round", "player_id", pl.col("spawn_code").alias("defuser_spawn_code")]),
                     left_on=["match_key", "round", "defuser_id"], right_on=["match_key", "round", "player_id"], how="left")
print("(a) planter: bomb code == spawn code (n with spawn code)")
for label, sub in [("affected d12-25", df_chk.filter(pl.col("aff") & (pl.col("win") == "d12-25"))),
                   ("control d12-25", df_chk.filter((~pl.col("aff")) & (pl.col("win") == "d12-25"))),
                   ("control d01-07", df_chk.filter((~pl.col("aff")) & (pl.col("win") == "d01-07")))]:
    s = sub.filter(pl.col("planter_spawn_code").is_not_null())
    d = sub.filter(pl.col("defuser_spawn_code").is_not_null() & pl.col("defuse_tick").is_not_null())
    print(f"  {label}: planter agree {(s['planter_code']==s['planter_spawn_code']).sum()}/{s.height};"
          f" defuser agree {(d['defuser_code']==d['defuser_spawn_code']).sum()}/{d.height}")

# (b) victim death-row code vs victim spawn-row code, same round, affected vs control
dv = pdeath.join(spawn_code.select(["match_key", "round", "player_id", "spawn_code"]), on=["match_key", "round", "player_id"], how="left")
dv = dv.join(x.select(["match_key", "round", "win", "aff"]), on=["match_key", "round"], how="inner")
print("(b) death-row code == spawn code (same player, same round), share of deaths with spawn code:")
for label, sub in [("affected d12-25", dv.filter(pl.col("aff") & (pl.col("win") == "d12-25"))),
                   ("control d12-25", dv.filter((~pl.col("aff")) & (pl.col("win") == "d12-25"))),
                   ("control d01-07", dv.filter((~pl.col("aff")) & (pl.col("win") == "d01-07")))]:
    s = sub.filter(pl.col("spawn_code").is_not_null())
    print(f"  {label}: deaths={sub.height} with spawn={s.height} code-agree={(s['player_team_code']==s['spawn_code']).sum()}")

# (c) defuser death before defuse tick
dd = x.filter(pl.col("defuse_tick").is_not_null()).join(pdeath.select(["match_key", "round", "tick", "player_id"]),
                                                       left_on=["match_key", "round", "defuser_id"],
                                                       right_on=["match_key", "round", "player_id"], how="inner")
dd = dd.filter(pl.col("tick") < pl.col("defuse_tick"))
print(f"(c) defuser killed before defuse tick (rows): affected={dd.filter(pl.col('aff')).height} control={dd.filter(~pl.col('aff')).height}")
