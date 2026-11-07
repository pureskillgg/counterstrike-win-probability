"""D6-4 verifier, step 12: the CT-alive-at-defuse and T-alive-at-explosion tests, run on the RAW E: records
(read-only) for the same affected matches as step 10, plus two controls. Uses the raw player_spawn,
player_death and bomb_state channels. Side coding: planter code = T (2) per match (ASSUMPTION P1 as in step 7).
Prints one line per match.
"""
import sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
RAW = Path(r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02")
SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
x = pl.read_parquet(str(SCR / "verify_D6_4_physical_2024_r13.parquet"))
sel = []
for day in ["12", "13", "15", "17", "20", "24"]:
    s = x.filter(pl.col("aff") & pl.col("match_key").str.contains(f"/2024/02/{day}/")).head(2)
    sel += s.select(["match_key", "aff", "bomb_side", "defuse_tick", "explode_tick", "plant_tick"]).rows()
c = x.filter((~pl.col("aff")) & pl.col("match_key").str.contains("/2024/02/12/")).head(1)
sel += c.select(["match_key", "aff", "bomb_side", "defuse_tick", "explode_tick", "plant_tick"]).rows()
for mk, aff, bomb, dt, et, pt in sel:
    parts = mk.split("/")
    folder = RAW / parts[3] / parts[4]
    try:
        spawn = pl.read_parquet(str(folder / "player_spawn")).filter(pl.col("round") == 13)
        death = pl.read_parquet(str(folder / "player_death")).filter(pl.col("round") == 13)
        bomb = pl.read_parquet(str(folder / "bomb_state")).filter(pl.col("round") == 13)
    except Exception as ex:
        print(parts[3], parts[4][:8], "raw read error", type(ex).__name__, str(ex)[:120])
        continue
    plant = bomb.filter(pl.col("event_type") == "bomb_planted")
    plant_code = plant["player_team_code"][0] if plant.height else None
    tcode = plant_code if plant_code is not None else 2
    ccode = 5 - tcode  # 2 <-> 3
    roster = spawn.group_by("player_id").agg(pl.col("player_team_code").mode().first().alias("side"))
    ct_roster = roster.filter(pl.col("side") == ccode).height
    t_roster = roster.filter(pl.col("side") == tcode).height
    defuse_row = bomb.filter(pl.col("event_type") == "bomb_defused")
    explode_row = bomb.filter(pl.col("event_type") == "bomb_exploded")
    tag = "AFFECTED" if aff else "CONTROL "
    line = f"{tag} {parts[3]} {parts[4][:8]} bomb={bomb_side_name(bomb) if False else ('defused' if defuse_row.height else 'exploded' if explode_row.height else 'none')} planter_code={plant_code} CTroster={ct_roster} Troster={t_roster}"
    if defuse_row.height:
        dtk = defuse_row["tick"][0]
        ct_dead = death.filter((pl.col("player_team_code") == ccode) & (pl.col("tick") < dtk))["player_id"].n_unique()
        line += f" defuser_code={defuse_row['player_team_code'][0]} CT_alive_at_defuse={ct_roster - ct_dead}"
    if explode_row.height:
        etk = explode_row["tick"][0]
        t_dead = death.filter((pl.col("player_team_code") == tcode) & (pl.col("tick") < etk))["player_id"].n_unique()
        line += f" T_alive_at_explode={t_roster - t_dead}"
    print(line)
