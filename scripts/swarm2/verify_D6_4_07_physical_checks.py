"""D6-4 verifier, step 7: physical checks that use neither round_end nor the score fields.

Idea: a defuse is performed by a CT-side player who must be alive at the defuse tick.
If the bomb_defused rows in affected round-13 matches come from CT-coded players who were alive,
the 'bomb outcome' used as the truth is physically consistent, and round_end (reason 9 = all CTs
killed) is contradicted by the death events.

Side coding: the bomb planter is always T. Planter team code in this tome is 2 in ~99.9% of rounds
(see logs/swarm2/D6_swap_detect.log section 5). ASSUMPTION P1: player_team_code on bomb/death/spawn
rows is side-attached (T=2, CT=3) inside the tome. The planter check below tests P1 per round.
Timing: tick_rate 64 per second (ASSUMPTION T1, A6 uses the same value).
Prints counts only.
"""
import sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(30)
pl.Config.set_tbl_cols(12)
ROOT = Path(r"G:\pureskill\csds-cs2")
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
base = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
TICK_RATE = 64  # ASSUMPTION T1

def ch(name):
    return [p for p in base.iterdir() if p.name.startswith(f"basic_{name}.")][0] / "dataframe_*"

bs = pl.scan_parquet(str(ch("bomb_state"))).select(["match_key", "round", "tick", "event_type", "player_team_code", "player_id"]).collect()
pdeath = pl.scan_parquet(str(ch("player_death"))).select(["match_key", "round", "tick", "player_id", "player_team_code", "attacker_team_code"]).collect()
pspawn = pl.scan_parquet(str(ch("player_spawn"))).select(["match_key", "round", "tick", "player_id", "player_team_code"]).collect()

j = pl.read_parquet(str(SCR / "verify_D6_4_join_2024.parquet")).filter(pl.col("player_count") == 20)
r13 = j.filter((pl.col("round") == 13) & pl.col("bomb_side").is_not_null()).select(
    ["match_key", "round", "day", "winner_team_code", "bomb_side", "score_side", "dT", "dCT"])
r13 = r13.with_columns(pl.when(pl.col("day").is_in([f"{d:02d}" for d in range(12, 26)])).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("win"),
                       (pl.col("bomb_side") != pl.col("winner_team_code")).alias("aff"))

# planter side per round (should be 2)
planter = (bs.filter(pl.col("event_type") == "bomb_planted").sort("tick")
           .group_by(["match_key", "round"]).agg(pl.col("player_team_code").first().alias("planter_code"),
                                                  pl.col("player_id").first().alias("planter_id"),
                                                  pl.col("tick").first().alias("plant_tick")))
defuse = (bs.filter(pl.col("event_type") == "bomb_defused").sort("tick")
          .group_by(["match_key", "round"]).agg(pl.col("player_team_code").first().alias("defuser_code"),
                                                 pl.col("player_id").first().alias("defuser_id"),
                                                 pl.col("tick").first().alias("defuse_tick")))
explode = (bs.filter(pl.col("event_type") == "bomb_exploded").sort("tick")
           .group_by(["match_key", "round"]).agg(pl.col("tick").first().alias("explode_tick"),
                                                  pl.col("player_team_code").first().alias("explode_code")))
# roster per side in the round (distinct players who spawned, with their side code)
roster = (pspawn.group_by(["match_key", "round", "player_id"]).agg(pl.col("player_team_code").mode().first().alias("side"))
          .group_by(["match_key", "round", "side"]).agg(pl.len().alias("roster")))
roster_ct = roster.filter(pl.col("side") == 3).select(["match_key", "round", pl.col("roster").alias("ct_roster")])
roster_t = roster.filter(pl.col("side") == 2).select(["match_key", "round", pl.col("roster").alias("t_roster")])
# CT victims with death tick
ct_deaths = (pdeath.filter(pl.col("player_team_code") == 3).select(["match_key", "round", "tick", "player_id"]))
ct_dead_by_defuse = (defuse.select(["match_key", "round", "defuse_tick"]).join(ct_deaths, on=["match_key", "round"], how="inner")
                     .filter(pl.col("tick") < pl.col("defuse_tick")).group_by(["match_key", "round"])
                     .agg(pl.col("player_id").n_unique().alias("ct_dead_before_defuse")))
ct_dead_total = ct_deaths.group_by(["match_key", "round"]).agg(pl.col("player_id").n_unique().alias("ct_dead_total"))

x = (r13.join(planter, on=["match_key", "round"], how="left")
        .join(defuse, on=["match_key", "round"], how="left")
        .join(explode, on=["match_key", "round"], how="left")
        .join(roster_ct, on=["match_key", "round"], how="left")
        .join(roster_t, on=["match_key", "round"], how="left")
        .join(ct_dead_by_defuse, on=["match_key", "round"], how="left")
        .join(ct_dead_total, on=["match_key", "round"], how="left"))
x = x.with_columns(
    ((pl.col("explode_tick") - pl.col("plant_tick")) / TICK_RATE).alias("plant_to_explode_s"),
    ((pl.col("defuse_tick") - pl.col("plant_tick")) / TICK_RATE).alias("plant_to_defuse_s"))

def summarize(df, label):
    print(f"\n--- {label}: n={df.height}")
    print("bomb outcome:", df.group_by("bomb_side").len().sort("bomb_side").rows())
    print("round_end winner:", df.group_by("winner_team_code").len().sort("winner_team_code").rows())
    print("planter side (should be 2):", df.group_by("planter_code").len().sort("planter_code", nulls_last=True).rows())
    d = df.filter(pl.col("defuse_tick").is_not_null())
    print(f"defused rounds: {d.height}; defuser side code:", d.group_by("defuser_code").len().sort("defuser_code", nulls_last=True).rows())
    print(f"  CT roster (spawn-based) distribution:", d.group_by("ct_roster").len().sort("ct_roster", nulls_last=True).rows())
    print(f"  CT players dead before defuse (distinct, death events):", d.group_by("ct_dead_before_defuse").len().sort("ct_dead_before_defuse", nulls_last=True).rows())
    print(f"  defused rounds with ALL CT dead before defuse: {(d['ct_dead_before_defuse'].fill_null(0) >= d['ct_roster'].fill_null(5)).sum()}")
    e = df.filter(pl.col("explode_tick").is_not_null())
    print(f"exploded rounds: {e.height}; explode-side code:", e.group_by("explode_code").len().sort("explode_code", nulls_last=True).rows())
    print("plant->explode seconds (median, min, max):",
          e["plant_to_explode_s"].median(), e["plant_to_explode_s"].min(), e["plant_to_explode_s"].max())
    print("plant->defuse seconds (median, min, max):",
          d["plant_to_defuse_s"].median(), d["plant_to_defuse_s"].min(), d["plant_to_defuse_s"].max())
    print("round_end reason-side check: CT roster known", df.filter(pl.col("ct_roster").is_not_null()).height)

aff = x.filter(pl.col("aff") & (pl.col("win") == "d12-25"))
ctl = x.filter((~pl.col("aff")) & (pl.col("win") == "d12-25"))
ctl01 = x.filter((~pl.col("aff")) & (pl.col("win") == "d01-07"))
summarize(aff, "AFFECTED round 13, d12-25")
summarize(ctl, "CONTROL round 13, d12-25 (bomb == round_end)")
summarize(ctl01, "CONTROL round 13, d01-07 (bomb == round_end)")

# the claim's reason-9 rows: round_end says CTs were all killed (T win), bomb says defuse.
r9 = aff.filter(pl.col("defuse_tick").is_not_null())
print("\nAffected defused rounds: round_end winner counts:", r9.group_by("winner_team_code").len().rows())
print("Affected defused rounds: CT alive at defuse (roster - dead_before_defuse) distribution:")
print((r9.with_columns((pl.col("ct_roster").fill_null(5) - pl.col("ct_dead_before_defuse").fill_null(0)).alias("ct_alive_at_defuse"))
       .group_by("ct_alive_at_defuse").len().sort("ct_alive_at_defuse", nulls_last=True)).rows())
x.write_parquet(str(SCR / "verify_D6_4_physical_2024_r13.parquet"))
print("\nwrote per-round physical table to scratchpad")
