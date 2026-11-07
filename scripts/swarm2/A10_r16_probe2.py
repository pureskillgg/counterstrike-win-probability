"""A10 round-16 probe 2 (read-only, raw E: channels for one match).

Follow-up to A10_r16_probe.py. Checks whether player_id 4 (T) is alive in the engine sense
around the round-16 live start: connect/spawn/hurt/death/status flags, and the round-end
reason (CT win by elimination at tick 99483 with only 4 recorded T deaths in round 16).
"""
import os
import sys

import polars as pl
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")

MATCH_DIR = r"E:\csds_final\csds\2026\10\07\m_7392c5c6d537a57c2e735e98"


def rd(ch, cols=None):
    path = os.path.join(MATCH_DIR, ch)
    if not os.path.exists(path):
        return None
    schema_names = pq.ParquetDataset(path).schema.names
    use = [c for c in (cols or schema_names) if c in schema_names]
    return pl.from_arrow(pq.read_table(path, columns=use))


for ch in ["player_connect", "player_spawn", "player_hurt", "player_info", "player_personal"]:
    df = rd(ch)
    if df is None:
        print(f"{ch}: MISSING")
        continue
    print(f"{ch}: rows={df.height} cols={df.columns}")

pc = rd("player_connect")
if pc is not None:
    cols = [c for c in ["round", "tick", "player_id", "player_id_fixed", "name", "event_type",
                        "address", "bot", "is_bot", "second"] if c in pc.columns]
    sub = pc.filter(pl.col("tick").is_between(80000, 100000)) if "tick" in pc.columns else pc
    print("player_connect ticks 80000-100000:", sub.select(cols).sort("tick").rows())

ps = rd("player_spawn")
if ps is not None and "tick" in ps.columns:
    cols = [c for c in ["round", "tick", "player_id", "player_id_fixed", "team_code", "second"]
            if c in ps.columns]
    sub = ps.filter(pl.col("round").is_between(14, 17))
    print("player_spawn rounds 14-17 rows:", sub.height)
    if "player_id" in ps.columns:
        sub4 = sub.filter(pl.col("player_id") == 4)
        print("player_spawn rows for player_id 4 (rounds 14-17):", sub4.select(cols).sort("tick").rows())
    print("player_spawn rounds 16 rows:", sub.filter(pl.col("round") == 16).select(cols).sort("tick").rows())

ph = rd("player_hurt")
if ph is not None and "tick" in ph.columns:
    cols = [c for c in ["round", "tick", "player_id", "attacker_id", "health", "dmg_health",
                        "weapon_name"] if c in ph.columns]
    sub = ph.filter(pl.col("round") == 16)
    print("player_hurt round 16 rows:", sub.height)
    if "player_id" in sub.columns:
        print("  victims player_id 4 in round 16:", sub.filter(pl.col("player_id") == 4).select(cols).sort("tick").head(10).rows())
        print("  attackers player_id 4 in round 16:",
              sub.filter(pl.col("attacker_id") == 4).select(cols).sort("tick").head(10).rows())

st = rd("player_status", ["round", "tick", "player_id", "player_id_fixed", "health", "armor",
                          "is_fake_player", "player_controller_id", "place_name", "is_in_buy_zone",
                          "equipment_value_calc", "money", "has_c4", "is_spotted", "second"])
p4 = st.filter((pl.col("round") == 16) & (pl.col("player_id") == 4)).sort("tick")
print("\nplayer 4 status round 16: rows", p4.height)
print("  is_fake_player values:", p4["is_fake_player"].value_counts().rows())
print("  controller ids:", p4["player_controller_id"].unique().to_list(),
      " player_id_fixed:", p4["player_id_fixed"].unique().to_list())
print("  first 3 rows:", p4.select(["tick", "health", "armor", "place_name", "is_in_buy_zone",
                                    "money", "equipment_value_calc", "second"]).head(3).rows())
print("  rows at tick 95318..95330:", p4.filter(pl.col("tick").is_between(95318, 95330)).select(
    ["tick", "health", "place_name", "money"]).rows())
print("  rows at round end 99483 +/-64:", p4.filter(pl.col("tick").is_between(99419, 99547)).select(
    ["tick", "health", "place_name"]).rows())

p8 = st.filter((pl.col("round") == 16) & (pl.col("player_id") == 8)).sort("tick")
print("player 8 (CT, alive to end) status round 16 first 2:", p8.select(["tick", "health", "is_fake_player", "place_name"]).head(2).rows())
print("player 8 at round end:", p8.filter(pl.col("tick").is_between(99419, 99547)).select(
    ["tick", "health", "place_name"]).rows())

# other players' is_fake_player in round 16 at the live start
fz_rows = st.filter((pl.col("round") == 16) & (pl.col("tick") == 94774)).select(
    ["player_id", "is_fake_player", "health"]).sort("player_id")
print("fake flags at fz 94774:", fz_rows.rows())

print("\nDONE")
