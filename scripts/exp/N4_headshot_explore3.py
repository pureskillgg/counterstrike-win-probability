import glob
import json
import os

import polars as pl

TOME = "G:/pureskill/csds-cs2/tomes/combined_test/tome/csds"
STATE = "G:/pureskill/csds-cs2/data/state_sample"


def pages(channel):
    folder = os.path.join(TOME, f"basic_{channel}.COMBINED")
    return sorted(glob.glob(os.path.join(folder, "dataframe_*")))


pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_width_chars(220)

# weapons.json structure
with open("G:/pureskill/csds-cs2/notes/eda/weapons.json", encoding="utf-8") as fh:
    wj = json.load(fh)
print("weapons.json top type:", type(wj).__name__, "len:", len(wj))
if isinstance(wj, dict):
    print("keys:", list(wj.keys())[:10])
    sample_items = list(wj.items())[:3]
    print("sample:", sample_items)
else:
    print("first items:", wj[:3])

# players_1hz player_id and tick alignment
p1 = pl.read_parquet(os.path.join(STATE, "players_1hz.parquet"))
print("\nplayer_id value counts:")
print(p1.group_by("player_id").agg(pl.len(), pl.col("team_code").unique().alias("teams")).sort("player_id"))
extra = p1.filter(pl.col("player_id") >= 10)
print("rows with player_id>=10:", extra.height)
print(extra.select(["match_key", "round", "tick", "seconds_since_live", "player_id", "team_code", "health"]).head(8))

rounds = pl.read_parquet(os.path.join(STATE, "rounds.parquet"))
j = p1.join(rounds.select(["match_key", "round", "live_start_tick", "round_end_tick"]), on=["match_key", "round"], how="left")
j = j.with_columns(((pl.col("tick") - pl.col("live_start_tick")) - 64 * pl.col("seconds_since_live")).alias("resid"))
print("\nresid (tick - live_start - 64*sec) value counts (top):")
print(j.group_by("resid").agg(pl.len()).sort("len", descending=True).head(8))
print("rows with null live_start join:", j.filter(pl.col("live_start_tick").is_null()).height)
print("tick ranges vs round_end (rows with tick > round_end_tick):", j.filter(pl.col("tick") > pl.col("round_end_tick")).height)

# players per round
per_round = p1.group_by(["match_key", "round"]).agg(pl.n_unique("player_id").alias("n_players"))
print("\nn distinct player_id per (match, round):")
print(per_round.group_by("n_players").agg(pl.len()).sort("n_players"))

# weapon codes for hurt rows (2026)
ph_cols = ["match_key", "round", "tick", "player_id", "attacker_id", "health", "weapon_name", "health_removed",
           "hit_box_code", "player_team_code", "attacker_team_code", "attacker_weapon_code", "player_weapon_code",
           "player_x_pos", "player_y_pos", "player_z_pos", "attacker_x_pos", "attacker_y_pos", "attacker_z_pos"]
hurt = pl.scan_parquet(pages("player_hurt"), missing_columns="insert").select(ph_cols).collect()
h26 = hurt.filter(pl.col("match_key").str.starts_with("csds/2026/"))
print("\nattacker_weapon_code for rifle names (2026):")
print(h26.filter(pl.col("weapon_name").is_in(["ak47", "m4a1"])).group_by(["weapon_name", "attacker_weapon_code"]).agg(pl.len()).sort(["weapon_name", "len"], descending=[False, True]).head(20))
print("player_weapon_code for rifle names (2026):")
print(h26.filter(pl.col("weapon_name").is_in(["ak47", "m4a1"])).group_by(["weapon_name", "player_weapon_code"]).agg(pl.len()).sort(["weapon_name", "len"], descending=[False, True]).head(20))

# health chain: compare previous-health-derived health_before with spec formula
h = h26.sort(["match_key", "round", "player_id", "tick", "health"], descending=[False, False, False, False, True])
h = h.with_columns(pl.col("health").shift(1).over(["match_key", "round", "player_id"]).alias("prev_health"))
h = h.with_columns(
    (pl.col("health") + pl.col("health_removed")).alias("hb_spec"),
    pl.col("prev_health").fill_null(100).alias("hb_chain"),
)
lethal = h.filter(pl.col("health") == 0)
print("\nlethal hurt rows (2026):", lethal.height)
print("lethal with health_removed > hb_spec (uncapped damage):", lethal.filter(pl.col("health_removed") > pl.col("hb_spec")).height)
print("lethal with hb_chain == 100:", lethal.filter(pl.col("hb_chain") == 100).height)
print("lethal with hb_chain != 100:", lethal.filter(pl.col("hb_chain") != 100).height)
print("lethal health_removed > hb_chain:", lethal.filter(pl.col("health_removed") > pl.col("hb_chain")).height)
print(lethal.filter(pl.col("health_removed") > pl.col("hb_chain")).select(
    ["tick", "player_id", "health", "health_removed", "hb_spec", "hb_chain", "weapon_name", "hit_box_code"]).head(8))
nonl = h.filter(pl.col("health") > 0)
print("non-lethal rows: health == hb_chain - health_removed:", nonl.filter(pl.col("health") == pl.col("hb_chain") - pl.col("health_removed")).height,
      "of", nonl.height)
print("non-lethal rows where prev_health is null:", nonl.filter(pl.col("prev_health").is_null()).height)

# head vs death alignment on lethal head hits
pd_cols = ["match_key", "round", "tick", "player_id", "attacker_id", "weapon_name", "is_headshot"]
pdeath = pl.scan_parquet(pages("player_death"), missing_columns="insert").select(pd_cols).collect()
pd26 = pdeath.filter(pl.col("match_key").str.starts_with("csds/2026/"))
lh = lethal.filter(pl.col("hit_box_code") == 1).select(["match_key", "round", "tick", "player_id", "attacker_id", "weapon_name"])
mm = lh.join(pd26.select(["match_key", "round", "tick", "player_id", "attacker_id", "is_headshot"]),
             on=["match_key", "round", "player_id", "attacker_id"], how="left", suffix="_d")
mm = mm.with_columns((pl.col("tick_d") - pl.col("tick")).alias("dt"))
print("\nlethal head hits:", lh.height, " with exact-tick death match:", mm.filter(pl.col("dt") == 0).height)
print("dt distribution (lethal head hits, same victim+attacker):")
print(mm.group_by("dt").agg(pl.len()).sort("dt").head(12))

# how many lethal hurt rows have a death row for the same victim within 0..3 ticks
dd = lethal.select(["match_key", "round", "tick", "player_id", "attacker_id"]).join(
    pd26.select(["match_key", "round", "tick", "player_id"]).rename({"tick": "tick_d"}),
    on=["match_key", "round", "player_id"], how="left")
dd = dd.with_columns((pl.col("tick_d") - pl.col("tick")).alias("dt"))
print("lethal rows:", lethal.height, " rows with death dt in [0,3]:", dd.filter(pl.col("dt").is_between(0, 3)).unique(subset=["match_key","round","tick","player_id","attacker_id"]).height)
print("dt distribution lethal rows (dt in -5..10):")
print(dd.filter(pl.col("dt").is_between(-5, 10)).group_by("dt").agg(pl.len()).sort("dt"))
