import glob, os
import polars as pl

BASE = r"G:\pureskill\csds-cs2\tomes\combined_test\tome\csds"

def pages(ch):
    d = os.path.join(BASE, f"basic_{ch}.COMBINED")
    return sorted(glob.glob(os.path.join(d, "dataframe_*")))

def load(ch):
    return pl.concat([pl.read_parquet(p) for p in pages(ch)], how="diagonal_relaxed")

re_ = load("round_end")
re26 = re_.filter(pl.col("match_key").str.starts_with("csds/2026/"))
clean = re26.filter(pl.col("player_count") == 10).select(["match_key", "round"]).unique()
ph = load("player_hurt").filter(pl.col("match_key").str.starts_with("csds/2026/"))
ph = ph.join(clean, on=["match_key", "round"], how="inner").with_columns(
    pl.col("tick").cast(pl.Int64), pl.col("round").cast(pl.Int64), pl.col("player_id").cast(pl.Int64), pl.col("attacker_id").cast(pl.Int64))
ph = ph.sort(["match_key", "round", "player_id", "tick", "health"], descending=[False, False, False, False, True])
ph = ph.with_columns(pl.col("health").shift(1).over(["match_key", "round", "player_id"]).alias("prev_health"))
ph = ph.with_columns(pl.when(pl.col("prev_health").is_null()).then(pl.lit(100)).otherwise(pl.col("prev_health")).alias("hp_seq"))
ph = ph.with_columns((pl.col("hp_seq") == 100).alias("full"), (pl.col("health") == 0).alias("lethal_h0"))
head = ph.filter(pl.col("weapon_name").is_in(["ak47", "m4a1"]) & (pl.col("hit_box_code") == 1)).with_row_index("hid")
lh = head.filter(pl.col("lethal_h0") & pl.col("full")).select(["hid", "match_key", "round", "tick", "player_id", "attacker_id", "weapon_name", "health_removed", "attacker_team_code", "player_team_code"])

pdth = load("player_death").filter(pl.col("match_key").str.starts_with("csds/2026/"))
pdth = pdth.join(clean, on=["match_key", "round"], how="inner").with_columns(
    pl.col("tick").cast(pl.Int64), pl.col("round").cast(pl.Int64), pl.col("player_id").cast(pl.Int64), pl.col("attacker_id").cast(pl.Int64))
mj = lh.join(pdth.select(["match_key", "round", "player_id", "attacker_id", "weapon_name", "tick", "is_headshot"]).rename({"tick": "death_tick", "weapon_name": "death_weapon"}),
             on=["match_key", "round", "player_id", "attacker_id"], how="left")
mj = mj.with_columns((pl.col("death_tick") - pl.col("tick")).alias("dt"))
ok = mj.filter((pl.col("dt") >= 0) & (pl.col("dt") <= 4) & (pl.col("death_weapon") == pl.col("weapon_name")))
okid = ok.select("hid").unique()
unm = lh.join(okid, on="hid", how="anti")
print("full-HP lethal head rows:", lh.height, " unmatched:", unm.height)
# categorize the unmatched by what death row exists for that victim-round
any_death = unm.join(pdth.select(["match_key", "round", "player_id", "tick", "attacker_id", "weapon_name", "is_headshot"]).rename({"tick": "d_tick", "attacker_id": "d_attacker", "weapon_name": "d_weapon"}),
                     on=["match_key", "round", "player_id"], how="left")
any_death = any_death.with_columns((pl.col("d_tick") - pl.col("tick")).abs().alias("absdt"))
best = any_death.sort(["hid", "absdt"]).unique(subset=["hid"], keep="first")
print("unmatched with ANY death row for same victim-round (within 64 ticks):", best.filter(pl.col("absdt") <= 64).height)
print("unmatched with ANY death row for same victim-round (any tick):", best.filter(pl.col("absdt").is_not_null()).height)
print("unmatched with NO death row for victim-round:", best.filter(pl.col("absdt").is_null()).height)
print("unmatched: attacker_team == victim_team (TK) count:", unm.filter(pl.col("attacker_team_code") == pl.col("player_team_code")).height)
print("unmatched: null attacker team:", unm.filter(pl.col("attacker_team_code").is_null()).height)
print("unmatched sample (best matches):")
print(best.head(8).select(["match_key", "round", "tick", "player_id", "attacker_id", "weapon_name", "d_tick", "d_attacker", "d_weapon", "absdt", "is_headshot"]).to_dicts())
print("unmatched by weapon:", unm.group_by("weapon_name").len().to_dicts())
print("matched weapon-name mismatch count (death weapon != hurt weapon, within 4 ticks, same attacker):",
      mj.filter((pl.col("dt") >= 0) & (pl.col("dt") <= 4) & (pl.col("death_weapon") != pl.col("weapon_name"))).height)
print("ALL DONE")
