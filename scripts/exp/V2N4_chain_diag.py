import glob, os
import numpy as np
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
ph = ph.join(clean, on=["match_key", "round"], how="inner")
ph = ph.with_columns(pl.col("tick").cast(pl.Int64), pl.col("round").cast(pl.Int64), pl.col("player_id").cast(pl.Int64))
ph = ph.with_row_index("rid")
ph = ph.sort(["match_key", "round", "player_id", "tick", "health"], descending=[False, False, False, False, True])
ph = ph.with_columns(
    pl.col("health").shift(1).over(["match_key", "round", "player_id"]).alias("prev_health"),
    pl.col("tick").shift(1).over(["match_key", "round", "player_id"]).alias("prev_tick"),
    pl.col("weapon_name").shift(1).over(["match_key", "round", "player_id"]).alias("prev_weapon"),
    pl.col("attacker_id").shift(1).over(["match_key", "round", "player_id"]).alias("prev_attacker"),
    pl.col("hit_box_code").shift(1).over(["match_key", "round", "player_id"]).alias("prev_hitbox"),
)
ph = ph.filter(pl.col("prev_health").is_not_null())
ph = ph.with_columns(
    (pl.col("health") + pl.col("health_removed")).alias("hp_a"),
    (pl.col("prev_health") - pl.col("health")).alias("true_loss"),
    (pl.col("tick") == pl.col("prev_tick")).alias("same_tick"),
)
nl = ph.filter((pl.col("prev_health") > 0) & (pl.col("health") > 0))
print("nonfirst nonlethal rows:", nl.height)
ok = nl.filter(pl.col("hp_a") == pl.col("prev_health"))
bad = nl.filter(pl.col("hp_a") != pl.col("prev_health"))
print("ok:", ok.height, " bad:", bad.height)
print("bad: same_tick share:", round(float(bad["same_tick"].mean()), 4), " ok same_tick share:", round(float(ok["same_tick"].mean()), 4))
print("bad: diff (prev_health - health - health_removed) value counts top:")
print(bad.with_columns((pl.col("prev_health") - pl.col("hp_a")).alias("gapdiff")).group_by("gapdiff").len().sort("len", descending=True).head(12).to_dicts())
print("bad: sample rows (first 12):")
cols = ["match_key", "round", "player_id", "tick", "prev_tick", "prev_health", "health", "health_removed", "effective_health_removed", "armor", "armor_removed", "weapon_name", "prev_weapon", "attacker_id", "prev_attacker", "hit_box_code", "prev_hitbox"]
print(bad.select(cols).head(12).to_dicts())
print("ok sample rows (first 5):")
print(ok.select(cols).head(5).to_dicts())
print("effective_health_removed == true_loss share (nonlethal, all):", round(float((nl["effective_health_removed"] == nl["true_loss"]).mean()), 4))
print("effective_health_removed == health_removed share (nonlethal, all):", round(float((nl["effective_health_removed"] == nl["health_removed"]).mean()), 4))
print("true_loss == health_removed share (nonlethal, all):", round(float((nl["true_loss"] == nl["health_removed"]).mean()), 4))
print("bad: effective == true_loss share:", round(float((bad["effective_health_removed"] == bad["true_loss"]).mean()), 4))
print("bad: effective == health_removed share:", round(float((bad["effective_health_removed"] == bad["health_removed"]).mean()), 4))
print("bad: effective == true_loss share (ok-true-loss relation):", round(float((bad["true_loss"] - bad["health_removed"]).abs().mean()), 3))
print("bad: |true_loss - health_removed| distribution:", bad.with_columns((pl.col("true_loss") - pl.col("health_removed")).alias("dd")).group_by("dd").len().sort("len", descending=True).head(10).to_dicts())
print("bad: armor_removed > 0 share:", round(float((bad["armor_removed"] > 0).mean()), 4), " ok:", round(float((ok["armor_removed"] > 0).mean()), 4))
print("bad: weapon fraction null/empty:", round(float(bad["weapon_name"].is_null().mean()), 4))
print("bad: hit_box distribution:", bad.group_by("hit_box_code").len().sort("hit_box_code").to_dicts())
print("ALL DONE")
