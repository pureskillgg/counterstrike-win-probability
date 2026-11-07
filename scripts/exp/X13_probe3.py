"""X13 probe 3: match_key format vs header, and win_reason codes vs bomb plant status."""
import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import polars as pl

ROOT = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
DATE = "2026-09-18,2026-10-08"
def chan(ch):
    return pl.scan_parquet(os.path.join(ROOT, f"basic_{ch}.{DATE}", "dataframe_*"))
KEYS = ["match_key", "round"]

re_ = chan("round_end").select(KEYS + ["tick", "winner_team_code", "win_reason_code", "player_count"]).collect()
print("round_end match_key samples:", re_["match_key"].unique().head(3).to_list())
hdr = pl.scan_parquet(os.path.join(ROOT, "basic_round_end." + DATE, "header", "dataframe_*")).select(["match_id", "key", "map_name"]).collect()
print("header match_id samples:", hdr["match_id"].head(3).to_list())
print("header key samples:", hdr["key"].head(3).to_list())
mk = set(re_["match_key"].unique().to_list())
print("overlap match_id:", len(mk & set(hdr["match_id"].to_list())), "overlap key:", len(mk & set(hdr["key"].to_list())))
for m in list(mk)[:1]:
    print("match_key len", len(m), repr(m)[:120])

pl_ = chan("bomb_state").filter(pl.col("event_type") == "bomb_planted").select(KEYS + [pl.col("tick").alias("plant_tick")]).collect()
r = re_.join(pl_, on=KEYS, how="left").with_columns(
    pl.col("plant_tick").is_not_null().alias("planted"),
    (pl.col("plant_tick") < pl.col("tick")).fill_null(False).alias("plant_before_end"))
print("win_reason x planted x plant_before_end (reason, winner, planted, before_end, n):")
print(r.group_by(["win_reason_code", "winner_team_code", "planted", "plant_before_end"]).agg(pl.len().alias("n")).sort(["win_reason_code", "planted", "plant_before_end"]).rows())

# plants at/after round_end: which reasons, and by how many ticks
ex = r.filter(pl.col("planted") & ~pl.col("plant_before_end")).with_columns((pl.col("plant_tick") - pl.col("tick")).alias("delta"))
print("plant at/after end, n:", ex.height, "delta ticks quantiles:", ex["delta"].quantile(0.1), ex["delta"].median(), ex["delta"].quantile(0.9))
print("  reasons:", ex.group_by("win_reason_code").agg(pl.len().alias("n")).sort("win_reason_code").rows())

# round_state round_end tick vs channel round_end tick for these rounds
rs = chan("round_state").filter(pl.col("event_type") == "round_end").select(KEYS + [pl.col("tick").alias("rs_end")]).collect()
j = re_.join(rs, on=KEYS, how="left")
print("round_end channel tick vs round_state round_end tick: equal share", (j.filter(pl.col("rs_end") == pl.col("tick")).height), "of", j.height, "rs_end null", j.filter(pl.col("rs_end").is_null()).height)
