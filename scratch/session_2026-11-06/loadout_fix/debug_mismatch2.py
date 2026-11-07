import glob, os
import polars as pl
cands = sorted(glob.glob("E:/csds_final/csds/2026/09/*/*/player_info"))
sample = cands[::max(1, len(cands) // 40)][:40]
hits = [c for c in sample if "/19/" in c.replace("\\", "/")]
print("sampled matches under 09/19:", [c.replace("\\", "/") for c in hits])
d = os.path.dirname(hits[1])
pi = pl.read_parquet(os.path.join(d, "player_info"))
ps = pl.read_parquet(os.path.join(d, "player_status"), columns=["tick", "round", "player_id", "player_id_fixed", "health", "is_fake_player"])
pdth = pl.read_parquet(os.path.join(d, "player_death"), columns=["round", "tick", "player_id", "player_team_code"])
rs = pl.read_parquet(os.path.join(d, "round_state"), columns=["round", "tick", "event_type"])
re_ = pl.read_parquet(os.path.join(d, "round_end"), columns=["round", "tick"])
print("round 6 roster:", pi.filter(pl.col("round") == 6).select(["team_code", "player_id", "player_id_fixed"]).sort("team_code").to_dicts())
print("round 6 deaths:", pdth.filter(pl.col("round") == 6).sort("tick").to_dicts())
print("round 6 freeze/end", rs.filter((pl.col("round") == 6) & (pl.col("event_type") == "round_freeze_end"))["tick"].to_list(), re_.filter(pl.col("round") == 6)["tick"].to_list())
sub = ps.filter((pl.col("round") == 6) & (pl.col("tick") == 65761))
print("status rows at tick 65761 round 6:", sub.to_dicts())
sub2 = ps.filter((pl.col("round") == 6) & (pl.col("tick") >= 65700) & (pl.col("tick") <= 65830))
print("per-tick distinct player_id in round 6 around tick 65761 (ids):", sub2.group_by("player_id").agg(pl.len().alias("n")).to_dicts())
print("players in status round 6 whole round, with ids:", ps.filter(pl.col("round") == 6)["player_id"].unique().to_list())
