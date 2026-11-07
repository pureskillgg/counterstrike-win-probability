import glob, os
import polars as pl
cands = sorted(glob.glob("E:/csds_final/csds/2026/09/*/*/player_info"))
target = [c for c in cands if os.path.dirname(c).replace("\\", "/").split("/")[-1].startswith("19")]
print("candidates starting with 19:", [os.path.dirname(c).replace("\\", "/").split("/")[-1] for c in target][:5])
# match prefix '19' in the sample ordering used by alive_vs_status_sep.py
sample = cands[::max(1, len(cands) // 40)][:40]
hit = [c for c in sample if os.path.dirname(c).replace("\\", "/").split("/")[-1].startswith("19")]
print("sampled match with prefix 19:", hit)
d = os.path.dirname(hit[0])
pi = pl.read_parquet(os.path.join(d, "player_info"))
ps = pl.read_parquet(os.path.join(d, "player_status"), columns=["tick", "round", "player_id", "player_id_fixed", "health", "is_fake_player"])
pdth = pl.read_parquet(os.path.join(d, "player_death"), columns=["round", "tick", "player_id", "player_team_code"])
print("round 6 roster:", pi.filter(pl.col("round") == 6).select(["team_code", "player_id", "player_id_fixed"]).sort("team_code").to_dicts())
print("round 6 deaths:", pdth.filter(pl.col("round") == 6).sort("tick").to_dicts())
sub = ps.filter((pl.col("round") == 6) & (pl.col("tick") == 65761))
print("status rows at tick 65761 round 6:", sub.to_dicts())
rs = pl.read_parquet(os.path.join(d, "round_state"), columns=["round", "tick", "event_type"])
print("round 6 freeze/end", rs.filter((pl.col("round") == 6) & (pl.col("event_type") == "round_freeze_end"))["tick"].to_list())
