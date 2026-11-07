import polars as pl
import glob, os

RAW = "/home/sagemaker-user/adam/builder/raw_cache_bucket/"
KEYS = [l.strip() for l in open("/home/sagemaker-user/adam/builder/fixes/freeze_affected_matches.txt") if l.strip()]
print("keys", len(KEYS), KEYS[:2])

def match_dir(key):
    # key looks like csds/YYYY/MM/DD/<uuid>/csds ; the match folder is its dirname
    return RAW + os.path.dirname(key) + "/"

d = match_dir(KEYS[0])
print("dir", d)

rs = pl.read_parquet(d + "round_state")
print("round_state event_type values:", rs["event_type"].unique().to_list())
print(rs.filter(pl.col("round") == 2).select(["round", "tick", "event_type", "phase", "second"]))

ps = pl.read_parquet(d + "player_status", columns=["tick", "round", "player_id", "player_controller_id", "second"])
pv = pl.read_parquet(d + "player_vector", columns=["tick", "round", "player_id", "team_code"])
print("status rows", ps.height, "vector rows", pv.height)

# how many controllers share player_id 0 within a round
z = (ps.filter(pl.col("player_id") == 0)
       .group_by(["round"]).agg(pl.col("player_controller_id").n_unique().alias("n_ctrl0")))
print("rounds with player_id0 controllers:", z.group_by("n_ctrl0").len().to_dicts())

# team codes per player_id per round (vector)
t = (pv.group_by(["round", "player_id"]).agg(pl.col("team_code").n_unique().alias("nteam"))
       .filter(pl.col("nteam") > 1))
print("round,player_id with >1 team_code:", t.height, t.head(5).to_dicts())
