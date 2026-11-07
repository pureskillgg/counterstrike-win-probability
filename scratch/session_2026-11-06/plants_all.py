"""Plant events (bomb_begin_plant) for every export match, bucket and laptop-sourced alike, for the parity check only."""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import boto3
import polars as pl
from botocore.config import Config

B = "pureskillgg-csgo-production-sagemaker-0"
BUILDER = os.path.expanduser("~/adam/builder")
FIN = BUILDER + "/final"
SCR = "/mnt/sagemaker-nvme/plants_all"
cli = boto3.client("s3", config=Config(max_pool_connections=64, retries={"max_attempts": 8},
                                       request_checksum_calculation="when_required",
                                       response_checksum_validation="when_required"))
keys = json.load(open(BUILDER + "/final/export_clean/match_keys.json"))
t0 = time.time()


def one(mk):
    folder = mk.rsplit("/", 1)[0]
    dst = f"{SCR}/{folder}/bomb_action"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        cli.download_file(B, f"data/{folder}/bomb_action", dst)
    except Exception:
        return None
    ba = pl.read_parquet(dst, columns=["round", "tick", "event_type", "site_code", "player_x_pos", "player_y_pos"])
    os.remove(dst)
    p = ba.filter(pl.col("event_type") == "bomb_begin_plant")
    if not p.height:
        return None
    return p.with_columns(pl.lit(mk).alias("match_key")).rename({"player_x_pos": "x", "player_y_pos": "y"}).select(
        "match_key", "round", "tick", "site_code", "x", "y")


with ThreadPoolExecutor(48) as ex:
    parts = [r for r in ex.map(one, keys) if r is not None]
PA = pl.concat(parts, how="vertical_relaxed").with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64))
PA.write_parquet(FIN + "/plants_all.parquet")
print("plants_all", PA.height, "matches with a plant", PA["match_key"].n_unique(), f"secs {time.time() - t0:.0f}", flush=True)
