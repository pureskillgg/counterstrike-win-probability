import os, time, shutil
from concurrent.futures import ThreadPoolExecutor
import boto3
from botocore.config import Config
from pureskillgg_dsdk.tome.builder import build_basic_tomes_from_fs

BUCKET = "pureskillgg-csgo-production-sagemaker-0"
DAY = "2026/10/14"
N = 100
LOCAL = os.path.expanduser("~/test_loose")
OUT = os.path.expanduser("~/test_tomes")
CHANNELS = ["bomb_action", "bomb_defuse", "bomb_state", "bot_takeover", "bullet_damage", "grenade_bounce",
            "grenade_state", "item_dropped", "item_refund", "molotov_fire", "molotov_state", "other_death",
            "player_blind", "player_chat", "player_connect", "player_death", "player_disconnect",
            "player_hurt", "player_info", "player_inputs", "player_name", "player_personal", "player_sound",
            "player_spawn", "rank_update", "round_end", "round_mvp", "round_start", "round_state",
            "score_update", "team_change", "weapon_action"]

s3 = boto3.client("s3", config=Config(max_pool_connections=64))
resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"data/csds/{DAY}/", Delimiter="/")
prefixes = sorted(p["Prefix"] for p in resp.get("CommonPrefixes", []))[:N]

def fetch_match(prefix):
    n = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
        for o in page.get("Contents", []):
            rel = o["Key"][len("data/"):]
            dst = os.path.join(LOCAL, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            s3.download_file(BUCKET, o["Key"], dst)
            n += 1
    return n

t = time.time()
with ThreadPoolExecutor(16) as ex:
    files = sum(ex.map(fetch_match, prefixes))
print(f"FETCH matches={len(prefixes)} files={files} secs={time.time() - t:.1f}", flush=True)

t = time.time()
build_basic_tomes_from_fs(CHANNELS, ds_type="csds",
                          tome_collection_root_path=OUT, ds_collection_root_path=LOCAL)
print(f"BUILD secs={time.time() - t:.1f}", flush=True)
print("TOME TREE", sorted(os.listdir(os.path.join(OUT, "tome", "csds")))[:8] if os.path.isdir(os.path.join(OUT, "tome", "csds")) else os.listdir(OUT), flush=True)
