import json, os, shutil, sys, time, glob
from concurrent.futures import ThreadPoolExecutor
import boto3
import polars as pl
from botocore.config import Config
from pureskillgg_dsdk.adx.unzip import extract_adx_zips
from pureskillgg_dsdk.tome.builder import build_basic_tomes_from_fs

SRC = "pureskillgg-csgo-production-dataexchange-csds-final-0"
WORK = "pureskillgg-csgo-production-sagemaker-0"
DAYS = ["26", "27", "28", "29", "30", "31"]
BUILD = os.path.expanduser("~/holdout_build")
TOME = os.path.expanduser("~/holdout_tome")
UPLOAD = "adam/tomes/holdout_2026-10-26_31"
CHANNELS = ["bomb_action", "bomb_defuse", "bomb_state", "bot_takeover", "bullet_damage", "grenade_bounce",
            "grenade_state", "item_dropped", "item_refund", "molotov_fire", "molotov_state", "other_death",
            "player_blind", "player_chat", "player_connect", "player_death", "player_disconnect",
            "player_hurt", "player_info", "player_inputs", "player_name", "player_personal", "player_sound",
            "player_spawn", "rank_update", "round_end", "round_mvp", "round_start", "round_state",
            "score_update", "team_change", "weapon_action"]
s3 = boto3.client("s3", config=Config(max_pool_connections=64, retries={"max_attempts": 5}))


def list_zips():
    keys = []
    for d in DAYS:
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=SRC, Prefix=f"csds/2026/10/{d}/"):
            keys += [o["Key"] for o in page.get("Contents", []) if o["Key"].endswith(".zip")]
    return sorted(keys)


def get(key):
    dst = os.path.join(BUILD, key)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    s3.download_file(SRC, key, dst)
    return key


def main():
    t0 = time.time()
    zips = list_zips()
    print(f"ZIPS {len(zips)}", flush=True)
    shutil.rmtree(BUILD, ignore_errors=True)
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(get, zips))
    print(f"DOWNLOAD secs={time.time() - t0:.0f}", flush=True)
    t1 = time.time()
    n = extract_adx_zips(BUILD, delete_zips=True)
    print(f"EXTRACT {n} secs={time.time() - t1:.0f}", flush=True)
    expected = sorted(f"csds/2026/10/{z.split('/')[3]}/{z.split('/')[4][:-4]}/csds" for z in zips)
    shutil.rmtree(TOME, ignore_errors=True)
    t2 = time.time()
    build_basic_tomes_from_fs(CHANNELS, ds_type="csds", tome_collection_root_path=TOME, ds_collection_root_path=BUILD)
    print(f"BUILD secs={time.time() - t2:.0f}", flush=True)
    hp = sorted(glob.glob(os.path.join(TOME, "tome", "csds", "header", "dataframe_*")))
    h = pl.concat([pl.read_parquet(f) for f in hp], how="diagonal_relaxed")
    built = set(h["key"].to_list())
    exp = set(expected)
    summary = {"zips": len(zips), "header_rows": h.height, "built_keys": len(built), "expected_keys": len(exp),
               "missing": len(exp - built), "extra": len(built - exp), "max_rounds_filled": h["max_rounds"].drop_nulls().len()}
    print("VERIFY " + json.dumps(summary), flush=True)
    if built != exp:
        print("VERIFY FAILED: not uploading", flush=True)
        sys.exit(3)
    keys_path = os.path.join(TOME, "match_keys.txt")
    with open(keys_path, "w") as f:
        f.write("\n".join(expected) + "\n")
    count = 0
    for root, _, files in os.walk(TOME):
        for name in files:
            local = os.path.join(root, name)
            s3.upload_file(local, WORK, f"{UPLOAD}/{os.path.relpath(local, TOME)}")
            count += 1
    print(f"UPLOAD files={count} prefix=s3://{WORK}/{UPLOAD}/", flush=True)
    print(f"ALL DONE total_secs={time.time() - t0:.0f}", flush=True)


if __name__ == "__main__":
    main()
