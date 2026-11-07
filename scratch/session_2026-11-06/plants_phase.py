"""Phase A for the position features: bomb-plant positions for the bucket matches, then per-map site centres fitted on TRAIN matches only.

Reads raw bomb_action (S3 GET, to NVMe scratch). Writes final/plants_bucket.parquet and final/sites_v1.json.
The laptop plant events (posport/data/positions_1hz/_plants) are added, so both sources share one set of centres.
"""
import glob
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

import boto3
import numpy as np
import polars as pl
from botocore.config import Config

B = "pureskillgg-csgo-production-sagemaker-0"
BUILDER = os.path.expanduser("~/adam/builder")
FIN = BUILDER + "/final"
SCR = "/mnt/sagemaker-nvme/plants"
PORT = BUILDER + "/posport/data"
cli = boto3.client("s3", config=Config(max_pool_connections=64, retries={"max_attempts": 8},
                                       request_checksum_calculation="when_required",
                                       response_checksum_validation="when_required"))
t0 = time.time()

cr = pl.read_parquet(FIN + "/clean_rounds.parquet", columns=["match_key", "use", "src"]).filter(pl.col("use"))
bkeys = sorted(cr.filter(pl.col("src") == "bucket")["match_key"].unique().to_list())
print("bucket matches", len(bkeys), flush=True)


def one(mk):
    folder = mk.rsplit("/", 1)[0]
    dst = f"{SCR}/{folder}/bomb_action"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        cli.download_file(B, f"data/{folder}/bomb_action", dst)
    except Exception:
        return None
    ba = pl.read_parquet(dst, columns=["round", "tick", "event_type", "site_code", "player_x_pos", "player_y_pos"])
    shutil.rmtree(f"{SCR}/{folder.split('/')[0]}", ignore_errors=True) if False else os.remove(dst)
    p = ba.filter(pl.col("event_type") == "bomb_begin_plant")
    if not p.height:
        return None
    return p.with_columns(pl.lit(mk).alias("match_key")).rename({"player_x_pos": "x", "player_y_pos": "y"}).select(
        "match_key", "round", "tick", "site_code", "x", "y")


with ThreadPoolExecutor(48) as ex:
    parts = [r for r in ex.map(one, bkeys) if r is not None]
PB = pl.concat(parts, how="vertical_relaxed").with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64))
PB.write_parquet(FIN + "/plants_bucket.parquet")
print("bucket plant events", PB.height, "| matches with a plant", PB["match_key"].n_unique(), f"| secs {time.time() - t0:.0f}", flush=True)

PL = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(PORT + "/positions_1hz/_plants/*.parquet"))], how="vertical_relaxed")
PL = PL.with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64))
ALL = pl.concat([PB.select("match_key", "round", "tick", "x", "y"), PL.select("match_key", "round", "tick", "x", "y")], how="vertical_relaxed")
mp = dict(zip(*pl.read_parquet(FIN + "/clean_matches.parquet", columns=["match_key", "map_name"]).unique("match_key").select("match_key", "map_name").to_dict(as_series=False).values()))
ALL = ALL.with_columns(pl.col("match_key").replace_strict(mp, default=None).alias("map_name")).drop_nulls(["x", "y", "map_name"])

split = json.load(open(FIN + "/split_manifest_v2.json"))["assign"]
train_keys = {k for k, v in split.items() if v == "train"}
ALLt = ALL.filter(pl.col("match_key").is_in(list(train_keys)))
print("plants total", ALL.height, "| on train matches", ALLt.height, flush=True)


def centres(xy):
    rng = np.random.default_rng(0)
    best = None
    for _ in range(25):
        c = xy[rng.choice(len(xy), 2, replace=False)]
        for _ in range(40):
            d = ((xy[:, None, :] - c[None]) ** 2).sum(-1)
            lab = d.argmin(1)
            c = np.stack([np.median(xy[lab == k], 0) if (lab == k).any() else c[k] for k in range(2)])
        d = ((xy[:, None, :] - c[None]) ** 2).sum(-1)
        lab = d.argmin(1)
        inertia = float(d.min(1).sum())
        share = min((lab == 0).mean(), (lab == 1).mean())
        if share >= 0.2 and (best is None or inertia < best[0]):
            best = (inertia, c.copy(), lab.copy())
    _, c, lab = best
    return c[np.argsort(c[:, 0])], [int((lab == k).sum()) for k in range(2)]


sites = {}
for (m,), g in ALLt.group_by("map_name"):
    xy = np.stack([g["x"].to_numpy(), g["y"].to_numpy()], 1).astype("float64")
    c, sizes = centres(xy)
    sites[m] = c.tolist()
    print(f"  {m}: {len(xy)} train plants, centres {np.round(c).astype(int).tolist()}, sizes {sizes}", flush=True)
json.dump(sites, open(FIN + "/sites_v1.json", "w"), indent=1)
print("wrote sites_v1.json", f"| secs {time.time() - t0:.0f}", flush=True)
