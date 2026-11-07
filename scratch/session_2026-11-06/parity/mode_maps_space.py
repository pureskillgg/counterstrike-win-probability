"""2026 dataset size by map and mode on the SageMaker space.

Headers are read from S3 (data/<folder>/header) for every 2026 match in rounds.parquet. The AWS-covered set is the
chunk set (admin keys). Read-only on S3 (get_object only). Writes per_match_headers_space.parquet and
mode_maps_space_summary.json next to this script.
"""
import glob
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor

import boto3
import polars as pl
import pyarrow.parquet as pq
from botocore.config import Config
from botocore.exceptions import ClientError

BUCKET = "pureskillgg-csgo-production-sagemaker-0"
COMB = os.path.expanduser("~/adam/status/data/combined")
ADMIN = os.path.expanduser("~/tomes/tome/csds/header")
HERE = os.path.expanduser("~/adam/status_scripts/parity")
POOL = ["de_ancient", "de_anubis", "de_cache", "de_dust2", "de_inferno", "de_mirage", "de_nuke"]
WIN_LO, WIN_HI = "2026/10/09", "2026/10/25"
s3 = boto3.client("s3", config=Config(max_pool_connections=128, retries={"max_attempts": 5}))


def admin_keys():
    fs = sorted(glob.glob(os.path.join(ADMIN, "dataframe_*")))
    return sorted(set(pl.concat([pl.read_parquet(f, columns=["key"]) for f in fs])["key"].to_list()))


def read_header(mk):
    folder = mk.rsplit("/", 1)[0]
    try:
        body = s3.get_object(Bucket=BUCKET, Key=f"data/{folder}/header")["Body"].read()
    except ClientError as e:
        code = e.response["Error"]["Code"]
        return (mk, None, None, None, None, "missing" if code in ("NoSuchKey", "404") else "error")
    try:
        t = pq.read_table(io.BytesIO(body)).to_pylist()[0]
    except Exception:
        return (mk, None, None, None, None, "unreadable")
    return (mk, t.get("map_name"), t.get("is_wingman"), t.get("max_unique_players"), t.get("match_type"), "ok")


def main():
    R = (pl.read_parquet(os.path.join(COMB, "rounds.parquet"), columns=["match_key", "round", "n_t", "n_ct", "era", "use", "round_status"])
         .filter(pl.col("era") == "2026"))
    mks = R["match_key"].unique().to_list()
    print(f"2026 rounds {R.height}, matches {len(mks)}", flush=True)

    cache = os.path.join(HERE, "per_match_headers_space.parquet")
    if os.path.exists(cache):
        H = pl.read_parquet(cache)
    else:
        with ThreadPoolExecutor(64) as ex:
            rows = list(ex.map(read_header, mks))
        H = pl.DataFrame(rows, schema=["match_key", "map_name", "is_wingman", "max_unique_players", "match_type", "status"],
                         orient="row")
        H.write_parquet(cache)
    print("header status:", H.group_by("status").agg(pl.len().alias("matches")).sort("matches", descending=True).rows(),
          flush=True)

    admin = set(admin_keys())
    days = [("/".join(k.split("/")[1:4])) for k in H["match_key"].to_list()]
    H = H.with_columns(pl.Series("aws_chunk", [k in admin for k in H["match_key"].to_list()]),
                       pl.Series("day", days))

    M = R.group_by("match_key").agg((pl.col("n_t") + pl.col("n_ct")).max().alias("maxsum"))
    RH = (R.join(H, on="match_key", how="left").join(M, on="match_key", how="left")
          .with_columns(pl.col("is_wingman").fill_null(False),
                        (((pl.col("n_t") == 5) & (pl.col("n_ct") <= 5)) | ((pl.col("n_ct") == 5) & (pl.col("n_t") <= 5))).alias("std"),
                        ((pl.col("n_t") == 5) & (pl.col("n_ct") == 5)).alias("fv")))

    out = {}
    ut = RH.filter(pl.col("use"))
    out["all_2026_rounds"] = RH.height
    out["use_true_rounds"] = ut.height
    out["use_true_matches"] = ut["match_key"].n_unique()
    out["use_true_matches_no_header"] = ut.filter(pl.col("status") != "ok")["match_key"].n_unique()
    print("use True rounds:", out["use_true_rounds"], "matches:", out["use_true_matches"],
          "| matches with no readable header:", out["use_true_matches_no_header"], flush=True)

    by_map_all = (ut.group_by("map_name").agg(pl.len().alias("rounds"), pl.col("match_key").n_unique().alias("matches"))
                  .sort("rounds", descending=True))
    print("use True by map (all modes):", by_map_all.rows(), flush=True)
    out["by_map_all_modes"] = by_map_all.rows()

    by_wing = ut.group_by("is_wingman").agg(pl.len().alias("rounds"), pl.col("match_key").n_unique().alias("matches")).sort("is_wingman")
    print("use True by is_wingman:", by_wing.rows(), flush=True)
    out["by_wingman"] = by_wing.rows()

    nw5 = ut.filter((~pl.col("is_wingman")) & pl.col("fv"))
    out["nonwing_5v5_rounds"] = nw5.height
    out["nonwing_5v5_matches"] = nw5["match_key"].n_unique()
    print("non-wingman 5v5 use rounds:", nw5.height, "matches:", nw5["match_key"].n_unique(), flush=True)

    pool = nw5.filter(pl.col("map_name").is_in(POOL))
    out["pool_rounds"] = pool.height
    out["pool_matches"] = pool["match_key"].n_unique()
    print("active duty pool, 5v5, non-wing, use:", pool.height, "rounds,", out["pool_matches"], "matches", flush=True)
    pool_map = pool.group_by("map_name").agg(pl.len().alias("rounds"), pl.col("match_key").n_unique().alias("matches")).sort("rounds", descending=True)
    print("pool by map:", pool_map.rows(), flush=True)
    out["pool_by_map"] = pool_map.rows()

    cov = pool.filter(pl.col("aws_chunk"))
    out["pool_aws_rounds"] = cov.height
    out["pool_aws_matches"] = cov["match_key"].n_unique()
    print("pool, AWS chunk set (all days):", cov.height, "rounds,", out["pool_aws_matches"], "matches", flush=True)

    win = cov.filter((pl.col("day") >= WIN_LO) & (pl.col("day") <= WIN_HI))
    out["pool_aws_window_rounds"] = win.height
    out["pool_aws_window_matches"] = win["match_key"].n_unique()
    print(f"pool, AWS chunk set, {WIN_LO} to {WIN_HI}:", win.height, "rounds,", out["pool_aws_window_matches"], "matches", flush=True)
    win_map = win.group_by("map_name").agg(pl.len().alias("rounds"), pl.col("match_key").n_unique().alias("matches")).sort("rounds", descending=True)
    print("window by map:", win_map.rows(), flush=True)
    out["window_by_map"] = win_map.rows()

    lost = pool.filter(~pl.col("aws_chunk"))
    out["pool_not_aws_rounds"] = lost.height
    out["pool_not_aws_matches"] = lost["match_key"].n_unique()
    print("pool rounds not in AWS chunk set:", lost.height, "matches", out["pool_not_aws_matches"],
          lost.group_by("day").agg(pl.len().alias("rounds")).sort("day").rows()[:12], flush=True)

    with open(os.path.join(HERE, "mode_maps_space_summary.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("wrote mode_maps_space_summary.json", flush=True)


if __name__ == "__main__":
    main()
