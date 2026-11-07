"""Parity: the laptop's status_1hz rows (34-match seeded sample, Oct 9-25) versus the reduced player_status that the
v4 build wrote into the chunk tomes on S3.

Join key: (match_key, round, tick, player_id). Laptop rows are already filtered to T/CT players (side 2 or 3); the
tome rows have no side column, so every tome row counts as a candidate for the match.

Reads S3 (tome pages only), writes nothing to S3. Writes parity_report.json next to this script.
Usage (space, JupyterLab terminal):  ~/venv-dsdk/bin/python -W ignore parity_compare.py
"""
import glob
import json
import os
import time

import boto3
import polars as pl
from botocore.config import Config

BUCKET = "pureskillgg-csgo-production-sagemaker-0"
ADMIN = os.path.expanduser("~/tomes/tome/csds/header")
HERE = os.path.expanduser("~/adam/status_scripts/parity")
WORK = "/mnt/sagemaker-nvme/parity"
CHUNK = 1000
CHANNEL = "player_status"
INV = ["inv_primary", "inv_secondary", "inv_flashbang", "inv_smokegrenade", "inv_hegrenade", "inv_molotov",
       "inv_incgrenade", "inv_decoy", "inv_c4"]
VALUE_COLS = ["health", "armor", "has_helmet", "has_defuser", "money", "equipment_value_calc", "is_defusing"] + INV
KEY = ["round", "tick", "player_id"]
JOIN = ["match_key"] + KEY
s3 = boto3.client("s3", config=Config(max_pool_connections=32, retries={"max_attempts": 5},
                                      request_checksum_calculation="when_required",
                                      response_checksum_validation="when_required"))


def admin_keys():
    fs = sorted(glob.glob(os.path.join(ADMIN, "dataframe_*")))
    return sorted(set(pl.concat([pl.read_parquet(f, columns=["key"]) for f in fs])["key"].to_list()))


def rows(df, n=10):
    return df.head(n).to_dicts()


def main():
    t0 = time.time()
    L_raw = pl.read_parquet(os.path.join(HERE, "laptop_status_1hz_sample.parquet"))
    keys = pl.read_csv(os.path.join(HERE, "sample_keys.csv"))
    sample = keys["match_key"].to_list()

    admin = admin_keys()
    pos = {k: i for i, k in enumerate(admin)}
    not_in_admin = [m for m in sample if m not in pos]
    chunk_of = {m: pos[m] // CHUNK for m in sample if m in pos}
    needed = sorted(set(chunk_of.values()))
    print(f"admin keys {len(admin)}; sample matches {len(sample)}; not in admin {len(not_in_admin)}; "
          f"chunks needed {needed}", flush=True)

    # --- tome side: reduced player_status for the sample, per chunk ---
    V_frames, chunk_info = [], {}
    for i in needed:
        prefix = f"adam/tomes/chunks/c{i:03d}/tome/csds/basic_{CHANNEL}."
        s3_keys = []
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
            for o in page.get("Contents", []):
                if "/dataframe_" in o["Key"]:
                    s3_keys.append(o["Key"])
        folders = sorted({k.split("/tome/csds/")[1].split("/")[0] for k in s3_keys})
        d = os.path.join(WORK, f"c{i:03d}")
        os.makedirs(d, exist_ok=True)
        local = []
        for k in s3_keys:
            dst = os.path.join(d, os.path.basename(k))
            if not os.path.exists(dst):
                s3.download_file(BUCKET, k, dst)
            local.append(dst)
        V_i = pl.concat([pl.scan_parquet(p).filter(pl.col("match_key").is_in(sample)).collect() for p in sorted(local)],
                        how="vertical_relaxed")
        V_frames.append(V_i)
        chunk_info[f"c{i:03d}"] = {"folders": folders, "pages": len(local), "sample_rows": V_i.height,
                                   "sample_matches_found": V_i["match_key"].n_unique()}
        print(f"c{i:03d}: folders {folders}, pages {len(local)}, sample rows {V_i.height}", flush=True)
    V = pl.concat(V_frames, how="vertical_relaxed")
    missing_cols = [c for c in VALUE_COLS if c not in V.columns]
    if missing_cols:
        raise SystemExit(f"tome is missing columns {missing_cols}; columns {V.columns}")

    # --- join ---
    L = L_raw.select(JOIN + VALUE_COLS).with_columns(
        [pl.col(c).cast(pl.Int64) for c in ["round", "tick", "player_id"] + VALUE_COLS]).with_columns(
        pl.lit(1).alias("in_L"))
    Vs = V.select(JOIN + VALUE_COLS).with_columns(
        [pl.col(c).cast(pl.Int64) for c in ["round", "tick", "player_id"] + VALUE_COLS]).with_columns(
        pl.lit(1).alias("in_V"))
    dup_L = L.height - L.unique(subset=JOIN).height
    dup_V = Vs.height - Vs.unique(subset=JOIN).height
    j = L.join(Vs, on=JOIN, how="full", suffix="_v", coalesce=True)
    j = j.with_columns(
        (pl.col("in_L").is_not_null() & pl.col("in_V").is_null()).alias("L_only"),
        (pl.col("in_L").is_null() & pl.col("in_V").is_not_null()).alias("V_only"),
        (pl.col("in_L").is_not_null() & pl.col("in_V").is_not_null()).alias("shared"),
    )
    j = j.with_columns(
        (pl.col("shared") & pl.any_horizontal([pl.col(c).ne_missing(pl.col(c + "_v")) for c in VALUE_COLS]))
        .alias("mism"))

    per_col = {c: int(j.filter(pl.col("shared") & pl.col(c).ne_missing(pl.col(c + "_v"))).height) for c in VALUE_COLS}

    # tick-set check per (match, round): same ticks on both sides?
    tl = L.group_by(["match_key", "round"]).agg(pl.col("tick").n_unique().alias("nL"))
    tv = Vs.group_by(["match_key", "round"]).agg(pl.col("tick").n_unique().alias("nV"))
    rt = tl.join(tv, on=["match_key", "round"], how="full", coalesce=True).with_columns(
        pl.col("nL").fill_null(0), pl.col("nV").fill_null(0))
    rt_diff = rt.filter(pl.col("nL") != pl.col("nV"))

    per_match = (j.group_by("match_key").agg(
        pl.col("in_L").is_not_null().sum().alias("L_rows"),
        pl.col("in_V").is_not_null().sum().alias("V_rows"),
        pl.col("L_only").sum().alias("L_only"),
        pl.col("V_only").sum().alias("V_only"),
        pl.col("mism").sum().alias("mismatch"),
    ).sort("match_key"))
    per_match_rows = per_match.to_dicts()
    for r in per_match_rows:
        r["chunk"] = f"c{chunk_of[r['match_key']]:03d}" if r["match_key"] in chunk_of else "not_in_admin"
    zero_side = [r for r in per_match_rows if r["L_rows"] == 0 or r["V_rows"] == 0]

    totals = {
        "L_rows": int(L.height), "V_rows": int(Vs.height), "shared_rows": int(j["shared"].sum()),
        "L_only_rows": int(j["L_only"].sum()), "V_only_rows": int(j["V_only"].sum()),
        "mismatch_rows": int(j["mism"].sum()), "dup_keys_L": int(dup_L), "dup_keys_V": int(dup_V),
        "round_tick_count_diffs": int(rt_diff.height),
        "matches_with_zero_rows_on_a_side": len(zero_side),
        "mismatch_by_column": per_col,
    }
    reasons = []
    for k in ["L_only_rows", "V_only_rows", "mismatch_rows", "dup_keys_L", "dup_keys_V", "round_tick_count_diffs",
              "matches_with_zero_rows_on_a_side"]:
        if totals[k]:
            reasons.append(f"{k}={totals[k]}")
    if not_in_admin:
        reasons.append(f"sample matches not in admin keys={len(not_in_admin)}")
    verdict = "PASS" if not reasons and totals["shared_rows"] > 0 else "FAIL: " + "; ".join(reasons)

    report = {
        "verdict": verdict, "seed_sample": "2 per day, Oct 9-25 (laptop status_1hz)",
        "sample_matches": len(sample), "matches_with_L_rows": int(L["match_key"].n_unique()),
        "matches_with_V_rows": int(Vs["match_key"].n_unique()), "not_in_admin": not_in_admin,
        "totals": totals, "chunks": chunk_info,
        "examples_L_only": rows(j.filter(pl.col("L_only")).select(JOIN + VALUE_COLS)),
        "examples_V_only": rows(j.filter(pl.col("V_only")).select(JOIN + [c + "_v" for c in VALUE_COLS])),
        "examples_mismatch": rows(j.filter(pl.col("mism")).select(
            JOIN + [c for c in VALUE_COLS] + [c + "_v" for c in VALUE_COLS])),
        "examples_round_tick_diff": rows(rt_diff),
        "per_match": per_match_rows,
        "seconds": round(time.time() - t0),
    }
    out = os.path.join(HERE, "parity_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1, default=str)

    print("VERDICT:", verdict, flush=True)
    print(json.dumps({k: v for k, v in totals.items()}, indent=1), flush=True)
    print(f"sample matches {len(sample)}, with laptop rows {report['matches_with_L_rows']}, with tome rows "
          f"{report['matches_with_V_rows']}, not in admin {len(not_in_admin)}", flush=True)
    print(f"report: {out}  ({report['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
