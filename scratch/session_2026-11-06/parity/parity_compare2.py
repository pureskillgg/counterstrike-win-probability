"""Parity v2: the laptop's status_1hz rows versus the reduced player_status in the chunk tomes on S3.

Usage (space):  ~/venv-dsdk/bin/python -W ignore parity_compare2.py <sample.parquet> <keys.csv> <report.json>
  sample.parquet  laptop status_1hz rows for the sample matches (column upload_day added)
  keys.csv        match_key (and upload_day) for the sample matches
Join key (match_key, round, tick, player_id). Laptop rows are T/CT only (side 2 or 3, already filtered by the laptop).
Tome rows carry no side, so every tome row of a sample match is a candidate.

Verdict has two parts:
  VALUES (sample matches that have tome rows only): no laptop-only rows, no value mismatch on shared rows, no duplicate
          keys, and no tome-only rows in a round that the grid uses (rounds.parquet use == True). Tome-only rows in
          unused rounds are counted and explained.
  COVERAGE: sample matches with laptop rows but no chunk entry (not in the admin key list). Their laptop rows are
          counted under L_only_rows_no_tome_match and round_tick_diffs_no_tome_match, not under VALUES.
Reads S3 for tome pages only (cached under /mnt/sagemaker-nvme/parity). Writes nothing to S3.
"""
import glob
import json
import os
import sys
import time

import boto3
import polars as pl
from botocore.config import Config

BUCKET = "pureskillgg-csgo-production-sagemaker-0"
ADMIN = os.path.expanduser("~/tomes/tome/csds/header")
HERE = os.path.expanduser("~/adam/status_scripts/parity")
COMB = os.path.expanduser("~/adam/status/data/combined")
WORK = "/mnt/sagemaker-nvme/parity"
CHUNK = 1000
CHANNEL = "player_status"
INV = ["inv_primary", "inv_secondary", "inv_flashbang", "inv_smokegrenade", "inv_hegrenade", "inv_molotov",
       "inv_incgrenade", "inv_decoy", "inv_c4"]
VALUE_COLS = ["health", "armor", "has_helmet", "has_defuser", "money", "equipment_value_calc", "is_defusing"] + INV
KEY = ["round", "tick", "player_id"]
MK_R = ["match_key", "round"]
JOIN = ["match_key"] + KEY
s3 = boto3.client("s3", config=Config(max_pool_connections=32, retries={"max_attempts": 5},
                                      request_checksum_calculation="when_required",
                                      response_checksum_validation="when_required"))


def admin_keys():
    fs = sorted(glob.glob(os.path.join(ADMIN, "dataframe_*")))
    return sorted(set(pl.concat([pl.read_parquet(f, columns=["key"]) for f in fs])["key"].to_list()))


def main():
    t0 = time.time()
    sample_file, keys_file, report_name = sys.argv[1], sys.argv[2], sys.argv[3]
    L_raw = pl.read_parquet(os.path.join(HERE, sample_file))
    sample = pl.read_csv(os.path.join(HERE, keys_file))["match_key"].to_list()

    admin = admin_keys()
    pos = {k: i for i, k in enumerate(admin)}
    not_in_admin = [m for m in sample if m not in pos]
    chunk_of = {m: pos[m] // CHUNK for m in sample if m in pos}
    needed = sorted(set(chunk_of.values()))
    print(f"sample matches {len(sample)}; admin keys {len(admin)}; sample not in admin {len(not_in_admin)}; "
          f"chunks needed {len(needed)}", flush=True)

    # --- tome side ---
    V_frames, chunk_info = [], {}
    for i in needed:
        prefix = f"adam/tomes/chunks/c{i:03d}/tome/csds/basic_{CHANNEL}."
        s3_keys = []
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
            for o in page.get("Contents", []):
                if "/dataframe_" in o["Key"]:
                    s3_keys.append(o["Key"])
        folders = sorted({k.split("/tome/csds/")[1].split("/")[0] for k in s3_keys})
        if len(folders) != 1:
            raise SystemExit(f"c{i:03d} has {len(folders)} basic_player_status folders: {folders}")
        d = os.path.join(WORK, f"c{i:03d}")
        os.makedirs(d, exist_ok=True)
        local = []
        for k in s3_keys:
            dst = os.path.join(d, os.path.basename(k))
            if not os.path.exists(dst):
                s3.download_file(BUCKET, k, dst)
            local.append(dst)
        V_i = pl.concat([pl.scan_parquet(p).filter(pl.col("match_key").is_in(sample)).collect()
                         for p in sorted(local)], how="vertical_relaxed")
        V_frames.append(V_i)
        chunk_info[f"c{i:03d}"] = {"folder": folders[0], "pages": len(local), "sample_rows": V_i.height}
    V = pl.concat(V_frames, how="vertical_relaxed")
    missing_cols = [c for c in VALUE_COLS if c not in V.columns]
    if missing_cols:
        raise SystemExit(f"tome is missing columns {missing_cols}")

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
        (pl.col("in_L").is_not_null() & pl.col("in_V").is_not_null()).alias("shared"))
    j = j.with_columns(
        (pl.col("shared") & pl.any_horizontal([pl.col(c).ne_missing(pl.col(c + "_v")) for c in VALUE_COLS]))
        .alias("mism"))
    per_col = {c: int(j.filter(pl.col("shared") & pl.col(c).ne_missing(pl.col(c + "_v"))).height) for c in VALUE_COLS}
    tome_matches = Vs["match_key"].unique().to_list()
    j = j.with_columns(pl.col("match_key").is_in(tome_matches).alias("has_tome"))

    # --- tome-only rows by round use flag ---
    R = (pl.read_parquet(os.path.join(COMB, "rounds.parquet"), columns=["match_key", "round", "use", "round_status"])
         .with_columns(pl.col("round").cast(pl.Int64)))
    vo_rounds = (j.filter(pl.col("V_only")).group_by(MK_R).agg(pl.len().alias("V_only_rows"))
                 .join(R, on=MK_R, how="left"))
    vo_use = int(vo_rounds.filter(pl.col("use") == True)["V_only_rows"].sum() or 0)  # noqa: E712
    vo_unknown = int(vo_rounds.filter(pl.col("use").is_null())["V_only_rows"].sum() or 0)
    vo_unused_by_status = (vo_rounds.filter(pl.col("use") == False)  # noqa: E712
                           .group_by("round_status").agg(pl.col("V_only_rows").sum().alias("rows"),
                                                         pl.len().alias("rounds"))
                           .sort("round_status").to_dicts())

    # --- round tick-count check ---
    tl = L.group_by(MK_R).agg(pl.col("tick").n_unique().alias("nL"))
    tv = Vs.group_by(MK_R).agg(pl.col("tick").n_unique().alias("nV"))
    rt = tl.join(tv, on=MK_R, how="full", coalesce=True).with_columns(
        pl.col("nL").fill_null(0), pl.col("nV").fill_null(0))
    rt_diff = rt.filter(pl.col("nL") != pl.col("nV")).join(R, on=MK_R, how="left")
    rt_diff_tome = rt_diff.filter(pl.col("match_key").is_in(tome_matches))
    rt_diff_in_use = rt_diff_tome.filter(pl.col("use") == True).height  # noqa: E712

    per_match = j.group_by("match_key").agg(
        pl.col("in_L").is_not_null().sum().alias("L_rows"),
        pl.col("in_V").is_not_null().sum().alias("V_rows"),
        pl.col("L_only").sum().alias("L_only"),
        pl.col("V_only").sum().alias("V_only"),
        pl.col("mism").sum().alias("mismatch"),
    ).sort("match_key").to_dicts()
    for r in per_match:
        r["chunk"] = f"c{chunk_of[r['match_key']]:03d}" if r["match_key"] in chunk_of else "not_in_admin"
    zero_side = [r for r in per_match if r["L_rows"] == 0 or r["V_rows"] == 0]

    totals = {
        "L_rows": int(L.height), "V_rows": int(Vs.height), "shared_rows": int(j["shared"].sum()),
        "L_only_rows": int(j.filter(pl.col("has_tome"))["L_only"].sum()),
        "L_only_rows_no_tome_match": int(j.filter(~pl.col("has_tome"))["L_only"].sum()),
        "V_only_rows": int(j["V_only"].sum()),
        "V_only_rows_in_use_rounds": vo_use, "V_only_rows_round_not_in_rounds_parquet": vo_unknown,
        "V_only_rows_in_unused_rounds": int(j["V_only"].sum()) - vo_use - vo_unknown,
        "mismatch_rows": int(j["mism"].sum()), "dup_keys_L": int(dup_L), "dup_keys_V": int(dup_V),
        "round_tick_diffs": int(rt_diff_tome.height), "round_tick_diffs_in_use_rounds": int(rt_diff_in_use),
        "round_tick_diffs_no_tome_match": int(rt_diff.height - rt_diff_tome.height),
        "matches_with_zero_rows_on_a_side": len(zero_side), "mismatch_by_column": per_col,
    }
    value_reasons = [f"{k}={totals[k]}" for k in ["L_only_rows", "mismatch_rows", "dup_keys_L", "dup_keys_V",
                                                  "V_only_rows_in_use_rounds",
                                                  "V_only_rows_round_not_in_rounds_parquet",
                                                  "round_tick_diffs_in_use_rounds"] if totals[k]]
    values_verdict = ("PASS" if not value_reasons and totals["shared_rows"] > 0
                      else "FAIL: " + "; ".join(value_reasons or ["no shared rows"]))
    coverage = {"sample_matches": len(sample), "with_laptop_rows": int(L["match_key"].n_unique()),
                "with_tome_rows": int(Vs["match_key"].n_unique()), "not_in_admin": not_in_admin}

    report = {
        "values_verdict": values_verdict, "coverage": coverage, "totals": totals,
        "vo_unused_by_round_status": vo_unused_by_status,
        "chunks": chunk_info,
        "examples_L_only": j.filter(pl.col("L_only")).select(JOIN + VALUE_COLS).head(10).to_dicts(),
        "examples_mismatch": j.filter(pl.col("mism")).select(JOIN + VALUE_COLS
                                                             + [c + "_v" for c in VALUE_COLS]).head(10).to_dicts(),
        "examples_V_only_in_use": (j.filter(pl.col("V_only")).group_by(MK_R).agg(pl.len().alias("rows"))
                                   .join(R, on=MK_R, how="left").filter(pl.col("use") == True)  # noqa: E712
                                   .head(10).to_dicts()),
        "per_match": per_match,
        "seconds": round(time.time() - t0),
    }
    out = os.path.join(HERE, report_name)
    with open(out, "w") as f:
        json.dump(report, f, indent=1, default=str)

    print("VALUES:", values_verdict, flush=True)
    print(json.dumps(totals, indent=1), flush=True)
    print("tome-only rows by round_status (unused rounds):", vo_unused_by_status, flush=True)
    print(f"COVERAGE: sample {coverage['sample_matches']}, with laptop rows {coverage['with_laptop_rows']}, "
          f"with tome rows {coverage['with_tome_rows']}, not in admin {len(not_in_admin)}", flush=True)
    print(f"report: {out} ({report['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
