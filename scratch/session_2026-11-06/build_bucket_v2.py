"""Bucket chunk tomes -> rounds on the space, using the laptop build scripts unchanged except for paths and header/channel padding.

Layout
  SCRATCH  /mnt/sagemaker-nvme/adam_builder   instance-store NVMe: chunks/ (S3 copies) and tomes/ (intermediate).
                                             Wiped when the space stops, so it only holds data that can be re-fetched.
  OUT      ~/adam/builder                     EBS: data/cNNN (per-chunk rounds), final/, logs. Survives a restart.
  SCRATCH/data is a symlink to OUT/data, because prepare.py writes its output next to its tome root.

Run on adam-gpu with ~/venv-dsdk/bin/python from ~/adam/builder:
  build_bucket.py ranges                      list each chunk's date range (S3 listing only)
  build_bucket.py run [--chunks ...] [--workers N]
                                              fetch chunks with download_file (GET only) and build each one as soon as its
                                              fetch finishes, so fetch and build overlap. Logs one line per chunk.
  build_bucket.py merge                       global duplicate pass -> final/bucket_matches.parquet, final/bucket_rounds.parquet
  build_bucket.py check                       compare final rounds with the existing laptop-built rounds on overlapping matches
Writes nothing to S3.
"""
import argparse
import glob
import json
import multiprocessing
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import boto3
import polars as pl
from boto3.s3.transfer import TransferConfig
from botocore.config import Config

SCRATCH = "/mnt/sagemaker-nvme/adam_builder"
OUT = os.path.expanduser("~/adam/builder")
EXISTING = os.path.expanduser("~/adam/status/data/combined/rounds.parquet")
BUCKET = "pureskillgg-csgo-production-sagemaker-0"
PREFIX = "adam/tomes/chunks"
MONTH = r"(\d{4}/\d{2})/\d{2}"
sys.path.insert(0, OUT)  # build_combined.py and prepare.py

CFG = Config(max_pool_connections=64, retries={"max_attempts": 8, "mode": "standard"},
             request_checksum_calculation="when_required", response_checksum_validation="when_required")
TC = TransferConfig(max_concurrency=8, multipart_threshold=64 * 1024 * 1024)

# Header columns prepare.py reads. Older chunks carry only part of the header (no is_wingman, no max_rounds);
# the laptop combined store fills a column missing in a source with nulls of the canonical type, so do the same here.
HEADER_REQ = {"build_num": pl.Int64, "map_name": pl.String, "match_date": pl.String, "match_type": pl.String,
              "is_wingman": pl.Boolean, "max_rounds": pl.Int64, "t_starters_avg_rank": pl.Float64,
              "ct_starters_avg_rank": pl.Float64, "t_starters_score_final": pl.Int64, "ct_starters_score_final": pl.Int64}
# Channels prepare.py scans. A chunk with no page for one of the non-dense ones (e.g. no bot takeovers in 2023) gets an
# empty page with the same key columns, so the join and concat in prepare.py see a zero-row channel instead of a missing one.
PREPARE_CH = ["round_end", "round_state", "round_start", "player_info", "player_death", "player_disconnect",
              "bot_takeover", "bomb_state"]
DENSE_CH = ["round_end", "round_start", "round_state", "player_death", "player_info", "bomb_state"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def client():
    return boto3.client("s3", config=CFG)


def all_chunks():
    out = []
    for page in client().get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=PREFIX + "/", Delimiter="/"):
        out += [p["Prefix"].rstrip("/").split("/")[-1] for p in page.get("CommonPrefixes", [])]
    return sorted(out)


def ranges(c):
    cli = client()
    names = []
    for page in cli.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=f"{PREFIX}/{c}/tome/csds/", Delimiter="/"):
        names += [p["Prefix"].rstrip("/").split("/")[-1] for p in page.get("CommonPrefixes", [])]
    rr = [n for n in names if n.startswith("basic_round_end.")]
    return c, (rr[0].split(".", 1)[1] if rr else "no round_end"), len(names)


def fetch_chunk(c):
    """Download every object under one chunk's tome/csds with download_file. Skips files already present at the right size."""
    t0 = time.time()
    cli = client()
    base = os.path.join(SCRATCH, "chunks", c)
    items = []
    for page in cli.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=f"{PREFIX}/{c}/tome/csds/"):
        items += [(o["Key"], o["Size"]) for o in page.get("Contents", [])]
    todo = []
    for key, size in items:
        dst = os.path.join(base, key[len(f"{PREFIX}/{c}/"):])
        if os.path.exists(dst) and os.path.getsize(dst) == size:
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        todo.append((key, size, dst))

    def one(t):
        key, size, dst = t
        cli.download_file(BUCKET, key, dst + ".part", Config=TC)
        if os.path.getsize(dst + ".part") != size:
            raise IOError(f"size mismatch for {key}")
        os.replace(dst + ".part", dst)

    with ThreadPoolExecutor(16) as ex:
        list(ex.map(one, todo))
    nbytes = sum(s for _, s, _ in todo)
    return c, len(items), len(todo), round(nbytes / 1e6), round(time.time() - t0, 1)


def pad_header(folder):
    for f in sorted(glob.glob(os.path.join(folder, "dataframe_*"))):
        df = pl.read_parquet(f)
        missing = [c for c in HEADER_REQ if c not in df.columns]
        if missing:
            df = df.with_columns([pl.lit(None, dtype=HEADER_REQ[c]).alias(c) for c in missing])
            df.write_parquet(f + ".tmp", compression="zstd")
            os.replace(f + ".tmp", f)


def ensure_channels(base):
    import build_combined as bc
    ref_folder = os.path.join(base, f"basic_player_disconnect{bc.SUFFIX}")
    ref = pl.read_parquet(sorted(glob.glob(os.path.join(ref_folder, "dataframe_*")))[0]).select("match_key", "round").clear()
    for ch in PREPARE_CH:
        folder = os.path.join(base, f"basic_{ch}{bc.SUFFIX}")
        if os.path.isdir(folder):
            continue
        if ch in DENSE_CH:
            raise SystemExit(f"{base}: required dense channel {ch} is missing; this chunk cannot be built")
        os.makedirs(folder)
        ref.write_parquet(os.path.join(folder, "dataframe_00000"), compression="zstd")
        pl.DataFrame({"_": pl.Series([], dtype=pl.String)}).write_parquet(os.path.join(folder, "keyset_00000"), compression="zstd")


def build_chunk(c):
    """One chunk: laptop build_channel for every channel, then the laptop prepare.main. Runs in a worker process."""
    import build_combined as bc
    import prepare as pp

    if os.path.exists(os.path.join(OUT, "data", c, "rounds.parquet")):
        return c, "exists", 0, 0
    t0 = time.time()
    tome_root = os.path.join(SCRATCH, "tomes", c)
    shutil.rmtree(tome_root, ignore_errors=True)
    base = os.path.join(tome_root, "tome", "csds")
    os.makedirs(base)
    bc.TOMES = os.path.join(SCRATCH, "chunks")  # list_channels reads TOMES/<chunk>/tome/csds
    per = {c: bc.list_channels(c)}
    sources = [(c, "2026")]  # era comes from match_key inside prepare; this value is not used there
    for ch in sorted(per[c]):
        folder = os.path.join(base, "header" if ch == "header" else f"basic_{ch}{bc.SUFFIX}")
        bc.build_channel(ch, sources, per, folder)
    pad_header(os.path.join(base, "header"))
    ensure_channels(base)
    pp.ROOT = SCRATCH  # prepare reads SCRATCH/tomes/<store>/tome/csds and writes SCRATCH/data/<store> (symlinked to OUT)
    sys.argv = ["prepare.py", "--store", c]
    pp.main()
    shutil.rmtree(tome_root, ignore_errors=True)
    rep = json.load(open(os.path.join(OUT, "data", c, "prep_report.json")))
    return c, "built", round(time.time() - t0, 1), rep.get("use_matches", 0)


def run(chunks, workers):
    os.makedirs(os.path.join(SCRATCH, "chunks"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "data"), exist_ok=True)
    link = os.path.join(SCRATCH, "data")
    if not os.path.lexists(link):
        os.symlink(os.path.join(OUT, "data"), link)
    errors = {}
    t0 = time.time()
    ctx = multiprocessing.get_context("spawn")  # spawn, not fork: fetch threads are running in this process
    with ThreadPoolExecutor(3) as fx, ProcessPoolExecutor(workers, mp_context=ctx) as px:
        fetch_f = {c: fx.submit(fetch_chunk, c) for c in chunks}
        build_f = {}
        for c in chunks:  # in chunk order; each build starts as soon as its own fetch is done
            try:
                log("fetch", *fetch_f[c].result())
            except Exception as e:
                errors[c] = "fetch " + repr(e)[:300]
                log("FETCH FAILED", c, repr(e)[:200])
                continue
            build_f[c] = px.submit(build_chunk, c)
        n_done = 0
        for c, f in build_f.items():
            try:
                r = f.result()
                n_done += 1
                rate = n_done / max((time.time() - t0) / 60, 1e-9)
                log("build", *r, f"done={n_done}/{len(chunks)} elapsed={time.time() - t0:.0f}s chunks_per_min={rate:.2f}")
            except Exception as e:
                errors[c] = "build " + repr(e)[:300]
                log("BUILD FAILED", c, repr(e)[:200])
                traceback.print_exc()
    with open(os.path.join(OUT, "run_errors.json"), "w") as f:
        json.dump(errors, f, indent=1)
    log("run finished, failures:", len(errors), "elapsed", round(time.time() - t0), "s")


def merge():
    os.makedirs(os.path.join(OUT, "final"), exist_ok=True)
    dirs = sorted(glob.glob(os.path.join(OUT, "data", "c[0-9][0-9][0-9]")))
    M = pl.concat([pl.read_parquet(os.path.join(d, "matches.parquet")) for d in dirs], how="diagonal_relaxed")
    R = pl.concat([pl.read_parquet(os.path.join(d, "rounds.parquet")) for d in dirs], how="diagonal_relaxed")
    print("chunks:", len(dirs), "match rows:", M.height, "distinct match_key:", M["match_key"].n_unique(),
          "round rows:", R.height, flush=True)
    M = M.unique("match_key", keep="first", maintain_order=True)
    M = M.drop([c for c in ["dup_keeper", "dup_group_size"] if c in M.columns])  # per-chunk values, replaced by the global ones
    M = M.sort(["dup_key", "n_ok_rounds", "match_key"], descending=[False, True, False], maintain_order=True)
    M = M.with_columns(
        pl.len().over("dup_key").alias("g_group_size"),
        (pl.int_range(pl.len()).over("dup_key") == 0).alias("g_keeper"),
    )
    print("global duplicate groups:", M.filter(pl.col("g_group_size") > 1)["dup_key"].n_unique(),
          "matches not kept:", M.filter(~pl.col("g_keeper")).height, flush=True)
    R = R.drop([c for c in ["dup_keeper", "dup_group_size", "channels_complete", "in_scope", "use"] if c in R.columns])
    R = R.join(M.select("match_key", "g_keeper", "g_group_size", "channels_complete", "in_scope"), on="match_key", how="left")
    R = R.rename({"g_keeper": "dup_keeper", "g_group_size": "dup_group_size"})
    R = R.with_columns((pl.col("round_status").eq("ok") & pl.col("in_scope") & pl.col("dup_keeper")
                        & pl.col("channels_complete")).alias("use"))
    M = M.rename({"g_keeper": "dup_keeper", "g_group_size": "dup_group_size"})
    M.write_parquet(os.path.join(OUT, "final", "bucket_matches.parquet"))
    R.write_parquet(os.path.join(OUT, "final", "bucket_rounds.parquet"))
    S = (R.with_columns(pl.col("match_key").str.extract(MONTH, 1).alias("month"))
         .group_by("month").agg(pl.col("match_key").n_unique().alias("matches"),
                                pl.col("match_key").filter(pl.col("use")).n_unique().alias("use_matches"),
                                pl.len().alias("rounds"), pl.col("use").sum().alias("use_rounds")).sort("month"))
    print(S, flush=True)
    print("TOTAL use matches:", R.filter(pl.col("use"))["match_key"].n_unique(),
          "use rounds:", int(R["use"].sum()), flush=True)


def check():
    cols = ["match_key", "round", "round_status", "final", "freeze", "end_eff", "use", "dup_keeper"]
    old = pl.read_parquet(EXISTING, columns=cols)
    new = pl.read_parquet(os.path.join(OUT, "final", "bucket_rounds.parquet"), columns=cols)
    both = set(old["match_key"].unique().to_list()) & set(new["match_key"].unique().to_list())
    print("matches in both (old laptop build, new bucket build):", len(both), flush=True)
    o = old.filter(pl.col("match_key").is_in(list(both)))
    n = new.filter(pl.col("match_key").is_in(list(both)))
    j = o.join(n, on=["match_key", "round"], how="full", suffix="_new", coalesce=True)
    print("rounds old-only:", j.filter(pl.col("round_status_new").is_null()).height,
          "new-only:", j.filter(pl.col("round_status").is_null()).height, flush=True)
    jj = j.filter(pl.col("round_status").is_not_null() & pl.col("round_status_new").is_not_null())
    print("rounds in both:", jj.height, flush=True)
    for c in ["round_status", "final", "freeze", "end_eff", "use", "dup_keeper"]:
        bad = jj.filter(~pl.col(c).eq_missing(pl.col(c + "_new")))
        print(f"mismatch {c}: {bad.height}", flush=True)
        if bad.height:
            print(bad.select("match_key", "round", c, c + "_new").head(5), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["ranges", "run", "merge", "check"])
    ap.add_argument("--chunks", default="")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    chunks = [c for c in a.chunks.split(",") if c] or all_chunks()
    if a.stage == "ranges":
        with ThreadPoolExecutor(16) as ex:
            for c, rng, n in ex.map(ranges, chunks):
                print(c, rng, "channels:", n, flush=True)
    elif a.stage == "run":
        run(chunks, a.workers)
    elif a.stage == "merge":
        merge()
    elif a.stage == "check":
        check()


if __name__ == "__main__":
    main()
