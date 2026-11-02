"""The 59-input random-tick pool, built one upload day at a time (the one-piece version ran out of memory).

For each seed: the 2026 rows of random_tick_pool_s<seed>.parquet (13 event inputs at each drawn tick) get the 46 loadout inputs from the
per-tick player status at those ticks (data/status_random, one file per upload day, read one day at a time through
features_status.loadout_features(files=...)). Each day is appended to the output as it finishes, so memory stays at one day's worth.

  ...python.exe -I -X utf8 scripts\\data\\build_random_tick_pool59_daily.py --seed 0 [--store combined] [--status-dir ...]
Writes data/<store>/random_tick_pool59_s<seed>.parquet.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
import polars as pl
import pyarrow.parquet as pq

import features_status as FS
import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")
BASIC = ["alive_ct", "alive_t", "ct_alive_minus_t", "bomb_planted", "seconds_since_round_live", "seconds_since_plant"] + [
    f"map_is_{m}" for m in ["ancient", "anubis", "cache", "dust2", "inferno", "mirage", "nuke"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--store", default="combined")
    ap.add_argument("--status-dir", default=None)
    a = ap.parse_args()
    d = os.path.join(W.ROOT, "data", a.store)
    status_dir = a.status_dir or os.path.join(W.ROOT, "data", "status_random")
    t0 = time.time()
    feats59 = json.load(open(os.path.join(W.ROOT, "models", "winprob_v3.meta.json"), encoding="utf-8"))["features"]
    st_names = [f for f in feats59 if f not in BASIC]

    pool = pl.read_parquet(os.path.join(d, f"random_tick_pool_s{a.seed}.parquet")).filter(pl.col("era") == "2026")
    day_of = pl.read_parquet(os.path.join(d, "grid.parquet"), columns=["match_key", "upload_day"]).unique(subset=["match_key"])
    pool = pool.join(day_of, on="match_key", how="left")
    days = sorted(pool["upload_day"].drop_nulls().unique().to_list())
    print(f"seed {a.seed}: {pool.height:,} 2026 rows over {len(days)} upload days ({time.time() - t0:.0f}s)", flush=True)

    out_path = os.path.join(d, f"random_tick_pool59_s{a.seed}.parquet")
    tmp = out_path + ".tmp"
    writer = None
    total = 0
    for day in days:
        fname = day.replace("/", "-") + ".parquet"
        f = os.path.join(status_dir, fname)
        sub = pool.filter(pl.col("upload_day") == day).drop("upload_day")
        if not os.path.exists(f):
            print(f"  {day}: no status file, skipped ({sub.height:,} rows)", flush=True)
            continue
        G_st = FS.loadout_features(sub.select(["match_key", "round", "tick"]), status_dir=status_dir, files=[f])
        missing = [c for c in st_names if c not in G_st.columns]
        if missing:
            sys.exit(f"loadout columns missing: {missing[:10]}")
        out = sub.select(["match_key", "round", "tick", "draw_rank", "label_ct_win"] + BASIC).hstack(G_st.select(st_names))
        tab = out.to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(tmp, tab.schema, compression="zstd")
        writer.write_table(tab.cast(writer.schema))
        total += out.height
        print(f"  {day}: {out.height:,} rows (total {total:,}, {time.time() - t0:.0f}s)", flush=True)
        del sub, G_st, out, tab
    if writer is None:
        sys.exit("no rows written")
    writer.close()
    os.replace(tmp, out_path)
    print(f"done: {total:,} rows -> {out_path} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
