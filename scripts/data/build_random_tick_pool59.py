"""The 59-input random-tick pool: for each seed, the 2026 rows of random_tick_pool_s<seed>.parquet (the 13 event inputs at each drawn tick)
joined with the 46 loadout inputs computed from the per-tick player status at those ticks (data/status_random, extracted by
extract_status_1hz.py --grid random_tick_extract.parquet). Loadout aggregates come from features_status.loadout_features, the same function
that builds the snapshot cache's loadout columns.

  ...python.exe -I -X utf8 scripts\\data\\build_random_tick_pool59.py --seed 0 [--store combined]
Writes data/<store>/random_tick_pool59_s<seed>.parquet.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
import polars as pl

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
    status_dir = a.status_dir or os.path.join(d, "status_random")
    t0 = time.time()
    feats59 = json.load(open(os.path.join(W.ROOT, "models", "winprob_v3.meta.json"), encoding="utf-8"))["features"]
    st_names = [f for f in feats59 if f not in BASIC]
    pool = pl.read_parquet(os.path.join(d, f"random_tick_pool_s{a.seed}.parquet")).filter(pl.col("era") == "2026")
    print(f"2026 random-tick rows, seed {a.seed}: {pool.height:,} ({time.time() - t0:.0f}s)", flush=True)
    G_st = FS.loadout_features(pool.select(["match_key", "round", "tick"]), status_dir=status_dir)
    missing = [c for c in st_names if c not in G_st.columns]
    if missing:
        sys.exit(f"loadout columns missing: {missing[:10]}")
    print(f"loadout columns: {G_st.width} (expected {len(st_names)}); rows {G_st.height:,} ({time.time() - t0:.0f}s)", flush=True)
    out = pool.select(["match_key", "round", "tick", "draw_rank", "label_ct_win"] + BASIC).hstack(G_st.select(st_names))
    out_path = os.path.join(d, f"random_tick_pool59_s{a.seed}.parquet")
    out.write_parquet(out_path + ".tmp", compression="zstd")
    os.replace(out_path + ".tmp", out_path)
    nulls = {c: int(out[c].is_null().sum()) for c in st_names if out[c].is_null().sum()}
    print(f"done: {out.height:,} rows, columns {out.width}, null loadout values {len(nulls)} columns ({time.time() - t0:.0f}s) -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
