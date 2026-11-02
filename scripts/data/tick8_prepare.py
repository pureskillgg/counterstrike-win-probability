"""Prepare the 8-ticks-per-second data for the tick-rate experiment, incrementally by (match, tick).

Stages (run in order, each can be rerun):
  subset   the same 25% of the 2026 training matches as exp_tick_rates.py (rng seed 0 on the cache's training matches); writes
           data/combined/tick_subset_mk.parquet (mk_id, the grid's integer match id) and tick_subset_matches.parquet (match_key, mk_id).
  newgrid  the 8-per-second rows that the 4-per-second grid does not have (ticks 8 mod 16 after the freeze end), with the match key,
           upload day and era the status extractor needs; writes data/combined/grid_8new_extract.parquet. Existing ticks are never redone.
  merge    one status file per day in data/status_8hz: the 4-per-second rows already extracted (data/status_16hz) for these matches plus the
           new 8-per-second rows (data/status_8new). Each (match, round, tick, player) appears once; checked for duplicates.

Leakage guards: only training matches are extracted; the join to the grid is on exact (round, tick) (extract_status_1hz.one_match); the
new ticks and the old ticks are disjoint by construction (anti-join); the merge refuses duplicate (match, round, tick, player_id) rows.

  ...python.exe -I -X utf8 scripts\\data\\tick8_prepare.py --stage subset|newgrid|merge
"""
import argparse
import glob
import os
import sys

import numpy as np
import polars as pl

ROOT = r"G:\pureskill\csds-cs2"
sys.path.insert(0, os.path.join(ROOT, "scripts", "model"))
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.join(ROOT, "data", "combined")
FRAC = 0.25


def stage_subset():
    import wp_common as W
    cache = pl.scan_parquet(os.path.join(D, "cache_snapshot_2026.parquet")).select("match_key").unique().collect()
    kdf = cache.with_columns(pl.lit(1.0).alias("w_round"), pl.lit(0).alias("label_ct_win"), pl.lit("2026").alias("era"), pl.lit("").alias("match_date"))
    sp = W.make_split("random", kdf, seed=0)
    tr = sorted(sp["train"])
    rng = np.random.default_rng(0)  # the same first draw as exp_tick_rates.py
    S = sorted(rng.choice(np.array(tr), size=int(round(FRAC * len(tr))), replace=False).tolist())
    mk_map = pl.read_parquet(os.path.join(D, "grid_step16_matches.parquet"), columns=["match_key", "mk_id"])
    sub = mk_map.filter(pl.col("match_key").is_in(S))
    assert sub.height == len(S), "every subset match needs a grid id"
    sub.select("mk_id").write_parquet(os.path.join(D, "tick_subset_mk.parquet"))
    sub.write_parquet(os.path.join(D, "tick_subset_matches.parquet"))
    print(f"subset: {sub.height} training matches of {len(tr)} (rng seed 0, same as exp_tick_rates.py)", flush=True)


def stage_newgrid():
    g8 = pl.read_parquet(os.path.join(D, "grid_step8_sub.parquet"))
    g16 = (pl.read_parquet(os.path.join(D, "grid_step16.parquet"), columns=["match_key", "round", "tick", "era"])
           .filter(pl.col("era") == "2026").select(["match_key", "round", "tick"]))
    sub_ids = pl.read_parquet(os.path.join(D, "tick_subset_mk.parquet"))["mk_id"].to_list()
    g16 = g16.filter(pl.col("match_key").is_in(sub_ids))
    new = g8.join(g16, on=["match_key", "round", "tick"], how="anti")
    print(f"8-step rows {g8.height:,}; already extracted (4-step) {g8.height - new.height:,}; new {new.height:,}", flush=True)
    mapping = pl.read_parquet(os.path.join(D, "grid_step8_matches.parquet"), columns=["match_key", "mk_id"])
    upload = pl.read_parquet(os.path.join(D, "grid.parquet"), columns=["match_key", "upload_day"]).unique(subset=["match_key"])
    out = (new.rename({"match_key": "mk_id"}).join(mapping, on="mk_id", how="inner")
           .join(upload, on="match_key", how="left")
           .select(["match_key", "round", "tick", "second", pl.col("era").cast(pl.Utf8), "upload_day"]))
    assert out.get_column("upload_day").null_count() == 0, "every new row needs an upload day"
    out.write_parquet(os.path.join(D, "grid_8new_extract.parquet"))
    print(f"wrote {out.height:,} rows to grid_8new_extract.parquet", flush=True)


def stage_merge():
    s16 = os.path.join(ROOT, "data", "status_16hz")
    s8new = os.path.join(ROOT, "data", "status_8new")
    s8 = os.path.join(ROOT, "data", "status_8hz")
    os.makedirs(s8, exist_ok=True)
    sub = pl.read_parquet(os.path.join(D, "tick_subset_matches.parquet"))["match_key"].to_list()
    days = sorted(os.path.basename(f) for f in glob.glob(os.path.join(s16, "*.parquet")))
    total = 0
    for name in days:
        a = pl.read_parquet(os.path.join(s16, name)).filter(pl.col("match_key").is_in(sub))
        fb = os.path.join(s8new, name)
        if os.path.exists(fb):
            b = pl.read_parquet(fb).filter(pl.col("match_key").is_in(sub))
            both = pl.concat([a, b], how="vertical_relaxed")
        else:
            both = a
        dup = both.height - both.unique(["match_key", "round", "tick", "player_id", "side"]).height
        if dup:
            sys.exit(f"{name}: {dup} duplicate (match, round, tick, player) rows: refusing to merge")
        both.write_parquet(os.path.join(s8, name + ".tmp"), compression="zstd")
        os.replace(os.path.join(s8, name + ".tmp"), os.path.join(s8, name))
        total += both.height
    print(f"merged {len(days)} day files into data/status_8hz: {total:,} player-tick rows", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["subset", "newgrid", "merge"])
    a = ap.parse_args()
    {"subset": stage_subset, "newgrid": stage_newgrid, "merge": stage_merge}[a.stage]()
