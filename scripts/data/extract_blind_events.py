"""Every flashbang blinding event of every used 2026 match, from the raw player_blind channel (E: only, read-only).

  data/blind_events/<yyyy-mm-dd>.parquet   match_key, round, tick, player_id, attacker_id, blind_duration
blind_duration (seconds) is how long that flash blinds the player (0.08 to about 5.2 s in the data: a weak flash vs a full one); player_status.flash_duration holds the
same value for the whole effect, so the remaining time at a later tick is  tick_event + 64 * blind_duration - tick.  Match-level incremental: a day file that exists gets only the matches not read before (a list of read matches is kept beside it).
  ...python.exe -I -X utf8 scripts\\data\\extract_blind_events.py [--store combined] [--threads 4]
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import polars as pl
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"G:\pureskill\csds-cs2"
E_CSDS = r"E:\csds_final\csds"
COLS = ["round", "tick", "player_id", "attacker_id", "blind_duration"]


def one(mk):
    p = mk.split("/")
    t = pl.from_arrow(pq.read_table(os.path.join(E_CSDS, p[1], p[2], p[3], p[4], "player_blind"), columns=COLS))
    return t.with_columns(pl.lit(mk).alias("match_key"), pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64), pl.col("player_id").cast(pl.Int64),
                          pl.col("attacker_id").cast(pl.Int64), pl.col("blind_duration").cast(pl.Float32)).select("match_key", *COLS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="combined")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    out_dir = os.path.join(ROOT, "data", "blind_events")
    os.makedirs(out_dir, exist_ok=True)
    G = pl.read_parquet(os.path.join(ROOT, "data", a.store, "grid.parquet"), columns=["match_key", "era", "upload_day"]).filter(pl.col("era") == "2026").unique("match_key")
    t_all = time.time()
    for day in sorted(G["upload_day"].unique().to_list()):
        fn = os.path.join(out_dir, day.replace("/", "-") + ".parquet")
        fm = fn + ".matches.txt"  # every match already read for this day (also those without any blind event)
        mks = G.filter(pl.col("upload_day") == day)["match_key"].sort().to_list()
        seen = set(open(fm, encoding="utf-8").read().split()) if os.path.exists(fm) else set()
        if not seen and os.path.exists(fn):  # a file from the first version of this script: its matches count as read
            seen = set(pl.read_parquet(fn, columns=["match_key"])["match_key"].unique().to_list())
        prior = pl.read_parquet(fn) if os.path.exists(fn) else None
        mks = [m for m in mks if m not in seen]
        if not mks:
            continue
        frames, failed = [], []

        def work(mk):
            try:
                return one(mk)
            except Exception as exc:  # noqa: BLE001
                return (mk, repr(exc)[:160])

        with ThreadPoolExecutor(a.threads) as ex:
            for r in ex.map(work, mks):
                (failed if isinstance(r, tuple) else frames).append(r)
        if frames:
            pl.concat(([prior] if prior is not None else []) + frames).write_parquet(fn + ".tmp", compression="zstd")
            os.replace(fn + ".tmp", fn)
            ok = {m for f in frames for m in f["match_key"].unique().to_list()} | (set(mks) - {m for m, _ in failed})
            with open(fm + ".tmp", "w", encoding="utf-8") as fh:
                fh.write(chr(10).join(sorted(seen | ok)))
            os.replace(fm + ".tmp", fm)
        print(f"{day}: matches {len(mks)} ok {len(frames)} failed {len(failed)} events {sum(f.height for f in frames)}", flush=True)
        for mk, why in failed[:3]:
            print("   FAILED", mk, why, flush=True)
    print(f"done in {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
