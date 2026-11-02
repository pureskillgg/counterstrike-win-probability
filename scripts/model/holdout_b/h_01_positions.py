"""S2 step 1: 1 Hz positions of the 653 holdout matches only (raw player_vector, E: read-only).

extract_positions_1hz.py has no match-list option (it loops over every match of a day), so this driver imports its one_match() and
repeats its roster / missing-side / bomb-plant set-up, restricted to the holdout keys. Output: data/holdout_b/positions_1hz/<day>.parquet
(same columns as data/positions_1hz). Nothing under data/positions_1hz, data/combined, tomes/ or E: is written.
"""
import glob
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import polars as pl

ROOT = r"G:\pureskill\csds-cs2"
sys.path.insert(0, os.path.join(ROOT, "scripts", "data"))
import extract_positions_1hz as EP  # noqa: E402

EP.RAW_ROOT = EP.E_CSDS
HB = os.path.join(ROOT, "data", "holdout_b")
OUT = os.path.join(HB, "positions_1hz")
THREADS = 4
H = [ln.strip() for ln in open(os.path.join(HB, "holdout_keys.txt"), encoding="utf-8") if ln.strip()]
assert len(H) == 653 and len(set(H)) == 653
Hs = set(H)
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
G = pl.read_parquet(os.path.join(HB, "grid.parquet"), columns=["match_key", "round", "tick", "second", "era", "upload_day"]).filter(pl.col("era") == "2026")
R = pl.read_parquet(os.path.join(HB, "rounds.parquet"), columns=["match_key", "round", "missing_side"])
pages = sorted(glob.glob(os.path.join(ROOT, "tomes", "combined", "tome", "csds", "basic_player_info.COMBINED", "dataframe_*")))
PI = (pl.concat([pl.read_parquet(f, columns=["match_key", "round", "player_id", "team_code"]).filter(pl.col("match_key").is_in(H)) for f in pages],
                how="vertical_relaxed")
      .unique(["match_key", "round", "player_id"], keep="last", maintain_order=True)
      .rename({"team_code": "side"}).with_columns(pl.col("round").cast(pl.Int64), pl.col("player_id").cast(pl.Int64)))
PI_by = {k[0]: v.select("round", "player_id", "side") for k, v in PI.partition_by("match_key", as_dict=True).items()}
R_by = {k[0]: v.select("round", "missing_side") for k, v in R.filter(pl.col("missing_side").is_not_null()).partition_by("match_key", as_dict=True).items()}
PI_empty, R_empty = PI.head(0).select("round", "player_id", "side"), R.head(0).select("round", "missing_side")
print(f"grid rows {G.height:,} matches {G['match_key'].n_unique()} roster matches {len(PI_by)} defect rounds {len(R_by)} ({time.time()-t0:.0f}s)", flush=True)
days = sorted(G["upload_day"].unique().to_list())
print("upload days:", days, flush=True)
tot_rows = 0
for day in days:
    fn = os.path.join(OUT, day.replace("/", "-") + ".parquet")
    gd = G.filter(pl.col("upload_day") == day)
    mks = gd["match_key"].unique().sort().to_list()
    parts = gd.partition_by("match_key", as_dict=True)
    t1 = time.time()
    frames, plants, failed = [], [], []

    def work(mk):
        try:
            return EP.one_match(mk, parts[(mk,)], PI_by.get(mk, PI_empty), R_by.get(mk, R_empty))
        except Exception as exc:  # a missing/corrupt raw file must not stop the day; it is listed
            return (mk, repr(exc)[:200])

    with ThreadPoolExecutor(THREADS) as ex:
        for r in ex.map(work, mks):
            if isinstance(r[0], str):
                failed.append(r)
            else:
                frames.append(r[0])
    if frames:
        pl.concat(frames).write_parquet(fn + ".tmp", compression="zstd")
        os.replace(fn + ".tmp", fn)
    rows = sum(f.height for f in frames)
    tot_rows += rows
    print(f"{day}: matches {len(mks)} ok {len(frames)} failed {len(failed)} rows {rows:,} ({time.time()-t1:.0f}s)", flush=True)
    for mk, why in failed[:5]:
        print("   FAILED", mk, why, flush=True)
    if failed:
        raise SystemExit(f"{day}: {len(failed)} matches failed; stopping before the next step")
got = pl.concat([pl.read_parquet(f, columns=["match_key"]) for f in sorted(glob.glob(os.path.join(OUT, "*.parquet")))])
print(f"positions written: rows {tot_rows:,}; matches with positions {got['match_key'].n_unique()} of {len(H)}; total {time.time()-t0:.0f}s", flush=True)
