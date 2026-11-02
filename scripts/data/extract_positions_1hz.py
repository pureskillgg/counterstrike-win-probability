"""1 Hz positions and view angles of the living players for every used 2026 round, from the raw per-tick player_vector channel (E: only, read-only).

Same method as extract_status_1hz.py: the grid fixes the snapshot ticks (freeze_end + 64 k); one column-projected read of player_vector per match,
rows kept at those ticks, side from the round roster (player_info; the one player the roster misses takes the side recorded in rounds.parquet missing_side,
else the vector's own team_code). Also writes every bomb_begin_plant event (position of the planter, site code) for bomb-site centres.

    data/positions_1hz/<yyyy-mm-dd>.parquet          match_key, round, tick, second, player_id, side, x, y, z, yaw      (alive players only)
    data/positions_1hz/_plants/<yyyy-mm-dd>.parquet  match_key, round, tick, site_code, x, y

yaw is theta_ang (horizontal view angle, degrees, 0 = +x, counter-clockwise). Done days are skipped; a day file that exists gets only the missing matches.
  ...python.exe -I -X utf8 scripts\\data\\extract_positions_1hz.py [--store combined] [--threads 4]
"""
import argparse
import glob
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
RAW_ROOT = E_CSDS  # set from --raw-root in main(); the 2026 default is the E: tree
T, CT = 2, 3
POS_COLS = ["tick", "round", "player_id", "x_pos", "y_pos", "z_pos", "theta_ang", "team_code"]
OUT_COLS = ["match_key", "round", "tick", "second", "player_id", "side", "x", "y", "z", "yaw"]
CASTS = {"round": pl.Int32, "tick": pl.Int64, "second": pl.Int32, "player_id": pl.Int32, "side": pl.Int8, "x": pl.Float32, "y": pl.Float32, "z": pl.Float32, "yaw": pl.Float32}


def raw_dir(mk):
    p = mk.split("/")  # csds/2026/10/08/<id>/csds  (2024: csds/2024/02/01/<id>/csds)
    return os.path.join(RAW_ROOT, p[1], p[2], p[3], p[4])


def one_match(mk, grid_m, roster_m, missing_m):
    d = raw_dir(mk)
    v = pl.from_arrow(pq.read_table(os.path.join(d, "player_vector"), columns=POS_COLS))
    v = v.with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64), pl.col("player_id").cast(pl.Int64))
    v = v.unique(["round", "tick", "player_id"], keep="first", maintain_order=True)
    j = v.join(grid_m.select("round", "tick", "second"), on=["round", "tick"], how="inner")
    j = j.join(roster_m, on=["round", "player_id"], how="left")
    j = j.join(missing_m, on="round", how="left").with_columns(pl.coalesce([pl.col("side"), pl.col("missing_side"), pl.col("team_code")]).alias("side")).drop("missing_side")
    j = j.filter(pl.col("side").is_in([T, CT])).with_columns(pl.lit(mk).alias("match_key"))
    j = j.rename({"x_pos": "x", "y_pos": "y", "z_pos": "z", "theta_ang": "yaw"}).select(OUT_COLS).cast(CASTS)
    ba = pl.from_arrow(pq.read_table(os.path.join(d, "bomb_action"), columns=["round", "tick", "event_type", "site_code", "player_x_pos", "player_y_pos"]))
    pl_ = (ba.filter(pl.col("event_type") == "bomb_begin_plant").with_columns(pl.lit(mk).alias("match_key"))
           .rename({"player_x_pos": "x", "player_y_pos": "y"}).select("match_key", "round", "tick", "site_code", "x", "y"))
    return j, pl_


def main():
    global RAW_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="combined")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-days", type=int, default=0)
    ap.add_argument("--era", default="2026", help="grid era to extract (2026 default; 2024 for the staged February matches)")
    ap.add_argument("--raw-root", default=E_CSDS, help="folder holding <yyyy>/<mm>/<dd>/<match_id>/player_vector (default: the E: 2026 tree)")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "data", "positions_1hz"), help="output folder (2024 runs must use a new folder)")
    ap.add_argument("--list-only", action="store_true", help="print the day list and output paths, read no raw data, write nothing")
    args = ap.parse_args()
    RAW_ROOT = args.raw_root
    d = os.path.join(ROOT, "data", args.store)
    out_dir = args.out_dir
    G = pl.read_parquet(os.path.join(d, "grid.parquet"), columns=["match_key", "round", "tick", "second", "era", "upload_day"]).filter(pl.col("era") == args.era)
    if args.list_only:
        for day in sorted(G["upload_day"].unique().to_list()):
            print(day, "->", os.path.join(out_dir, day.replace("/", "-") + ".parquet"), flush=True)
        print(f"raw root: {RAW_ROOT}  era: {args.era}  out dir: {out_dir}", flush=True)
        return
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL
    except Exception:
        pass
    os.makedirs(os.path.join(out_dir, "_plants"), exist_ok=True)
    R = pl.read_parquet(os.path.join(d, "rounds.parquet")).select("match_key", "round", "missing_side")
    pi_pages = sorted(glob.glob(os.path.join(ROOT, "tomes", args.store, "tome", "csds", "basic_player_info.COMBINED", "dataframe_*")))
    PI = (pl.concat([pl.read_parquet(f, columns=["match_key", "round", "player_id", "team_code"]) for f in pi_pages], how="vertical_relaxed")
          .unique(["match_key", "round", "player_id"], keep="last", maintain_order=True)
          .rename({"team_code": "side"}).with_columns(pl.col("round").cast(pl.Int64), pl.col("player_id").cast(pl.Int64)))
    days = sorted(G["upload_day"].unique().to_list())
    if args.max_days:
        days = days[:args.max_days]
    print(f"{len(days)} upload days, {G['match_key'].n_unique()} matches", flush=True)
    t_all = time.time()
    for day in days:
        fn = os.path.join(out_dir, day.replace("/", "-") + ".parquet")
        fp = os.path.join(out_dir, "_plants", day.replace("/", "-") + ".parquet")
        gd = G.filter(pl.col("upload_day") == day)
        mks = gd["match_key"].unique().sort().to_list()
        prior = pl.read_parquet(fn) if os.path.exists(fn) else None
        prior_p = pl.read_parquet(fp) if os.path.exists(fp) else None
        if prior is not None:
            done = set(prior["match_key"].unique().to_list())
            mks = [m for m in mks if m not in done]
            if not mks:
                continue
        parts = gd.partition_by("match_key", as_dict=True)
        t0 = time.time()
        frames, plants, failed = [], [], []

        def work(mk):
            try:
                return one_match(mk, parts[(mk,)], PI.filter(pl.col("match_key") == mk).select("round", "player_id", "side"),
                                 R.filter((pl.col("match_key") == mk) & pl.col("missing_side").is_not_null()).select("round", "missing_side"))
            except Exception as exc:  # a missing/corrupt raw file must not stop the day
                return (mk, repr(exc)[:200])

        with ThreadPoolExecutor(args.threads) as ex:
            for r in ex.map(work, mks):
                if isinstance(r[0], str):
                    failed.append(r)
                else:
                    frames.append(r[0])
                    plants.append(r[1])
        if frames:
            pl.concat(([prior] if prior is not None else []) + frames).write_parquet(fn + ".tmp", compression="zstd")
            os.replace(fn + ".tmp", fn)
            pl.concat(([prior_p] if prior_p is not None else []) + plants, how="vertical_relaxed").write_parquet(fp + ".tmp", compression="zstd")
            os.replace(fp + ".tmp", fp)
        print(f"{day}: matches {len(mks)} ok {len(frames)} failed {len(failed)} rows {sum(f.height for f in frames)} ({time.time() - t0:.0f}s)", flush=True)
        for mk, why in failed[:3]:
            print("   FAILED", mk, why, flush=True)
    print(f"done in {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
