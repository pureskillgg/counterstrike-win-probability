"""1 Hz living-player state for every used 2026 round, from the raw per-tick player_status channel (E: only, read-only).

For each match in data/<store>/grid.parquet (era 2026) the grid already fixes the snapshot ticks (freeze_end + 64 k, strictly
before the effective end). This reads E:\\csds_final\\csds\\<yyyy>\\<mm>\\<dd>\\<id>\\player_status once per match (column-
projected), keeps the rows at those ticks, takes each player's side from the round roster (player_info; the one player the
roster misses takes the side recorded in rounds.parquet missing_side) and writes one parquet per upload day:

    data/status_1hz/<yyyy-mm-dd>.parquet   (done days are skipped, so the job is resumable after a crash)

Rows are alive players only. Columns: match_key, round, tick, second, player_id, side, health, armor, has_helmet,
has_defuser, money, equipment_value_calc, is_defusing, inv_primary .. inv_c4.
Nothing is written under E:. Reads use a small thread pool (IO-bound) at below-normal priority.

  ...python.exe -I -X utf8 scripts\\data\\extract_status_1hz.py [--store combined] [--threads 3] [--wait-for-sync]
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
INV =["inv_primary", "inv_secondary", "inv_flashbang", "inv_smokegrenade", "inv_hegrenade", "inv_molotov",
       "inv_incgrenade", "inv_decoy", "inv_c4"]
STATUS_COLS = ["tick", "round", "player_id", "health", "armor", "has_helmet", "has_defuser", "money",
               "equipment_value_calc", "is_defusing"] + INV
OUT_COLS = ["match_key", "round", "tick", "second", "player_id", "side"] + STATUS_COLS[3:]
CASTS = {"round": pl.Int32, "tick": pl.Int64, "second": pl.Int32, "player_id": pl.Int32, "side": pl.Int8, "health": pl.Int16,
         "armor": pl.Int16, "has_helmet": pl.Boolean, "has_defuser": pl.Boolean, "money": pl.Int32,
         "equipment_value_calc": pl.Int32, "is_defusing": pl.Boolean, **{c: pl.Int16 for c in INV}}


def raw_path(mk):
    p = mk.split("/")  # csds/2026/10/08/<id>/csds  (2024: csds/2024/02/01/<id>/csds)
    return os.path.join(RAW_ROOT, p[1], p[2], p[3], p[4], "player_status")


def one_match(mk, grid_m, roster_m, missing_m):
    st = pl.from_arrow(pq.read_table(raw_path(mk), columns=STATUS_COLS))
    st = st.with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64), pl.col("player_id").cast(pl.Int64))
    st = st.unique(["round", "tick", "player_id"], keep="first", maintain_order=True)
    j = st.join(grid_m.select("round", "tick", "second"), on=["round", "tick"], how="inner")
    j = j.join(roster_m, on=["round", "player_id"], how="left")
    j = j.join(missing_m, on="round", how="left").with_columns(
        pl.coalesce([pl.col("side"), pl.col("missing_side")]).alias("side")).drop("missing_side")
    j = j.filter(pl.col("side").is_in([T, CT])).with_columns(pl.lit(mk).alias("match_key"))
    return j.select(OUT_COLS).cast(CASTS)


def sync_running():
    import subprocess
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq aws.exe"], capture_output=True, text=True).stdout
    return "aws.exe" in out


def main():
    global RAW_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="combined")
    ap.add_argument("--tomes", default=None)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--wait-for-sync", action="store_true")
    ap.add_argument("--max-days", type=int, default=0)
    ap.add_argument("--era", default="2026", help="grid era to extract (2026 default; 2024 for the staged February matches)")
    ap.add_argument("--raw-root", default=E_CSDS, help="folder holding <yyyy>/<mm>/<dd>/<match_id>/player_status (default: the E: 2026 tree)")
    ap.add_argument("--list-only", action="store_true", help="print the day list and output paths, read no raw data, write nothing")
    ap.add_argument("--grid", default="grid.parquet", help="grid file in data/<store>/ whose ticks are extracted (e.g. grid_offset32.parquet)")
    ap.add_argument("--out-dir", default=None, help="default data/status_1hz; use a different folder for an offset grid")
    ap.add_argument("--match-list", default=None, help="text file with one match_key per line: only these matches")
    args = ap.parse_args()
    tomes = args.tomes or args.store
    RAW_ROOT = args.raw_root
    if args.list_only:
        G0 = pl.read_parquet(os.path.join(ROOT, "data", args.store, args.grid), columns=["era", "upload_day"]).filter(pl.col("era") == args.era)
        out0 = args.out_dir or os.path.join(ROOT, "data", "status_1hz")
        for day in sorted(G0["upload_day"].unique().to_list()):
            print(day, "->", os.path.join(out0, day.replace("/", "-") + ".parquet"), flush=True)
        print(f"raw root: {RAW_ROOT}  era: {args.era}  out dir: {out0}", flush=True)
        return
    if args.wait_for_sync:
        while sync_running():
            print("aws sync still running; waiting", flush=True)
            time.sleep(120)
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL
    except Exception:
        pass
    d = os.path.join(ROOT, "data", args.store)
    out_dir = args.out_dir or os.path.join(ROOT, "data", "status_1hz")
    os.makedirs(out_dir, exist_ok=True)
    G = pl.read_parquet(os.path.join(d, args.grid), columns=["match_key", "round", "tick", "second", "era", "upload_day"]).filter(pl.col("era") == args.era)
    if args.match_list:
        keep = [ln.strip() for ln in open(args.match_list, encoding="utf-8") if ln.strip()]
        G = G.filter(pl.col("match_key").is_in(keep))
    R = pl.read_parquet(os.path.join(d, "rounds.parquet")).select("match_key", "round", "missing_side")
    pi_pages = sorted(glob.glob(os.path.join(ROOT, "tomes", tomes, "tome", "csds", "basic_player_info.COMBINED", "dataframe_*")))
    PI = (pl.concat([pl.read_parquet(f, columns=["match_key", "round", "player_id", "team_code"]) for f in pi_pages], how="vertical_relaxed")
          .unique(["match_key", "round", "player_id"], keep="last", maintain_order=True)
          .rename({"team_code": "side"}).with_columns(pl.col("round").cast(pl.Int64), pl.col("player_id").cast(pl.Int64)))
    # per-match lookups grouped once (filtering the whole table for every match was most of the time per match)
    PI_by = {k[0]: v.select("round", "player_id", "side") for k, v in PI.partition_by("match_key", as_dict=True).items()}
    R_by = {k[0]: v.select("round", "missing_side") for k, v in R.filter(pl.col("missing_side").is_not_null()).partition_by("match_key", as_dict=True).items()}
    PI_empty, R_empty = PI.head(0).select("round", "player_id", "side"), R.head(0).select("round", "missing_side")
    days = sorted(G["upload_day"].unique().to_list())
    if args.max_days:
        days = days[:args.max_days]
    print(f"{len(days)} upload days, {G['match_key'].n_unique()} matches", flush=True)
    t_all = time.time()
    for day in days:
        fn = os.path.join(out_dir, day.replace("/", "-") + ".parquet")
        gd = G.filter(pl.col("upload_day") == day)
        mks = gd["match_key"].unique().sort().to_list()
        # incremental: a day file that exists already holds some matches; only the matches NOT in it are extracted (new ones added to the day
        # later, and ones that failed before), so a failed or late match is never skipped forever
        prior = pl.read_parquet(fn) if os.path.exists(fn) else None
        if prior is not None:
            done = set(prior["match_key"].unique().to_list())
            mks = [m for m in mks if m not in done]
            if not mks:
                continue
        parts = gd.partition_by("match_key", as_dict=True)
        t0 = time.time()
        frames, failed = [], []

        def work(mk):
            g_m = parts[(mk,)]
            try:
                return one_match(mk, g_m, PI_by.get(mk, PI_empty), R_by.get(mk, R_empty))
            except Exception as exc:  # a missing/corrupt raw file must not stop the day
                return (mk, repr(exc)[:200])

        with ThreadPoolExecutor(args.threads) as ex:
            for r in ex.map(work, mks):
                (failed if isinstance(r, tuple) else frames).append(r)
        if frames:
            pl.concat(([prior] if prior is not None else []) + frames).write_parquet(fn + ".tmp", compression="zstd")
            os.replace(fn + ".tmp", fn)
        print(f"{day}: matches {len(mks)} ok {len(frames)} failed {len(failed)} rows {sum(f.height for f in frames)} "
              f"({time.time() - t0:.0f}s)", flush=True)
        for mk, why in failed[:3]:
            print("   FAILED", mk, why, flush=True)
    print(f"done in {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
