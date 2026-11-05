"""Port of scripts/data/extract_status_1hz.py to the SageMaker space (no data leaves AWS).

Reads the raw per-match player_status channel from a local sync of the S3 archive (csds/2026/10/<dd>/<id>/player_status)
and writes the same 1 Hz living-player table as the laptop script, one parquet per upload day:

    <out>/status_1hz/<yyyy-mm-dd>.parquet

Inputs copied to the space (keys and derived tables only, no PII):
    <home>/adam/status/data/combined/grid.parquet      the snapshot ticks (same grid the laptop used)
    <home>/adam/status/data/combined/rounds.parquet    missing_side per round

Roster (side per player per round) comes from the tome basic_player_info on the space, read with the bake-off's fv.read.
Matches whose raw player_status is not in the synced folder are reported and skipped, not guessed.

Usage (JupyterLab terminal, foreground):
    ~/venv-dsdk/bin/python extract_status_aws.py --days 2026/10/09 --limit 20     (test)
    ~/venv-dsdk/bin/python extract_status_aws.py                                  (all days in the grid that are in AWS)
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

HOME = Path.home()
sys.path.insert(0, str(HOME / "pipe" / "scripts" / "model"))
import feature_validation as fv  # noqa: E402  (fv.read for the tome roster)

T, CT = 2, 3
INV = ["inv_primary", "inv_secondary", "inv_flashbang", "inv_smokegrenade", "inv_hegrenade", "inv_molotov",
       "inv_incgrenade", "inv_decoy", "inv_c4"]
STATUS_COLS = ["tick", "round", "player_id", "health", "armor", "has_helmet", "has_defuser", "money",
               "equipment_value_calc", "is_defusing"] + INV
OUT_COLS = ["match_key", "round", "tick", "second", "player_id", "side"] + STATUS_COLS[3:]
CASTS = {"round": pl.Int32, "tick": pl.Int64, "second": pl.Int32, "player_id": pl.Int32, "side": pl.Int8,
         "health": pl.Int16, "armor": pl.Int16, "has_helmet": pl.Boolean, "has_defuser": pl.Boolean,
         "money": pl.Int32, "equipment_value_calc": pl.Int32, "is_defusing": pl.Boolean,
         **{c: pl.Int16 for c in INV}}
RAW_ROOT = Path("/mnt/sagemaker-nvme/raw/csds")


def raw_path(mk):
    p = mk.split("/")  # csds/2026/10/09/<id>/csds
    return RAW_ROOT / p[1] / p[2] / p[3] / p[4] / "player_status"


def one_match(mk, grid_m, roster_m, missing_m):
    st = pl.from_arrow(pq.read_table(raw_path(mk), columns=STATUS_COLS))
    st = st.with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64),
                         pl.col("player_id").cast(pl.Int64))
    st = st.unique(["round", "tick", "player_id"], keep="first", maintain_order=True)
    j = st.join(grid_m.select("round", "tick", "second"), on=["round", "tick"], how="inner")
    j = j.join(roster_m, on=["round", "player_id"], how="left")
    j = j.join(missing_m, on="round", how="left").with_columns(
        pl.coalesce([pl.col("side"), pl.col("missing_side")]).alias("side")).drop("missing_side")
    j = j.filter(pl.col("side").is_in([T, CT])).with_columns(pl.lit(mk).alias("match_key"))
    return j.select(OUT_COLS).cast(CASTS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", default="", help="comma list of upload days, e.g. 2026/10/09,2026/10/10 (default: all in AWS range)")
    ap.add_argument("--limit", type=int, default=0, help="test only: first N matches per day")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="/mnt/sagemaker-nvme/status_run")
    a = ap.parse_args()
    t0 = time.time()
    base = HOME / "adam" / "status" / "data" / "combined"
    out_dir = Path(a.out) / "status_1hz"
    out_dir.mkdir(parents=True, exist_ok=True)

    G = (pl.read_parquet(base / "grid.parquet", columns=["match_key", "round", "tick", "second", "era", "upload_day"])
         .filter(pl.col("era") == "2026"))
    if a.days:
        G = G.filter(pl.col("upload_day").is_in(a.days.split(",")))
    # AWS raw covers Oct 9 onward; days before are reported as not in AWS and skipped
    in_aws = G.filter(pl.col("upload_day") >= "2026/10/09")
    skipped = G.height - in_aws.height
    G = in_aws
    R = pl.read_parquet(base / "rounds.parquet").select("match_key", "round", "missing_side")
    PI = (fv.read("player_info", ["match_key", "round", "player_id", "team_code"])
          .unique(["match_key", "round", "player_id"], keep="last", maintain_order=True)
          .rename({"team_code": "side"})
          .with_columns(pl.col("round").cast(pl.Int64), pl.col("player_id").cast(pl.Int64)))
    PI_by = {k[0]: v.select("round", "player_id", "side") for k, v in PI.partition_by("match_key", as_dict=True).items()}
    R_by = {k[0]: v.select("round", "missing_side")
            for k, v in R.filter(pl.col("missing_side").is_not_null()).partition_by("match_key", as_dict=True).items()}
    PI_empty = PI.head(0).select("round", "player_id", "side")
    R_empty = R.head(0).select("round", "missing_side")
    print(f"days {G['upload_day'].n_unique()}, matches {G['match_key'].n_unique()}; skipped (not in AWS): {skipped} rows",
          flush=True)

    for day in sorted(G["upload_day"].unique().to_list()):
        fn = out_dir / (day.replace("/", "-") + ".parquet")
        gd = G.filter(pl.col("upload_day") == day)
        mks = gd["match_key"].unique().sort().to_list()
        if a.limit:
            mks = mks[: a.limit]
        parts = gd.partition_by("match_key", as_dict=True)
        ok, frames, missing_raw = 0, [], []
        t1 = time.time()

        def work(mk):
            if not raw_path(mk).exists():
                return ("missing", mk, "no raw player_status in the synced folder")
            try:
                return ("ok", mk, one_match(mk, parts[(mk,)], PI_by.get(mk, PI_empty), R_by.get(mk, R_empty)))
            except Exception as exc:  # a bad raw file must not stop the day
                return ("failed", mk, repr(exc)[:200])

        with ThreadPoolExecutor(a.threads) as ex:
            for status, mk, payload in ex.map(work, mks):
                if status == "ok":
                    ok += 1
                    frames.append(payload)
                else:
                    missing_raw.append((status, mk, payload))
        if frames:
            pl.concat(frames).write_parquet(str(fn) + ".tmp", compression="zstd")
            os.replace(str(fn) + ".tmp", fn)
        rows = sum(f.height for f in frames)
        print(f"{day}: matches {len(mks)} ok {ok} not-ok {len(missing_raw)} rows {rows} ({time.time() - t1:.0f}s)",
              flush=True)
        for status, mk, why in missing_raw[:3]:
            print(f"   {status.upper()} {mk} {why}", flush=True)
    print(f"done in {time.time() - t0:.0f}s; output {out_dir}", flush=True)


if __name__ == "__main__":
    main()
