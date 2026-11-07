"""M5 probe: cross-tab of batch with build, server source, platform, rushb/ppp versions.

Purpose: decide which batch proxy to control for. Batch (match_date day) is nearly collinear with
the build number and the hosting source. This prints the joint counts at match level and the
round-level CT rate of each cell (clean 5v5 rounds, after the duplicate removal in M5_load.py).
Read-only on the tome header; reuses the scratchpad round table written by M5_load.py.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(50)

TOME = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")
SCRATCH = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\M5")
RPQ = SCRATCH / "M5_rounds.parquet"


def main():
    fs = sorted(glob.glob(str(TOME / "header" / "dataframe_*")))
    h = (pl.scan_parquet(fs)
         .select(["key", "server_name", "build_num", "rushb_version", "ppp_version", "game_directory",
                  "platform", "match_type", "map_name", "match_date", "is_wingman", "is_gotv_recording"])
         .collect()
         .rename({"key": "match_key"}))
    h = h.with_columns(
        pl.col("server_name").str.starts_with("FACEIT").alias("faceit_server"),
        pl.col("game_directory").str.starts_with("/home").alias("home_dir_path"),
    )
    r = pl.read_parquet(RPQ).select(["match_key", "batch", "ct_win", "map_name"])
    m = (r.group_by("match_key").agg(pl.col("batch").first()))
    hm = h.join(m, on="match_key", how="inner")
    print("kept (deduped, clean 5v5) matches by batch:", hm.group_by("batch").agg(pl.len()).sort("batch").rows())

    print("\n== batch x build_num (matches) ==")
    print(hm.group_by(["batch", "build_num"]).agg(pl.len().alias("matches")).sort(["batch", "build_num"]))
    print("\n== batch x faceit_server (matches) ==")
    print(hm.group_by(["batch", "faceit_server"]).agg(pl.len().alias("matches")).sort(["batch", "faceit_server"]))
    print("\n== batch x platform (matches) ==")
    print(hm.group_by(["batch", "platform"]).agg(pl.len().alias("matches")).sort(["batch", "platform"]))
    print("\n== batch x match_type (matches) ==")
    print(hm.group_by(["batch", "match_type"]).agg(pl.len().alias("matches")).sort(["batch", "match_type"]))
    print("\n== batch x rushb_version x ppp_version (matches) ==")
    print(hm.group_by(["batch", "rushb_version", "ppp_version"]).agg(pl.len().alias("matches"))
          .sort(["batch", "rushb_version", "ppp_version"]))
    print("\n== batch x home_dir_path (matches) ==")
    print(hm.group_by(["batch", "home_dir_path"]).agg(pl.len().alias("matches")).sort(["batch", "home_dir_path"]))

    print("\n== faceit_server x build_num (matches, all batches) ==")
    print(hm.group_by(["faceit_server", "build_num"]).agg(pl.len().alias("matches")).sort(["faceit_server", "build_num"]))
    print("\n== faceit_server x platform (matches, all batches) ==")
    print(hm.group_by(["faceit_server", "platform"]).agg(pl.len().alias("matches")).sort(["faceit_server", "platform"]))

    # round-level CT rate per cell of batch x build x faceit_server
    rr = r.join(h.select(["match_key", "build_num", "faceit_server", "platform"]), on="match_key", how="left")
    print("\n== round-level CT rate: batch x build_num x faceit_server ==")
    cells = (rr.group_by(["batch", "build_num", "faceit_server"])
             .agg(pl.len().alias("rounds"), pl.col("match_key").n_unique().alias("matches"),
                  (pl.col("ct_win").mean() * 100).round(2).alias("ct_pct"))
             .sort(["batch", "build_num", "faceit_server"]))
    print(cells)
    print("\n== round-level CT rate: faceit_server x platform (all batches) ==")
    print(rr.group_by(["faceit_server", "platform"]).agg(pl.len().alias("rounds"),
          pl.col("match_key").n_unique().alias("matches"),
          (pl.col("ct_win").mean() * 100).round(2).alias("ct_pct")).sort(["faceit_server", "platform"]))


if __name__ == "__main__":
    main()
