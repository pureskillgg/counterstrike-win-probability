"""M5 probe: do header fields (build, server, tick settings, metadata) differ between batches?

Read-only on the tome header. Prints value counts per batch for version-like and environment fields.
A game patch or server change on 2026-10-08 would show up here as a build_num or server split.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(60)

TOME = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")


def batch_label(d):
    if d == "2026-10-08":
        return "b1008"
    if d is not None and "2026-10-01" <= d <= "2026-10-07":
        return "b1001_1007"
    return "other"


def main():
    fs = sorted(glob.glob(str(TOME / "header" / "dataframe_*")))
    h = (pl.scan_parquet(fs)
         .select(["key", "match_date", "build_num", "tick_rate", "freeze_time", "round_time_defuse",
                  "server_name", "client_name", "demo_version_name", "game", "is_gotv_recording",
                  "max_unique_players", "network_protocol", "fullpackets_version", "rushb_version",
                  "ppp_version", "game_directory", "map_name", "is_wingman", "max_rounds",
                  "match_type", "platform"])
         .collect())
    h = h.with_columns(pl.col("match_date").str.slice(0, 10).map_elements(
        batch_label, return_dtype=pl.String).alias("batch"))
    print("matches by batch:", h.group_by("batch").agg(pl.len()).sort("batch").rows())
    for col in ["build_num", "tick_rate", "freeze_time", "round_time_defuse", "server_name",
                "client_name", "demo_version_name", "game", "is_gotv_recording", "max_unique_players",
                "network_protocol", "fullpackets_version", "rushb_version", "ppp_version",
                "game_directory", "is_wingman", "max_rounds", "match_type", "platform"]:
        tab = (h.group_by([col, "batch"]).agg(pl.len().alias("n"))
               .sort([col, "batch"]))
        n_levels = tab[col].n_unique()
        print(f"\n== {col} ({n_levels} levels) ==")
        if n_levels <= 12:
            print(tab.pivot(values="n", index=col, on="batch", aggregate_function="sum").sort(col))
        else:
            top = tab.sort("n", descending=True).head(12)
            print("top 12 (value, batch, n):")
            print(top)
    # build_num distribution by day
    dd = (h.with_columns(pl.col("match_date").str.slice(0, 10).alias("day"))
          .group_by(["day", "build_num"]).agg(pl.len().alias("n")).sort(["day", "build_num"]))
    print("\nbuild_num by day:")
    print(dd)


if __name__ == "__main__":
    main()
