"""M4 final checks: per-map data loss (is map a proxy for dropped matches?) and the full batch-by-map table.

Header counts from the 2026 tome; kept matches and rounds from scratchpad rounds_2026.parquet (the
bake-off's kept set, validated in M4_build.py). Read-only on the tome.
"""
import glob
import os

import polars as pl

SCR = (r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
       r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\M4")
T2026 = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"


def main():
    files = sorted(glob.glob(os.path.join(T2026, "header", "dataframe_*")))
    H = pl.concat([pl.read_parquet(f, columns=["key", "map_name", "match_date"]) for f in files],
                  how="vertical_relaxed")
    tot = H.group_by("map_name").agg(pl.len().alias("header_matches")).sort("map_name")
    R = pl.read_parquet(os.path.join(SCR, "rounds_2026.parquet"))
    kept = R.group_by("map_name").agg(pl.col("match_key").n_unique().alias("kept_matches"),
                                      pl.len().alias("kept_rounds"))
    j = tot.join(kept, on="map_name", how="left").with_columns(
        pl.col("kept_matches").fill_null(0), pl.col("kept_rounds").fill_null(0)).with_columns(
        (1 - pl.col("kept_matches") / pl.col("header_matches")).round(3).alias("drop_share"))
    print("per-map matches: header vs kept (bake-off kept set)")
    print(j.sort("header_matches", descending=True))
    print(f"\nheader matches {H.height}; kept {R['match_key'].n_unique()}; "
          f"overall drop {1 - R['match_key'].n_unique() / H.height:.3f}")
    print("\nCT win rate by map and batch (1008 = match_date 2026-10-08), kept rounds")
    bt = (R.group_by(["map_name", "batch_1008"]).agg(pl.len().alias("n"), pl.col("ct_win").mean().round(3).alias("ct"))
          .sort(["map_name", "batch_1008"]))
    for r in bt.iter_rows(named=True):
        print(f"  {r['map_name']:<11} batch_1008={str(r['batch_1008']):<5} n={r['n']:>5} CT={r['ct']:.3f}")
    print("\nbatch share of rounds by map:")
    sh = R.group_by("map_name").agg(pl.col("batch_1008").mean().round(3).alias("share_1008"),
                                    pl.len().alias("n")).sort("n", descending=True)
    print(sh)


if __name__ == "__main__":
    main()
