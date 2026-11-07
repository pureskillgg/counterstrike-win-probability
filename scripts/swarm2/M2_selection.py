"""M2 selection check: does the bake-off kept-set filter change map-level CT win rates?

For each era and map: rounds by filter status (from M2_build.py) and the CT win rate for each
status. Also the share of each status within the map, and the kept-set rate vs the raw rate.
Decomposes the kept-vs-raw rate gap into a part from dropped rounds.

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\M2_selection.py
"""

import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(40)
SCRATCH = Path("C:/Users/13144/AppData/Local/Temp/claude/C--Users-13144-Documents-pureskill/"
               "e0a18ad2-aa9e-4a3c-9946-29641b2cde82/scratchpad")
MAPS_KEEP = ["de_mirage", "de_dust2", "de_inferno", "de_ancient", "de_anubis", "de_nuke", "de_cache",
             "de_vertigo", "de_overpass"]

for era in ["2026", "2024"]:
    r = pl.read_parquet(SCRATCH / f"M2_rounds_{era}.parquet")
    r = r.filter(pl.col("winner").is_in([2, 3]))
    print(f"\n===== {era}: rounds with a winner {r.height}")
    tot = r.group_by("status").agg(pl.len().alias("rounds"), pl.col("label_ct_win").mean().alias("ct_rate")).sort("rounds", descending=True)
    print("overall by status:")
    print(tot)
    # per map: status shares and CT rate by status
    by = (r.group_by(["map_name", "status"])
          .agg(pl.len().alias("rounds"), pl.col("label_ct_win").mean().alias("ct_rate"))
          .sort(["map_name", "status"]))
    print("\nper map and status (rounds, CT rate):")
    for mp in MAPS_KEEP:
        sub = by.filter(pl.col("map_name") == mp)
        if sub.height == 0:
            continue
        total = int(sub["rounds"].sum())
        parts = "; ".join(f"{row['status']} {row['rounds']} ({row['rounds'] / total:.1%}) CT {row['ct_rate']:.3f}"
                          for row in sub.sort("rounds", descending=True).iter_rows(named=True))
        print(f"  {mp:<12} total {total}: {parts}")
    # kept vs raw per map, and the share of rounds lost
    raw = r.group_by("map_name").agg(pl.len().alias("raw_rounds"), pl.col("label_ct_win").mean().alias("raw_ct"))
    kept = (r.filter(pl.col("status") == "kept").group_by("map_name")
            .agg(pl.len().alias("kept_rounds"), pl.col("label_ct_win").mean().alias("kept_ct")))
    nk = (r.filter(pl.col("status") != "kept").group_by("map_name")
          .agg(pl.len().alias("dropped_rounds"), pl.col("label_ct_win").mean().alias("dropped_ct")))
    j = (raw.join(kept, on="map_name", how="left").join(nk, on="map_name", how="left")
         .with_columns((pl.col("kept_rounds") / pl.col("raw_rounds")).alias("share_kept"))
         .sort("raw_rounds", descending=True))
    print("\nraw vs kept CT rate by map:")
    print(j)
    # dropped rounds' CT rate by status, pooled over maps in this era
    print("\nCT rate by status, pooled across maps:")
    print(tot.select(["status", "rounds", "ct_rate"]))
