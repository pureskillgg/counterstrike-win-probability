"""D3-6 side check (read-only): is the 2024-02-07 UTC match_date cutover confounded with map?

Reads only the header table of the 2024-02-g tome (key, match_date, and the map column if present).
Cutover groups use the UTC match_date: pre = 2024-02-01..06, post = 2024-02-07 onward (ASSUMPTION:
the cutover date, which section 4 of verify_D3_6_crosscheck.py shows from the data).

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D3_6_mapcheck.py
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
TOME_G = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
files = sorted(glob.glob(str(TOME_G / "header" / "dataframe_*")))
schema = pl.read_parquet_schema(files[0])
print("header columns:", list(schema.keys()))
map_cols = [c for c in schema if "map" in c.lower()]
print("map-like columns:", map_cols)
cols = ["key", "match_date"] + map_cols[:1]
H = pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")
H = H.with_columns(
    pl.col("match_date").str.slice(0, 10).alias("md"),
    pl.when(pl.col("match_date").str.slice(0, 10) <= "2024-02-06").then(pl.lit("pre")).otherwise(pl.lit("post")).alias("cut"),
)
if map_cols:
    mc = map_cols[0]
    print(f"\nmatches by {mc} x cutover:")
    print(H.group_by([mc, "cut"]).len().sort([mc, "cut"]))
    tot = H.group_by("cut").len().sort("cut")
    print("\ncutover totals:")
    print(tot)
    print("\nmap share within each cutover group:")
    for cut in ["pre", "post"]:
        sub = H.filter(pl.col("cut") == cut)
        cnt = collections.Counter(sub[mc].to_list())
        n = sub.height
        print(f"  {cut} (n={n}): " + ", ".join(f"{m}={c} ({c / n:.1%})" for m, c in cnt.most_common()))
    print("\nmatches per map in pre and post (chi-square-style 2-way table printed above)")
else:
    print("no map column in the header table; cannot test map confounding from this tome")
