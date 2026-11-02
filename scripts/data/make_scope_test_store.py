"""Build data/scopetest: the matches the model was NOT trained for, as a store the normal grid / extraction code can read.

  ...python.exe -I -X utf8 scripts\\data\\make_scope_test_store.py [--reason map_out_of_pool]

Takes matches.parquet / rounds.parquet / deaths.parquet of data/combined and keeps the matches whose scope_reason is --reason (default: a 5v5 competitive match on a map that
is not one of the seven the model knows: vertigo, overpass, train, ...). Round rule = the normal `use` rule WITHOUT the scope condition
(round_status ok, duplicate keeper, channels complete). Then run, in this order (each writes under data/scopetest or data/status_1hz_scopetest, nothing in data/combined):
  make_grid.py --store scopetest
  extract_status_1hz.py --store scopetest --tomes combined --out-dir data\\status_1hz_scopetest
  eval_scope_maps.py
"""
import argparse
import os
import sys

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"G:\pureskill\csds-cs2"

ap = argparse.ArgumentParser()
ap.add_argument("--reason", default="map_out_of_pool")
a = ap.parse_args()
src, dst = os.path.join(ROOT, "data", "combined"), os.path.join(ROOT, "data", "scopetest")
os.makedirs(dst, exist_ok=True)
M = pl.read_parquet(os.path.join(src, "matches.parquet"))
sel = M.filter((pl.col("scope_reason") == a.reason) & pl.col("dup_keeper") & pl.col("channels_complete"))
keys = sel["match_key"].to_list()
R = pl.read_parquet(os.path.join(src, "rounds.parquet")).filter(pl.col("match_key").is_in(keys))
R = R.with_columns((pl.col("round_status").eq("ok") & pl.col("dup_keeper") & pl.col("channels_complete")).alias("use"))
D = pl.read_parquet(os.path.join(src, "deaths.parquet")).filter(pl.col("match_key").is_in(keys))
sel.write_parquet(os.path.join(dst, "matches.parquet"))
R.write_parquet(os.path.join(dst, "rounds.parquet"))
D.write_parquet(os.path.join(dst, "deaths.parquet"))
print(f"scope '{a.reason}': {sel.height} matches ({dict(sel.group_by('era').len().iter_rows())}), maps {dict(sel.group_by('map_name').len().sort('len', descending=True).iter_rows())}")
print(f"rounds {R.height:,}, usable {int(R['use'].sum()):,}; deaths {D.height:,}")
