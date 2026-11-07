import glob
import hashlib
import json
import os

import polars as pl

SCR = "C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad"
keys = json.load(open(SCR + "/export_match_keys.json"))
PM = "E:/pureskill_tomes/pertick_status_1001_1008/per_match"
files = set(os.listdir(PM))
have = [k for k in keys if hashlib.sha1(k.encode("utf-8")).hexdigest()[:16] + ".parquet" in files]
print("export keys", len(keys), "| per_match files on E:", len(files), "| export keys with a per_match file", len(have))
tome = "G:/pureskill/csds-cs2/tomes/2026-10-01_to_08/tome/csds"
hdr = pl.concat([pl.read_parquet(f, columns=["key"]) for f in sorted(glob.glob(tome + "/header/dataframe_*"))])
tk = set(hdr["key"].to_list())
print("tome header keys", len(tk), "| export keys with per_match file that are in tome", len([k for k in have if k in tk]))
one = have[0]
p = os.path.join(PM, hashlib.sha1(one.encode("utf-8")).hexdigest()[:16] + ".parquet")
df = pl.read_parquet(p)
print("sample file", one, "rows", df.height)
print("columns", df.columns)
print("dtypes", {c: str(t) for c, t in zip(df.columns, df.dtypes)})
if "health" in df.columns:
    print("rows with health 0", df.filter(pl.col("health") == 0).height, "| health null", df["health"].null_count())
    print("health distribution (first 10 by count)", df["health"].value_counts().sort("count", descending=True).head(10).rows())
print("distinct ticks", df["tick"].n_unique() if "tick" in df.columns else None, "| tick mod 64 counts", df.with_columns((pl.col("tick") % 64).alias("m"))["m"].value_counts().sort("count", descending=True).head(5).rows() if "tick" in df.columns else None)
if "match_key" in df.columns:
    print("match_key rows unique", df["match_key"].n_unique())
json.dump(have, open(SCR + "/have_keys.json", "w"))
print("saved have_keys", len(have))
