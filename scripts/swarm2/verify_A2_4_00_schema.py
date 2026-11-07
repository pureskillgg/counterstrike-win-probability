# verify_A2_4 step 00: schema probe for the 2024 tome header and one channel (read-only).
import os, json
import polars as pl
TOME = r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"
hdir = os.path.join(TOME, "header")
print("header folder files:", sorted(os.listdir(hdir)))
hm = json.load(open(os.path.join(hdir, "tome"), encoding="utf8"))
print("header tome pages:", len(hm.get("pages", [])), "keys:", sorted(hm.keys()))
pages = [p["dataframe"]["dataframe"] for p in hm["pages"]]
h = pl.concat([pl.read_parquet(os.path.join(hdir, p)) for p in pages], how="vertical_relaxed")
print("header rows:", h.height, "unique key:", h["key"].n_unique())
print("header schema:", h.schema)
print("header head:")
with pl.Config(tbl_cols=20, tbl_width_chars=250):
    print(h.head(3))
print("build_num counts:", h.group_by("build_num").len().sort("build_num").rows())
rs_dir = [x for x in os.listdir(TOME) if x.startswith("basic_round_state")][0]
rm = json.load(open(os.path.join(TOME, rs_dir, "tome"), encoding="utf8"))
print("round_state pages:", len(rm["pages"]))
f = pl.scan_parquet(os.path.join(TOME, rs_dir, rm["pages"][0]["dataframe"]["dataframe"])).head(3).collect()
print("round_state schema:", f.schema)
print("sample match_key:", f["match_key"][0])
