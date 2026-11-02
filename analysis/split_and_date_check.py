import json
import polars as pl

ROOT = "G:/pureskill/csds-cs2"
SCR = "C:/Users/13144/AppData/Local/Temp/claude/C--Users-13144-Documents-pureskill/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad"
frozen = set(json.load(open(SCR + "/frozen_test_keys.json")))
man = json.load(open(ROOT + "/data/combined/split_manifest.json"))["assign"]

idx = pl.read_parquet(ROOT + "/data/gnn/index.parquet", columns=["match_key", "upload_day"]).unique()
print("index matches:", idx.height)
idx = idx.with_columns(
    pl.col("match_key").map_elements(lambda k: "frozen_test" if k in frozen else man.get(k, "unassigned"), return_dtype=pl.Utf8).alias("split"))
print(idx.group_by(["upload_day", "split"]).len().sort(["upload_day", "split"]).filter(pl.col("upload_day") >= "2026/10/09").to_pandas().to_string() if False else "")
g = idx.filter(pl.col("upload_day") >= "2026/10/09").group_by(["upload_day", "split"]).len().sort(["upload_day", "split"])
for r in g.iter_rows(named=True):
    print(r["upload_day"], r["split"], r["len"])
print("frozen test matches by upload_day:")
ft = idx.filter(pl.col("split") == "frozen_test").group_by("upload_day").len().sort("upload_day")
print(ft.tail(8))
print("frozen test min/max day:", ft["upload_day"].min(), ft["upload_day"].max())
print("Oct 14-25 matches by split:")
print(idx.filter((pl.col("upload_day") >= "2026/10/14") & (pl.col("upload_day") <= "2026/10/25")).group_by("split").len())
