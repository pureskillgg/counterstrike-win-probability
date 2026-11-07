# verify A2-4 (round-1 missing in build 9842): schema exploration only.
import os, json
import polars as pl

TOME = r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"


def channel_dir(ch):
    d = [x for x in os.listdir(TOME) if x.split(".", 1)[0] == "basic_" + ch]
    assert len(d) == 1, (ch, d)
    return os.path.join(TOME, d[0])


def pages(ch):
    dp = channel_dir(ch)
    meta = json.load(open(os.path.join(dp, "tome"), encoding="utf8"))
    return dp, [os.path.join(dp, p["dataframe"]["dataframe"]) for p in meta["pages"]]


def schema_of(ch):
    dp, files = pages(ch)
    lf = pl.scan_parquet(files[0])
    return lf.collect_schema(), len(files), files[0]


hdr_dir = os.path.join(TOME, "header")
hdr_files = [os.path.join(hdr_dir, "dataframe_00000")]
hs = pl.scan_parquet(hdr_files[0])
print("HEADER schema:", hs.collect_schema(), flush=True)
hdf = hs.head(3).collect()
print("HEADER head(3):", hdf, flush=True)
print("HEADER rows:", hs.select(pl.len()).collect().item(), flush=True)

for ch in ["round_state", "round_start", "round_end"]:
    sch, n, f = schema_of(ch)
    print(ch, "pages:", n, "schema:", sch, flush=True)
