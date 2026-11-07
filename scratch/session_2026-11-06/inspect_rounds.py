import polars as pl
import glob

R = "/home/sagemaker-user/adam/status/data/combined/rounds.parquet"
r = pl.read_parquet(R)
print("rounds shape", r.shape)
print(r.schema)
print(r.head(3))
for c in r.columns:
    if c in ("use",):
        print("use value counts:", r.group_by("use").len())

keys = ["csds/2026/10/21/m_47360c7367d0788ec8c5e842/csds",
        "csds/2026/10/17/m_1caf727ba135e50429bfabf7/csds",
        "csds/2026/10/21/m_72b83f4542327f39379d7253/csds"]
if "match_key" in r.columns:
    for k in keys:
        sub = r.filter(pl.col("match_key") == k)
        print(k, "rounds rows:", sub.shape[0])
        if sub.shape[0]:
            print(sub.select([c for c in sub.columns if c in ("round", "use", "match_key")]).sort("round").head(40))

# scan tome pages for these match keys, lazily reading only match_key
pages = sorted(glob.glob("/mnt/sagemaker-nvme/parity/c*/dataframe_*"))
print("n pages", len(pages))
for k in keys:
    hits = []
    for p in pages:
        n = pl.scan_parquet(p).filter(pl.col("match_key") == k).select(pl.len()).collect().item()
        if n:
            hits.append((p, n))
    print(k, hits)
