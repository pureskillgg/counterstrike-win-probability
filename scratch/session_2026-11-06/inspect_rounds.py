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

keys = ["csds/2026/10/21/24193878-5c8d-429d-b536-e4af4a2a2538/csds",
        "csds/2026/10/17/78871936-76dd-4b02-b27c-f267edb977a4/csds",
        "csds/2026/10/21/38adf222-ba2e-4ee6-9cd3-1f99a1741e78/csds"]
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
