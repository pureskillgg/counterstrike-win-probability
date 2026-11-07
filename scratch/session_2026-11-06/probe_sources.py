import json, collections, csv
import numpy as np
import polars as pl
FIN = "/home/sagemaker-user/adam/builder/final/"
EXP = FIN + "export_clean_v4/"
keys = json.load(open(EXP + "match_keys.json"))
rm = np.load(EXP + "round_match.npy").astype(np.int64)
rn = np.load(EXP + "round_num.npy").astype(np.int64)
rf = np.load(EXP + "round_freeze.npy").astype(np.int64)
print("export rounds", len(rm), "matches", len(keys))
print("clean_rounds columns:", pl.read_parquet_schema(FIN + "clean_rounds.parquet") if hasattr(pl, "read_parquet_schema") else pl.read_parquet(FIN + "clean_rounds.parquet", n_rows=2).columns)
cr = pl.read_parquet(FIN + "clean_rounds.parquet", columns=["match_key", "use", "src"])
print("clean_rounds rows", cr.height, "distinct match", cr["match_key"].n_unique())
print(cr.group_by(["src", "use"]).agg(pl.len().alias("rows"), pl.col("match_key").n_unique().alias("matches")).sort(["src", "use"]))
cm = pl.read_parquet(FIN + "clean_matches.parquet")
print("clean_matches columns", cm.columns, cm.height)
srcs = [c for c in cm.columns if c in ("src", "source", "use")]
for c in srcs:
    print(cm.group_by(c).len())
st = list(csv.DictReader(open(EXP + "match_loadout_status.csv")))
print("status cols", list(st[0].keys()) if st else None, len(st))
if st:
    ks = [k for k in st[0].keys() if "status" in k or "source" in k]
    for k in ks:
        print(k, collections.Counter(r[k] for r in st).most_common(12))
# per-match round counts
cnt = np.bincount(rm, minlength=len(keys))
print("rounds per match: min", cnt.min(), "max", cnt.max(), "mean", round(cnt.mean(), 2))
