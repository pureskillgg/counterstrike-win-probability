"""How much wider is the match-cluster standard error of the mean log loss than the naive one that treats every second as independent?
  ...python.exe -I -X utf8 scripts\\viz\\cluster_se.py --pred <test predictions parquet> --out data\\combined\\cluster_se_<model>.json
Per-second loss l_i = -(y log p + (1-y) log(1-p)); naive SE = sd(l)/sqrt(n); cluster SE = ratio-of-sums linearisation with one cluster per match:
SE = sqrt(sum_c (S_c - mean*n_c)^2) / n   (S_c = loss sum of match c, n_c its rows, n all rows). Also round-level clusters for comparison.
"""
import argparse
import json
import math
import sys

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ap = argparse.ArgumentParser()
ap.add_argument("--pred", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
d = pl.read_parquet(a.pred, columns=["match_key", "round", "p", "label_ct_win"])
p = np.clip(d["p"].to_numpy().astype("float64"), 1e-6, 1 - 1e-6)
y = d["label_ct_win"].to_numpy().astype("float64")
loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
n = len(loss)
mean = float(loss.mean())
naive = float(loss.std(ddof=1) / math.sqrt(n))
out = {"rows": int(n), "mean_logloss": mean, "naive_se": naive}
for name, keys in (("match", d["match_key"]), ("round", d["match_key"] + "|" + d["round"].cast(pl.Utf8))):
    codes = keys.to_physical().to_numpy() if keys.dtype == pl.Categorical else pl.Series(keys).cast(pl.Categorical).to_physical().to_numpy()
    k = int(codes.max()) + 1
    s = np.bincount(codes, weights=loss, minlength=k)
    c = np.bincount(codes, minlength=k).astype("float64")
    se = float(math.sqrt(((s - mean * c) ** 2).sum()) / n * math.sqrt(k / (k - 1)))
    out[f"{name}_clusters"] = k
    out[f"{name}_cluster_se"] = se
    out[f"{name}_cluster_ratio"] = se / naive
json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
print(json.dumps(out, indent=1))
