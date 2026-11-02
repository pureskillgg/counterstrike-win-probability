"""Detail of the frozen-key column differences (dup_group_size, modal_pc): how many rows differ and example values. Counts only."""
import json, os
import polars as pl
ROOT = r"G:\pureskill\csds-cs2"
BK = os.path.join(ROOT, "data", "combined", "_backup_pre_22_26_20261027")
CU = os.path.join(ROOT, "data", "combined")
old = list(json.load(open(os.path.join(BK, "split_manifest.json"), encoding="utf-8"))["assign"].keys())
for name, keys in [("matches.parquet", ["match_key"]), ("rounds.parquet", ["match_key", "round"])]:
    a = pl.read_parquet(os.path.join(BK, name)).filter(pl.col("match_key").is_in(old)).sort(keys)
    b = pl.read_parquet(os.path.join(CU, name)).filter(pl.col("match_key").is_in(old)).sort(keys)
    print(name, "dtypes equal:", a.schema == b.schema)
    for c in ["dup_group_size", "modal_pc", "dup_group", "dup_keeper", "dup_key"]:
        if c not in a.columns:
            continue
        x = a[c]; y = b[c]
        neq = ((x != y) & ~(x.is_null() & y.is_null())) | (x.is_null() != y.is_null())
        n = int(neq.sum())
        print(f"  {c}: rows differing {n} of {a.height}", end="")
        if n:
            idx = neq.arg_true()[:3].to_list()
            print(" examples backup->current:", [(x[i], y[i]) for i in idx])
        else:
            print()
    print("  match_key sets equal:", a["match_key"].to_list() == b["match_key"].to_list())
