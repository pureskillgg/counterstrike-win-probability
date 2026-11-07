import sys

import numpy as np
import polars as pl

OLD = "/home/sagemaker-user/adam/builder/final/pos_parts/bucket_c000.parquet"
NEW = sys.argv[1]
K = ["match_key", "round", "tick"]
old = pl.read_parquet(OLD)
new = pl.read_parquet(NEW)
print("old rows", old.height, "| old cols", old.width, "| new rows", new.height, "| new cols", new.width)
print("new has second column:", "second" in new.columns)
ne = new.filter(pl.col("second") % 2 == 0)
no = new.filter(pl.col("second") % 2 == 1)
print("new even rows", ne.height, "| new odd rows", no.height, "| odd/even ratio", round(no.height / max(ne.height, 1), 4))
print("duplicate keys: old", old.height - old.unique(K).height, "| new", new.height - new.unique(K).height)
feat = [c for c in old.columns if c not in K]
print("feature columns in old:", len(feat), "| all present in new:", all(c in new.columns for c in feat))
print("dtypes equal (even rows vs old):", all(old.schema[c] == ne.schema[c] for c in feat))
a = old.sort(K)
b = ne.sort(K)
print("key sets equal (old vs even rows):", a.select(K).equals(b.select(K)))
bad = {}
for c in feat:
    x, y = a[c], b[c]
    nmask = int(np.count_nonzero(x.is_null().to_numpy() != y.is_null().to_numpy()))
    vx = np.ascontiguousarray(x.fill_null(0).to_numpy(), dtype=np.float32)
    vy = np.ascontiguousarray(y.fill_null(0).to_numpy(), dtype=np.float32)
    vdiff = int(np.count_nonzero(vx.view(np.uint32) != vy.view(np.uint32)))
    if vdiff or nmask:
        bad[c] = (vdiff, nmask)
print("feature columns compared:", len(feat), "| columns with value-bit or null-mask differences:", len(bad))
print("differences per column (value bits, null mask):", bad)
print("BYTE-IDENTICAL even rows vs pos_parts:", (a.height == b.height) and a.select(K).equals(b.select(K)) and not bad)
print("odd rows sample null share:", {c: round(no[c].null_count() / max(no.height, 1), 4) for c in ["c_sees_enemy", "min_enemy_dist", "c_min_bomb_dist"]})
print("odd rows with a value in c_sees_enemy:", int(no["c_sees_enemy"].is_not_null().sum()))
