# Read-only check on 2026-09-01 only (not a holdout day): does the Nov-2 every1 file contain the Nov-4 every2_sites_v1 rows, value for value?
import polars as pl
R = "G:/pureskill/csds-cs2/data/pos_features_1hz/"
a = pl.read_parquet(R + "every1/2026-09-01.parquet")
b = pl.read_parquet(R + "every2_sites_v1/2026-09-01.parquet")
print("every1 rows", a.height, "| every2_sites_v1 rows", b.height, "| ratio", round(a.height / b.height, 4))
print("every1 columns", a.width, "| every2_sites_v1 columns", b.width)
K = ["match_key", "round", "tick"]
feats = [c for c in b.columns if c not in K]
j = b.join(a, on=K, how="left", suffix="_e1")
print("every2_sites_v1 rows found in every1:", int(j.filter(pl.col(feats[0] + "_e1").is_not_null() | pl.col(feats[0]).is_null()).height), "of", b.height)
both = b.join(a, on=K, how="inner", suffix="_e1")
print("inner matched rows", both.height)
for c in feats:
    x = both[c].cast(pl.Float64); y = both[c + "_e1"].cast(pl.Float64)
    nd = int(((x.is_null() != y.is_null()) | (x.is_not_null() & y.is_not_null() & ((x - y).abs() > 1e-4))).sum())
    print(f"  {c:22s} differing {nd}")
odd = a.join(b.select(K), on=K, how="anti")
print("every1 rows not in every2_sites_v1 (odd seconds):", odd.height)
