import sys
import polars as pl
base = sys.argv[1]
cols = ["c_min_bomb_dist", "c_n_bomb_500", "defuse_slack", "t_min_bomb_dist", "c_site_hi", "t_blind_sum"]
for c in ["c050", "c060"]:
    d = pl.read_parquet(f"{base}/bucket_{c}.parquet", columns=cols)
    print(c, "rows", d.height, {x: int(d[x].null_count()) for x in cols}, flush=True)
