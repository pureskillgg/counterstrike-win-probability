import polars as pl, json, glob
B = "/home/sagemaker-user/adam/builder"; F = B + "/final"
bm = pl.read_parquet(F+"/bucket_matches.parquet"); tm = pl.read_parquet(F+"/training_matches_v1.parquet"); cm = pl.read_parquet(F+"/clean_matches.parquet")
cr = pl.read_parquet(F+"/clean_rounds.parquet", columns=["match_key","round","round_status","use","src"])
bu = bm.filter(pl.col("dup_keeper") & pl.col("in_scope") & (pl.col("n_ok_rounds")>0))
t0 = tm.filter((pl.col("n_ok_rounds")==0) & pl.col("dup_keeper") & pl.col("in_scope") & (pl.col("src")=="bucket"))
ks = [k for k in t0["match_key"].to_list() if k in set(bu["match_key"].to_list())]
print("24 keys", len(ks))
rs = cr.filter(pl.col("match_key").is_in(ks))
print("clean_rounds rows for these keys", rs.height, "status", rs.group_by("round_status").len().sort("len", descending=True).rows()[:6], "use", rs.group_by("use").len().rows())
print("bm n_ok_rounds for these", bm.filter(pl.col("match_key").is_in(ks))["n_ok_rounds"].describe()["mean"] if False else bm.filter(pl.col("match_key").is_in(ks)).select(pl.col("n_ok_rounds").min().alias("min"), pl.col("n_ok_rounds").max().alias("max")).rows())
print("cm n_rounds_re for these", cm.filter(pl.col("match_key").is_in(ks)).select(pl.col("n_rounds_re").min().alias("min"), pl.col("n_rounds_re").max().alias("max")).rows())
# chunk-level prep: which chunk and per-chunk n_ok from data
chunk = {}
for c in range(68):
    for k in pl.read_parquet(f"{B}/data/c{c:03d}/matches.parquet", columns=["match_key"])["match_key"].to_list():
        if k in set(ks): chunk[k] = c
print("chunks of the 24", sorted(set(chunk.values())))
d = pl.concat([pl.read_parquet(f"{B}/data/c{c:03d}/matches.parquet", columns=["match_key","n_ok_rounds","dup_keeper","in_scope"]).filter(pl.col("match_key").is_in(ks)) for c in sorted(set(chunk.values()))])
print("chunk-level n_ok for the 24 (min,max)", d.select(pl.col("n_ok_rounds").min().alias("min"), pl.col("n_ok_rounds").max().alias("max")).rows(), "keepers", d["dup_keeper"].sum(), "in_scope", d["in_scope"].sum())
