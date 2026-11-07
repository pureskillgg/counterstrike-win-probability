import polars as pl, json
B = "/home/sagemaker-user/adam/builder"; F = B + "/final"
bm = pl.read_parquet(F+"/bucket_matches.parquet"); lm = pl.read_parquet(F+"/laptop_matches.parquet"); cm = pl.read_parquet(F+"/clean_matches.parquet")
cr = pl.read_parquet(F+"/clean_rounds.parquet", columns=["match_key","use","src"])
EK = set(json.load(open(F+"/export_clean_v2/match_keys.json")))
ub = cr.filter((pl.col("use")) & (pl.col("src")=="bucket"))["match_key"].unique().to_list()
print("distinct keys with use&src=bucket", len(ub), "in EK", len(set(ub)&EK), "not in EK", len(set(ub)-EK))
ch = {}
for c in range(47, 68):
    for k in pl.read_parquet(f"{B}/data/c{c:03d}/matches.parquet", columns=["match_key"])["match_key"].to_list(): ch[k] = c
c47 = set(ch)
print("c047-c067 keys", len(c47), "clean_rounds rows src=bucket for these", cr.filter(pl.col("src")=="bucket").filter(pl.col("match_key").is_in(list(c47))).height, "distinct keys with src=bucket & use", len(set(ub)&c47))
gid = 5313067263640903444
mem = pl.concat([lm.filter(pl.col("dup_group")==gid).select(["match_key","map_name","match_date","dup_keeper","n_ok_rounds"]).with_columns(pl.lit("laptop").alias("tab")),
                 bm.filter(pl.col("dup_group")==gid).select(["match_key","map_name","match_date","dup_keeper","n_ok_rounds"]).with_columns(pl.lit("bucket").alias("tab"))])
print("dup group", gid, "members", mem.rows())
print("members in clean_matches", [k in set(cm["match_key"].to_list()) for k in mem["match_key"].to_list()])
for k in ["csds/2026/10/17/m_b7a4fb4c98afd959ee15e5b3/csds", "csds/2026/10/17/m_79ee2999743ab9b3df4ba29a/csds"]:
    r = lm.filter(pl.col("match_key")==k)
    print("laptop row for", k[-40:], r.select(["map_name","match_date","dup_keeper","n_ok_rounds","dup_group","dup_group_size"]).rows())
    same = cm.filter(pl.col("match_date").cast(pl.Utf8).str.starts_with("2026-10-17T00:50:48")) if "match_date" in cm.columns else None
    print("  clean_matches cols", cm.columns[:6])
    dm = bm.filter(pl.col("match_date").cast(pl.Utf8).str.starts_with("2026-10-17T00:50:48")).select(["match_key","dup_keeper","dup_group","n_ok_rounds","sig_hash"])
    print("  bucket rows with same date", dm.rows())
    dl = lm.filter(pl.col("match_date").cast(pl.Utf8).str.starts_with("2026-10-17T00:50:48")).select(["match_key","dup_keeper","dup_group","n_ok_rounds"])
    print("  laptop rows with same date", dl.rows())
