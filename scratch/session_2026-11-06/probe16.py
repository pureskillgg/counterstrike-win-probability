import polars as pl, json
F = "/home/sagemaker-user/adam/builder/final"
bm = pl.read_parquet(F+"/bucket_matches.parquet"); tm = pl.read_parquet(F+"/training_matches_v1.parquet"); cm = pl.read_parquet(F+"/clean_matches.parquet")
EK = set(json.load(open(F+"/export_clean_v2/match_keys.json")))
bu = bm.filter(pl.col("dup_keeper") & pl.col("in_scope") & (pl.col("n_ok_rounds")>0))
t = tm.select(["match_key","src","dup_keeper","in_scope","n_ok_rounds"]).rename({"src":"t_src","dup_keeper":"t_keeper","in_scope":"t_in","n_ok_rounds":"t_nok"})
c = cm.select(["match_key","src","dup_keeper","in_scope","n_ok_rounds"]).rename({"src":"c_src","dup_keeper":"c_keeper","in_scope":"c_in","n_ok_rounds":"c_nok"})
j = bu.select(["match_key","n_ok_rounds"]).join(t, on="match_key", how="left").join(c, on="match_key", how="left")
diff = j.filter(pl.col("t_nok").is_null() | (pl.col("t_nok")==0) | (pl.col("t_keeper")==False) | (pl.col("t_src")=="laptop") )
diff = diff.filter(pl.col("t_src")!="laptop")
print("bm-usable keys in training but not src=bucket usable or absent:", diff.height)
print(diff.group_by(["t_src","t_keeper","t_nok","c_src","c_keeper","c_nok"], maintain_order=True).len().rows()[:10])
print("of these, in export:", sum(1 for k in diff["match_key"].to_list() if k in EK), "in clean_matches:", diff.filter(pl.col("c_src").is_not_null()).height)
print("training n_ok=0 among bm-usable bucket keys:", j.filter(pl.col("t_nok")==0).height, "in export:", sum(1 for k in j.filter(pl.col("t_nok")==0)["match_key"].to_list() if k in EK))
