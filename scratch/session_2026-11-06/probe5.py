import polars as pl, json, subprocess, collections
B = "/home/sagemaker-user/adam/builder"
F = B + "/final"
bm = pl.read_parquet(F+"/bucket_matches.parquet")
lm = pl.read_parquet(F+"/laptop_matches.parquet")
cm = pl.read_parquet(F+"/clean_matches.parquet")
BK = set(bm["match_key"].to_list()); LK = set(lm["match_key"].to_list())
CK = set(cm["match_key"].to_list())
EK = set(json.load(open(F+"/export_clean_v2/match_keys.json")))
HB = set(l.strip() for l in open(F+"/holdout_b_keys.txt") if l.strip())
print("CK-(BK|LK)", len(CK-(BK|LK)), "(BK|LK)-CK", len(((BK|LK)-CK)), "BK-CK", len(BK-CK), "LK-CK", len(LK-CK))
print("cm rows", cm.height, "unique keys", len(CK))
print("cm src dist", cm.group_by("src").len().sort("src").rows())
cmd = {r["match_key"]: r for r in cm.iter_rows(named=True)}
print("EK in cm", len(EK & CK), "EK-cm", len(EK-CK))
print("EK by src", collections.Counter(cmd[k]["src"] for k in EK if k in cmd))
print("EK by (src,dup_keeper)", collections.Counter((cmd[k]["src"], cmd[k]["dup_keeper"]) for k in EK if k in cmd))
print("EK in_scope", collections.Counter(cmd[k]["in_scope"] for k in EK if k in cmd))
print("EK n_ok>0", collections.Counter(cmd[k]["n_ok_rounds"]>0 for k in EK if k in cmd))
# cm rows not in EK and in_scope and n_ok>0
notEK = [k for k in CK if k not in EK]
print("cm not in EK", len(notEK), "by src", collections.Counter(cmd[k]["src"] for k in notEK))
print("cm not in EK by (in_scope, dup_keeper, n_ok>0)", collections.Counter((cmd[k]["in_scope"], cmd[k]["dup_keeper"], cmd[k]["n_ok_rounds"]>0) for k in notEK).most_common(12))
# chunk membership
per = {}
for c in range(68):
    per[c] = pl.read_parquet(f"{B}/data/c{c:03d}/matches.parquet", columns=["match_key"])["match_key"].to_list()
ch = {}
for c in per:
    for k in per[c]: ch[k] = c
def bucket_part(lo, hi):
    ks = [k for c in range(lo,hi+1) for k in per[c]]
    return ks
for lo, hi, label in [(0,46,"c000-c046"), (47,67,"c047-c067")]:
    ks = bucket_part(lo,hi)
    print(label, "keys", len(ks), "in EK", sum(k in EK for k in ks), "in cm", sum(k in CK for k in ks),
          "src", collections.Counter(cmd[k]["src"] for k in ks if k in cmd),
          "EK src", collections.Counter(cmd[k]["src"] for k in ks if k in EK and k in cmd))
    print("   in cm not EK by in_scope/dup_keeper/n_ok>0", collections.Counter((cmd[k]["in_scope"], cmd[k]["dup_keeper"], cmd[k]["n_ok_rounds"]>0) for k in ks if k in cmd and k not in EK).most_common(8))
    print("   bucket_matches dup_keeper, in_scope, n_ok>0 for chunk keys", collections.Counter((r["dup_keeper"], r["in_scope"], r["n_ok_rounds"]>0) for r in bm.filter(pl.col("match_key").is_in(ks)).iter_rows(named=True)).most_common(8))
# bucket_matches usable definitions
bu = bm.filter(pl.col("dup_keeper") & pl.col("in_scope") & (pl.col("n_ok_rounds")>0))
print("bm dup_keeper&in_scope&n_ok>0", bu.height, " bm dup_keeper&in_scope", bm.filter(pl.col("dup_keeper") & pl.col("in_scope")).height, " bm dup_keeper", bm.filter(pl.col("dup_keeper")).height)
print("bm usable_by_chunk? (dup_keeper&n_ok>0)", bm.filter(pl.col("dup_keeper") & (pl.col("n_ok_rounds")>0)).height)
print("cm src=bucket & EK", sum(1 for k in EK if k in cmd and cmd[k]["src"]=="bucket"), " cm src=laptop & EK", sum(1 for k in EK if k in cmd and cmd[k]["src"]=="laptop"))
print("EK & HB", len(EK&HB))
