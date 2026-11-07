import polars as pl, json, collections
B = "/home/sagemaker-user/adam/builder"
F = B + "/final"
bm = pl.read_parquet(F+"/bucket_matches.parquet")
lm = pl.read_parquet(F+"/laptop_matches.parquet")
cm = pl.read_parquet(F+"/clean_matches.parquet")
tm = pl.read_parquet(F+"/training_matches_v1.parquet")
BK = set(bm["match_key"].to_list()); LK = set(lm["match_key"].to_list())
CK = set(cm["match_key"].to_list()); TK = set(tm["match_key"].to_list())
EK = set(json.load(open(F+"/export_clean_v2/match_keys.json")))
HB = set(l.strip() for l in open(F+"/holdout_b_keys.txt") if l.strip())
import subprocess
r = subprocess.run(["aws","s3","cp","s3://pureskillgg-csgo-production-sagemaker-0/adam/tomes/holdout_2026-10-26_31/match_keys.txt","-"], capture_output=True)
HT = set(l.strip() for l in r.stdout.decode().splitlines() if l.strip())
v1 = json.load(open(F+"/split_manifest.json"))["assign"]
mp = {}
for df in (bm, lm):
    for k, m in zip(df["match_key"].to_list(), df["map_name"].to_list()): mp[k] = m
MIR1 = {k for k,s in v1.items() if s=='test' and mp.get(k)=='de_mirage'}
EXCL_B = HB | MIR1
print("LK-CK", len(LK-CK), "of which HT", len((LK-CK)&HT), "BK", len((LK-CK)&BK), "other", len((LK-CK)-HT-BK), "other in HB|MIR1", len(((LK-CK)-HT-BK)&EXCL_B))
print("BK-CK", len(BK-CK), "in LK", len((BK-CK)&LK), "in HB|MIR1", len((BK-CK)&EXCL_B))
print("HB in CK", len(HB&CK), "HB in TK", len(HB&TK), "MIR1 in CK", len(MIR1&CK), "MIR1 in TK", len(MIR1&TK), "MIR1 in EK", len(MIR1&EK))
print("TK-CK", len(TK-CK), "CK-TK", len(CK-TK), "TK&EK", len(TK&EK))
print("TK - CK sample src/keys in BK/LK:", len((TK-CK)&BK), len((TK-CK)&LK))
# CK - EK partition by reason
cmd = {r["match_key"]: r for r in cm.iter_rows(named=True)}
notEK = CK - EK
print("CK-EK", len(notEK), "by src", collections.Counter(cmd[k]["src"] for k in notEK))
reason = collections.Counter()
for k in notEK:
    r = cmd[k]
    if not r["in_scope"]: reason["out_of_scope"] += 1
    elif not r["dup_keeper"]: reason["in_scope_dup_nonkeeper"] += 1
    elif r["n_ok_rounds"] == 0: reason["in_scope_keeper_no_ok_rounds"] += 1
    else: reason["in_scope_keeper_ok_rounds_removed_empty"] += 1
print("CK-EK reasons", dict(reason))
rem = [k for k in notEK if cmd[k]["in_scope"] and cmd[k]["dup_keeper"] and cmd[k]["n_ok_rounds"]>0]
print("removed-empty (in_scope&keeper&n_ok>0 not in EK)", len(rem), "src", collections.Counter(cmd[k]["src"] for k in rem), "in BK", len(set(rem)&BK), "in HB|MIR1", len(set(rem)&EXCL_B))
# BK partition
bk_reason = collections.Counter()
for k in BK:
    if k in EXCL_B: bk_reason["excluded_HB_or_MIR1"] += 1
    elif k not in CK: bk_reason["not_in_clean_matches"] += 1
    elif k in EK: bk_reason["exported"] += 1
    else: bk_reason["in_clean_not_exported"] += 1
print("BK partition", dict(bk_reason), "sum", sum(bk_reason.values()))
# usable definitions across chunk files
tot_use = 0; tot_dk = 0; tot_rows = 0; per_chunk_keys = {}
ch_use = {}
rows = []
for c in range(68):
    d = pl.read_parquet(f"{B}/data/c{c:03d}/matches.parquet")
    cols = d.columns
    rows.append((c, d.height, cols[:0]))
    if c == 0: print("data/cNNN/matches.parquet columns", cols)
    if "dup_keeper" in cols and "in_scope" in cols and "n_ok_rounds" in cols:
        u = d.filter(pl.col("dup_keeper") & pl.col("in_scope") & (pl.col("n_ok_rounds")>0)).height
        tot_use += u
    if "dup_keeper" in cols:
        tot_dk += d.filter(pl.col("dup_keeper")).height
print("sum usable (dup_keeper&in_scope&n_ok>0) over data/cNNN", tot_use, "sum dup_keeper", tot_dk)
print("bm usable", bm.filter(pl.col("dup_keeper") & pl.col("in_scope") & (pl.col("n_ok_rounds")>0)).height)
rep = json.load(open(B+"/data/c047/prep_report.json"))
print("c047 prep_report keys", list(rep.keys())[:20])
print("c047 prep_report", json.dumps(rep)[:700])
