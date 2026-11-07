import polars as pl, json, subprocess
B = "/home/sagemaker-user/adam/builder"; F = B + "/final"
bm = pl.read_parquet(F+"/bucket_matches.parquet"); lm = pl.read_parquet(F+"/laptop_matches.parquet"); cm = pl.read_parquet(F+"/clean_matches.parquet"); tm = pl.read_parquet(F+"/training_matches_v1.parquet")
EK = set(json.load(open(F+"/export_clean_v2/match_keys.json")))
HB = set(l.strip() for l in open(F+"/holdout_b_keys.txt") if l.strip())
HT = set(l.strip() for l in subprocess.run(["aws","s3","cp","s3://pureskillgg-csgo-production-sagemaker-0/adam/tomes/holdout_2026-10-26_31/match_keys.txt","-"],capture_output=True).stdout.decode().splitlines() if l.strip())
for k in ["csds/2026/10/28/m_64f6ec2252d36f59d33270dc/csds", "csds/2026/10/28/m_050fafc3594bb51ae5ec231b/csds"]:
    print("holdout-tome membership", k[-40:], "in HT", k in HT, "in HB", k in HB)
print("HT members with dup_group 5313067263640903444 or 9132813463601617849:", [k for k in HT if k.endswith("m_64f6ec2252d36f59d33270dc/csds") or k.endswith("m_050fafc3594bb51ae5ec231b/csds")])
# bucket usable: bm definition vs training flags
bu = set(bm.filter(pl.col("dup_keeper") & pl.col("in_scope") & (pl.col("n_ok_rounds")>0))["match_key"].to_list())
tu = set(tm.filter(pl.col("dup_keeper") & pl.col("in_scope") & (pl.col("n_ok_rounds")>0) & (pl.col("src")=="bucket"))["match_key"].to_list())
BK = set(bm["match_key"].to_list())
print("bm usable", len(bu), "training usable (src=bucket)", len(tu), "training usable src=bucket & in BK", len(tu & BK))
print("bm usable not training usable", len(bu - tu), "training usable not bm usable", len((tu & BK) - bu))
tflags = {r["match_key"]: r for r in tm.iter_rows(named=True)}
cat = {}
for k in bu - tu:
    r = tflags.get(k)
    if r is None: c = "absent_from_training"
    else: c = f"training src={r['src']} keeper={r['dup_keeper']} in_scope={r['in_scope']} n_ok>0={r['n_ok_rounds']>0}"
    cat[c] = cat.get(c, 0) + 1
print("categories of bm-usable not training-usable", cat)
print("keys not training-usable but bm usable and in HB or MIR1:", len([k for k in bu - tu if k in HB]))
# per-chunk prep use_matches vs recompute
bad = []; tot_p = 0; tot_r = 0
for c in range(68):
    rep = json.load(open(f"{B}/data/c{c:03d}/prep_report.json"))
    d = pl.read_parquet(f"{B}/data/c{c:03d}/matches.parquet", columns=["match_key","dup_keeper","in_scope","n_ok_rounds"])
    r = d.filter(pl.col("dup_keeper") & pl.col("in_scope") & (pl.col("n_ok_rounds")>0)).height
    tot_p += rep.get("use_matches", 0); tot_r += r
    if rep.get("use_matches", 0) != r: bad.append((c, rep.get("use_matches", 0), r))
print("per-chunk prep use_matches vs recompute: chunks differing", len(bad), bad[:6], "totals", tot_p, tot_r)
# export 2026/09 source
e09 = [k for k in EK if k.split("/")[1:3] == ["2026","09"]]
print("export 2026/09 keys", len(e09), "src", pl.DataFrame({"match_key": e09}).join(cm.select(["match_key","src"]), on="match_key", how="left").group_by("src").len().rows())
