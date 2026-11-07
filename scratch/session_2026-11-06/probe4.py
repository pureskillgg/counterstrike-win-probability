import polars as pl, json, subprocess, collections
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
r = subprocess.run(["aws","s3","cp","s3://pureskillgg-csgo-production-sagemaker-0/adam/tomes/holdout_2026-10-26_31/match_keys.txt","-"], capture_output=True)
HT = set(l.strip() for l in r.stdout.decode().splitlines() if l.strip())
print("sizes BK", len(BK), "LK", len(LK), "CK", len(CK), "TK", len(TK), "EK", len(EK), "HB", len(HB), "HT", len(HT))
print("BK&LK", len(BK&LK), "BK|LK", len(BK|LK), "BK-LK", len(BK-LK), "LK-BK", len(LK-BK))
print("EK&BK", len(EK&BK), "EK&LK", len(EK&LK), "EK-(BK|LK)", len(EK-(BK|LK)))
print("HT&BK", len(HT&BK), "HT&LK", len(HT&LK), "HT&EK", len(HT&EK), "HT&TK", len(HT&TK), "HT&CK", len(HT&CK))
print("HB&BK", len(HB&BK), "HB&LK", len(HB&LK), "HB&EK", len(HB&EK), "HB&HT", len(HB&HT), "HB&TK", len(HB&TK))
# map lookup
mp = {}
for df in (bm, lm):
    for k, mname in zip(df["match_key"].to_list(), df["map_name"].to_list()):
        mp[k] = mname
v1 = json.load(open(F+"/split_manifest.json"))["assign"]
v2 = json.load(open(F+"/split_manifest_v2.json"))["assign"]
print("v1 assign keys in BK|LK", len(set(v1)&(BK|LK)), "v1 total", len(v1), "v2 keys in BK|LK", len(set(v2)&(BK|LK)), "v2 total", len(v2))
print("HT in v2 assign", len(HT & set(v2)), "HT in v1 assign", len(HT & set(v1)), "HB in v1", len(HB&set(v1)), "HB in v2", len(HB&set(v2)))
print("HT v2 split dist", collections.Counter(v2.get(k,'NA') for k in HT))
print("HB v2 split dist", collections.Counter(v2.get(k,'NA') for k in HB))
mir1 = {k for k,s in v1.items() if s=='test' and mp.get(k)=='de_mirage'}
mir2 = {k for k,s in v2.items() if s=='test' and mp.get(k)=='de_mirage'}
mirall = {k for k,s in v2.items() if mp.get(k)=='de_mirage' and s=='test'}
print("mirage v1 test", len(mir1), "in BK", len(mir1&BK), "in LK", len(mir1&LK), "in EK", len(mir1&EK), "in HB", len(mir1&HB), "in HT", len(mir1&HT))
print("mirage v2 test", len(mir2), "in BK", len(mir2&BK), "in LK", len(mir2&LK), "in EK", len(mir2&EK), "in HB", len(mir2&HB), "in HT", len(mir2&HT))
print("mirage v1&v2 test equal", mir1==mir2)
print("BK&HB&mir1", len(BK&HB&mir1), "BK&mir1", len(BK&mir1), "LK&mir1", len(LK&mir1), "HB&mir1", len(HB&mir1))
print("export split dist v2", collections.Counter(v2.get(k,'NA') for k in EK))
print("export keys with map de_mirage", sum(1 for k in EK if mp.get(k)=='de_mirage'))
print("export keys per map", collections.Counter(mp.get(k,'NA') for k in EK).most_common(10))
print("HB/HT maps", collections.Counter(mp.get(k,'NA') for k in HB).most_common(8), collections.Counter(mp.get(k,'NA') for k in HT).most_common(8))
# chunk membership
chunk = {}
per = {}
for c in range(68):
    p = f"{B}/data/c{c:03d}/matches.parquet"
    ks = pl.read_parquet(p, columns=["match_key"])["match_key"].to_list()
    per[c] = ks
    for k in ks:
        chunk.setdefault(k, []).append(c)
allchunk = [k for c in per for k in per[c]]
print("chunk keys total", len(allchunk), "unique", len(set(allchunk)), "union==BK", set(allchunk)==BK, "BK-union", len(BK-set(allchunk)), "union-BK", len(set(allchunk)-BK))
print("chunk keys in HT", len(HT & set(allchunk)), "in HB", len(HB & set(allchunk)), "in mir1", len(mir1 & set(allchunk)))
print("chunk keys per chunk sample", {c: len(per[c]) for c in (0,46,47,67)})
print("chunks 47..67 keys", sum(len(per[c]) for c in range(47,68)), "in LK", sum(1 for c in range(47,68) for k in per[c] if k in LK), "in EK", sum(1 for c in range(47,68) for k in per[c] if k in EK))
print("chunks 0..46 keys", sum(len(per[c]) for c in range(0,47)), "in EK", sum(1 for c in range(0,47) for k in per[c] if k in EK))
print("chunk keys per c047-c067 in tome", sum(1 for c in range(47,68) for k in per[c] if k in HT))
