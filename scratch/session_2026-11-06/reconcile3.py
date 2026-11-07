import json, re, subprocess
from collections import Counter
import pyarrow.parquet as pq

B = "/home/sagemaker-user/adam/builder"
F = B + "/final"
X = F + "/export_clean_v2"
E_DIR = F + "/export_clean"
S3 = "s3://pureskillgg-csgo-production-sagemaker-0/adam/tomes"


def s3cat(uri):
    r = subprocess.run(["aws", "s3", "cp", uri, "-"], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def lines_set(text):
    return set(l.strip() for l in text.splitlines() if l.strip())


def flags(path, extra):
    cols = ["match_key", "dup_keeper", "in_scope", "n_ok_rounds"] + extra
    t = pq.read_table(path, columns=cols).to_pydict()
    return {k: {c: t[c][i] for c in cols[1:]} for i, k in enumerate(t["match_key"])}


def usable(d):
    return bool(d["dup_keeper"]) and bool(d["in_scope"]) and (d["n_ok_rounds"] or 0) > 0


chunk = {}
for i in range(68):
    c = "c%03d" % i
    t = pq.read_table("%s/data/%s/matches.parquet" % (B, c), columns=["match_key", "dup_keeper", "in_scope", "n_ok_rounds", "map_name"]).to_pydict()
    for j, k in enumerate(t["match_key"]):
        chunk[k] = dict(dup_keeper=t["dup_keeper"][j], in_scope=t["in_scope"][j], n_ok_rounds=t["n_ok_rounds"][j], map_name=t["map_name"][j], c=c)
bm = flags(F + "/bucket_matches.parquet", ["map_name"])
lp = flags(F + "/laptop_matches.parquet", ["map_name"])
tr = flags(F + "/training_matches_v1.parquet", ["src", "gkey", "gkeeper", "ggroup", "map_name"])
export = set(json.load(open(X + "/match_keys.json")))
V = set(json.load(open(E_DIR + "/match_keys.json")))
hb = lines_set(open(F + "/holdout_b_keys.txt").read())
ho = lines_set(s3cat(S3 + "/holdout_2026-10-26_31/match_keys.txt"))
assign = json.load(open(F + "/split_manifest.json"))["assign"]
test_keys = set(k for k, v in assign.items() if v == "test")
bucket = set(chunk)
laptop = set(lp)
mirage = set(k for k in test_keys if (tr.get(k) or bm.get(k) or lp.get(k) or {}).get("map_name") == "de_mirage")
EXC = mirage | hb

# 1. split of the 14,708 non-usable bucket rows (chunk flags)
dup_removed = [k for k in bucket if not chunk[k]["dup_keeper"]]
keepers = [k for k in bucket if chunk[k]["dup_keeper"]]
keep_oos = [k for k in keepers if not chunk[k]["in_scope"]]
keep_in_nook = [k for k in keepers if chunk[k]["in_scope"] and (chunk[k]["n_ok_rounds"] or 0) == 0]
print("NONUSABLE_SPLIT dup_not_keeper=%d keeper_out_of_scope=%d keeper_in_scope_no_ok_rounds=%d total=%d" % (
    len(dup_removed), len(keep_oos), len(keep_in_nook), len(dup_removed) + len(keep_oos) + len(keep_in_nook)))

# 2. the 26 bucket keys usable by bm but not in training usable set
T = set(k for k in tr if usable(tr[k]))
U_bm_b = set(k for k in bucket if usable(bm[k]))
d26 = sorted(U_bm_b - T)
print("BM_USABLE_NOT_TRAIN_USABLE n=%d" % len(d26))
c26 = Counter()
for k in d26:
    t = tr.get(k)
    c26["in_train dup=%s gkeeper=%s in_scope=%s n_ok>0=%s src=%s" % (
        t["dup_keeper"], t["gkeeper"], t["in_scope"], (t["n_ok_rounds"] or 0) > 0, t["src"]) if t else "absent_from_train"] += 1
print("BM_NOT_TRAIN_DETAIL", dict(c26))
for k in d26:
    if tr.get(k) is None:
        print("ABSENT_FROM_TRAIN key=%s chunk=%s map=%s in_laptop=%s" % (k, chunk[k]["c"], chunk[k]["map_name"], k in laptop))

# 3. the 513 chunk-usable but global dup_keeper False in bucket_matches
U_chunk = set(k for k in bucket if usable(chunk[k]))
d513 = U_chunk - U_bm_b
print("CHUNK_USABLE_NOT_BM_USABLE n=%d" % len(d513))
print("  of which in_laptop=%d  in training=%d  in_training_usable=%d" % (
    sum(1 for k in d513 if k in laptop), sum(1 for k in d513 if k in tr), sum(1 for k in d513 if k in T)))

# 4. laptop side breakdown
lap_only = laptop - bucket
print("LAPTOP rows=%d in_bucket=%d laptop_only=%d" % (len(laptop), len(laptop & bucket), len(lap_only)))
ho_in_lap = ho & laptop
print("HOLDOUT_TOME in_laptop=%d in_bucket=%d laptop_only=%d usable_laptop_flags=%d in_train=%d in_export=%d" % (
    len(ho & laptop), len(ho & bucket), len(ho & lap_only), len(set(k for k in ho if k in lp and usable(lp[k]))), len(ho & set(tr)), len(ho & export)))
rest = lap_only - ho
print("LAPTOP_ONLY_NOT_HOLDOUT=%d usable_laptop_flags=%d" % (len(rest), len(set(k for k in rest if usable(lp[k])))))
print("LAPTOP_ONLY_MIRAGE=%d (of mirage test 1033; in bucket %d)" % (len(mirage & lap_only), len(mirage & bucket)))
exp_lap_only = export - bucket
print("EXPORT_LAPTOP_ONLY=%d  check_in_laptop=%d" % (len(exp_lap_only), len(exp_lap_only & laptop)))
print("LAPTOP_ONLY_USABLE_TRAIN=%d LAPTOP_ONLY_TRAIN_USABLE_EXCLUDED=%d" % (
    len(set(k for k in lap_only if k in T)), len(set(k for k in lap_only if k in T and k in EXC))))

# 5. the 47 empties
emp = V - export
print("EMPTIES n=%d in_bucket=%d in_laptop=%d" % (len(emp), len(emp & bucket), len(emp & laptop)))
print("EMPTIES_CHUNKS", dict(Counter(chunk[k]["c"] for k in emp if k in chunk)))

# 6. pos_bucket log tail and assembler reads
print("POS_LOG_TAIL")
print(subprocess.run(["tail", "-n", "12", B + "/pos_bucket.log"], capture_output=True, text=True).stdout)
print("POS_ASSEMBLE_LOG_TAIL")
print(subprocess.run(["tail", "-n", "8", B + "/pos_assemble.log"], capture_output=True, text=True).stdout)
print("POS_ASSEMBLE_READS")
print(subprocess.run(["grep", "-n", "pos_parts\\|bucket_c\\|glob\\|laptop", B + "/pos_assemble.py"], capture_output=True, text=True).stdout)
print("POS_BUCKET_PY_READS")
print(subprocess.run(["grep", "-n", "chunk\\|range\\|c0", B + "/pos_bucket.py"], capture_output=True, text=True).stdout[:3000])
