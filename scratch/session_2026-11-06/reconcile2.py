import json, os, re, subprocess, time
from collections import Counter, defaultdict
import pyarrow.parquet as pq

B = "/home/sagemaker-user/adam/builder"
F = B + "/final"
E_DIR = F + "/export_clean"
X = F + "/export_clean_v2"
S3 = "s3://pureskillgg-csgo-production-sagemaker-0/adam/tomes"
OCT = re.compile(r"^csds/2026/10/(2[6-9]|3[01])/")


def s3cat(uri):
    r = subprocess.run(["aws", "s3", "cp", uri, "-"], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def lines_set(text):
    return set(l.strip() for l in text.splitlines() if l.strip())


def load_flags(path, extra):
    cols = ["match_key", "dup_keeper", "in_scope", "n_ok_rounds"] + extra
    t = pq.read_table(path, columns=cols).to_pydict()
    out = {}
    for i, k in enumerate(t["match_key"]):
        out[k] = {c: t[c][i] for c in cols[1:]}
    return out


def usable(d):
    return bool(d["dup_keeper"]) and bool(d["in_scope"]) and (d["n_ok_rounds"] or 0) > 0


# ---------- load ----------
bm = load_flags(F + "/bucket_matches.parquet", ["map_name"])
lp = load_flags(F + "/laptop_matches.parquet", ["map_name"])
tr = load_flags(F + "/training_matches_v1.parquet", ["src", "gkey", "gkeeper", "ggroup", "map_name"])
chunk_flags = {}
chunk_of = {}
chunk_keys = defaultdict(set)
for i in range(68):
    c = "c%03d" % i
    t = pq.read_table("%s/data/%s/matches.parquet" % (B, c), columns=["match_key", "dup_keeper", "in_scope", "n_ok_rounds"]).to_pydict()
    for j, k in enumerate(t["match_key"]):
        chunk_flags[k] = {"dup_keeper": t["dup_keeper"][j], "in_scope": t["in_scope"][j], "n_ok_rounds": t["n_ok_rounds"][j]}
        chunk_of[k] = c
        chunk_keys[c].add(k)
bucket_keys = set(chunk_flags)
export = set(json.load(open(X + "/match_keys.json")))
V = set(json.load(open(E_DIR + "/match_keys.json")))
hb = lines_set(open(F + "/holdout_b_keys.txt").read())
ho = lines_set(s3cat(S3 + "/holdout_2026-10-26_31/match_keys.txt"))
assign = json.load(open(F + "/split_manifest.json"))["assign"]
test_keys = set(k for k, v in assign.items() if v == "test")
mirage = set(k for k in test_keys if (tr.get(k) or bm.get(k) or lp.get(k) or {}).get("map_name") == "de_mirage")
EXC = mirage | hb

print("LOADED bucket_keys=%d bm=%d lp=%d tr=%d export=%d V=%d hb=%d ho=%d mirage=%d EXC=%d" % (
    len(bucket_keys), len(bm), len(lp), len(tr), len(export), len(V), len(hb), len(ho), len(mirage), len(EXC)))

# ---------- 1. chunk prep flags vs bucket_matches.parquet flags ----------
dis = Counter()
ex_dis = []
for k in bucket_keys:
    a = chunk_flags[k]
    b = bm.get(k)
    if b is None:
        dis["missing_in_bm"] += 1
        continue
    for f in ["dup_keeper", "in_scope"]:
        if bool(a[f]) != bool(b[f]):
            dis[f] += 1
    if ((a["n_ok_rounds"] or 0) > 0) != ((b["n_ok_rounds"] or 0) > 0):
        dis["n_ok_gt0"] += 1
    if usable(a) != usable(b):
        dis["usable"] += 1
        if len(ex_dis) < 3:
            ex_dis.append((k, chunk_of[k], a, b))
print("FLAG_DIFF chunk_vs_bucket_matches", dict(dis))
for e in ex_dis:
    print("  example", e)

U_chunk = set(k for k in bucket_keys if usable(chunk_flags[k]))
U_bm = set(k for k in bm if usable(bm[k]))
U_lp = set(k for k in lp if usable(lp[k]))
T = set(k for k in tr if usable(tr[k]))
print("USABLE U_chunk=%d U_bm=%d U_lp=%d T_training=%d" % (len(U_chunk), len(U_bm), len(U_lp), len(T)))
print("U_chunk_minus_U_bm=%d U_bm_minus_U_chunk=%d" % (len(U_chunk - U_bm), len(U_bm - U_chunk)))

# ---------- 2. export and training consistency ----------
print("V_minus_T=%d T_minus_V=%d T_minus_V_minus_EXC=%d V_eq_T_minus_EXC=%s" % (
    len(V - T), len(T - V), len((T - V) - EXC), V == (T - EXC)))
print("E_minus_V=%d V_minus_E=%d" % (len(export - V), len(V - export)))
print("T_bucket=%d T_laptop=%d T_only_laptop=%d T_only_bucket=%d T_neither=%d" % (
    len(T & bucket_keys), len(T & set(lp)), len(T - bucket_keys - set(lp)) , len((T & bucket_keys) - set(lp)), len(T - bucket_keys - set(lp))))
print("E_bucket=%d E_laptop=%d E_only_laptop=%d E_both=%d" % (
    len(export & bucket_keys), len(export & set(lp)), len(export - bucket_keys), len(export & bucket_keys & set(lp))))
print("LAPTOP_BUCKET_OVERLAP all=%d usable_bucket_and_usable_laptop=%d" % (len(set(lp) & bucket_keys), len(U_chunk & U_lp)))

# ---------- 3. the gap: chunk-usable, not excluded, not in export ----------
gap = U_chunk - EXC - V
print("GAP_CHUNK_USABLE_NOT_EXCLUDED_NOT_IN_V=%d" % len(gap))
gc = Counter()
for k in gap:
    t = tr.get(k)
    if t is None:
        gc["not_in_training"] += 1
    else:
        gc["in_training src=%s dup=%s gkeeper=%s in_scope=%s" % (t["src"], t["dup_keeper"], t["gkeeper"], usable(t) and "usable" or "not_usable")] += 1
print("GAP_CATEGORIES", dict(gc))
gl = Counter()
for k in gap:
    gl["in_laptop" if k in lp else "not_in_laptop"] += 1
print("GAP_LAPTOP_MEMBERSHIP", dict(gl))
gb = Counter()
for k in gap:
    gb["in_bm_usable" if k in U_bm else "not_in_bm_usable"] += 1
print("GAP_IN_BM_USABLE", dict(gb))
# training row flags for gap keys with ggroup info
gg = Counter()
for k in gap:
    t = tr.get(k)
    if t is None:
        continue
    gg["dup_keeper=%s gkeeper=%s" % (t["dup_keeper"] if "dup_keeper" in t else "NA", t["gkeeper"])] += 1
print("GAP_TRAIN_GROUP_FLAGS", dict(gg))
# if training dup_keeper is False, check group keeper src
keepers_by_group = defaultdict(list)
for k, t in tr.items():
    if t.get("gkeeper"):
        keepers_by_group[t["ggroup"]].append(k)
gk = Counter()
for k in gap:
    t = tr.get(k)
    if t is None:
        continue
    kk = keepers_by_group.get(t["ggroup"], [])
    gk["group_keeper_found=%s keeper_in_laptop=%s" % (len(kk) > 0, any(x in lp for x in kk))] += 1
print("GAP_GROUP_KEEPER", dict(gk))

# ---------- 4. per-chunk gap and pos coverage ----------
rows = []
for i in range(68):
    c = "c%03d" % i
    ch_keys = chunk_keys[c]
    u_c = set(k for k in ch_keys if usable(chunk_flags[k]))
    v_c = V & ch_keys
    exc_c = EXC & ch_keys
    gap_c = (u_c - EXC) - V
    pp = "%s/pos_parts/bucket_%s.parquet" % (F, c)
    pk = None
    if os.path.exists(pp):
        pk = set(pq.read_table(pp, columns=["match_key"]).column("match_key").to_pylist())
    pos_cov = None
    if pk is not None:
        pos_cov = (len(pk & v_c), len(v_c - pk), len(pk - v_c))
    ex_c = len(export & ch_keys)
    rows.append((c, len(ch_keys), len(u_c), len(v_c), len(exc_c), len(gap_c), ex_c, pos_cov))
print("PERCHUNK c raw usable_prep v_in_export_clean excluded gap ex_export pos(in_v,v_missing_from_pos,pos_extra)")
for r in rows:
    print("PC", r)
print("GAP_SUM_BY_CHUNK=%d" % sum(r[5] for r in rows))
print("EX_SUM_BY_CHUNK=%d" % sum(r[6] for r in rows))
print("EX_SUM_c047_c067=%d" % sum(r[6] for r in rows if int(r[0][1:]) >= 47))
print("EX_SUM_c046_c067=%d" % sum(r[6] for r in rows if int(r[0][1:]) >= 46))
print("EX_SUM_c000_c045=%d" % sum(r[6] for r in rows if int(r[0][1:]) <= 45))
print("MIN_EX_CHUNK=%s MAX_EX_CHUNK=%s" % (min(rows, key=lambda r: r[6])[:1] + (min(r[6] for r in rows),), max(rows, key=lambda r: r[6])[:1] + (max(r[6] for r in rows),)))
print("ZERO_EX_CHUNKS=%s" % [r[0] for r in rows if r[6] == 0])
pos_ok = [r[0] for r in rows if r[7] is not None and r[7][1] == 0 and r[7][2] == 0]
pos_bad = [(r[0], r[7]) for r in rows if r[7] is not None and not (r[7][1] == 0 and r[7][2] == 0)]
print("POS_EXACT_MATCH_V chunks=%d list=%s" % (len(pos_ok), ",".join(pos_ok)))
print("POS_MISMATCH chunks=%s" % pos_bad)

# ---------- 5. c046 detail ----------
c046 = chunk_keys["c046"]
v46 = V & c046
pk46 = set(pq.read_table(F + "/pos_parts/bucket_c046.parquet", columns=["match_key"]).column("match_key").to_pylist())
miss46 = sorted(v46 - pk46)
print("C046 v_in=%d pos_distinct=%d v_missing_from_pos=%d pos_extra=%d sample_missing=%s" % (
    len(v46), len(pk46), len(miss46), len(pk46 - v46), miss46[:2]))
print("C046 missing_in_laptop=%d" % len([k for k in miss46 if k in lp]))
print("C046 pos_rows=%d" % pq.ParquetFile(F + "/pos_parts/bucket_c046.parquet").metadata.num_rows)
pr46 = pq.ParquetFile(F + "/pos_parts/bucket_c046.parquet").metadata
print("C046 row_groups=%d" % pr46.num_row_groups)
pr00 = pq.ParquetFile(F + "/pos_parts/bucket_c045.parquet").metadata
print("C045 rows=%d row_groups=%d" % (pr00.num_rows, pr00.num_row_groups))

# ---------- 6. timestamps ----------
paths = [
    F + "/bucket_matches.parquet", F + "/laptop_matches.parquet", F + "/training_matches_v1.parquet",
    F + "/export_clean/match_keys.json", X + "/match_keys.json", F + "/clean_report.json", F + "/clean_matches.parquet",
    B + "/data/c000/matches.parquet", B + "/data/c000/prep_report.json", B + "/data/c046/prep_report.json",
    B + "/data/c067/prep_report.json", F + "/pos_parts/bucket_c000.parquet", F + "/pos_parts/bucket_c046.parquet",
    F + "/holdout_b_keys.txt", F + "/split_manifest.json",
]
for p in paths:
    st = os.stat(p)
    print("MTIME %s %s bytes=%d" % (time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime)), p, st.st_size))

print("RESULT_JSON " + json.dumps(dict(
    U_chunk=len(U_chunk), U_bm=len(U_bm), U_lp=len(U_lp), T=len(T), gap=len(gap),
    export=len(export), V=len(V), EXC=len(EXC), E_bucket=len(export & bucket_keys), E_only_laptop=len(export - bucket_keys))))
