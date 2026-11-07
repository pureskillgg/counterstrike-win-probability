import json, os, re, subprocess
from collections import Counter
import pyarrow.parquet as pq

B = "/home/sagemaker-user/adam/builder"
F = B + "/final"
E = F + "/export_clean"
X = F + "/export_clean_v2"
S3 = "s3://pureskillgg-csgo-production-sagemaker-0/adam/tomes"
OCT = re.compile(r"^csds/2026/10/(2[6-9]|3[01])/")


def s3cat(uri):
    r = subprocess.run(["aws", "s3", "cp", uri, "-"], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def lines_set(text):
    return set(l.strip() for l in text.splitlines() if l.strip())


def cols_present(path, want):
    names = pq.read_schema(path).names
    return [c for c in want if c in names], names


def usable_keys(tbl):
    out = set()
    for k, a, b, n in zip(tbl["match_key"], tbl["dup_keeper"], tbl["in_scope"], tbl["n_ok_rounds"]):
        if a and b and n and n > 0:
            out.add(k)
    return out


# ---------- export and v1 key sets ----------
export_list = json.load(open(X + "/match_keys.json"))
export = set(export_list)
v1_list = json.load(open(E + "/match_keys.json"))
v1 = set(v1_list)
print("EXPORT_V2 list_len=%d unique=%d" % (len(export_list), len(export)))
print("EXPORT_V1 list_len=%d unique=%d" % (len(v1_list), len(v1)))
print("export_minus_v1=%d" % len(export - v1))
empties = v1 - export
print("v1_minus_export (empty-removed)=%d" % len(empties))

# ---------- holdout sets ----------
ho_txt = s3cat(S3 + "/holdout_2026-10-26_31/match_keys.txt")
ho = lines_set(ho_txt)
print("HOLDOUT_TOME s3_bytes=%d keys=%d oct_window=%d" % (len(ho_txt), len(ho), sum(1 for k in ho if OCT.match(k))))
hb = lines_set(open(F + "/holdout_b_keys.txt").read())
print("HOLDOUT_B keys=%d oct_window=%d" % (len(hb), sum(1 for k in hb if OCT.match(k))))

assign = json.load(open(F + "/split_manifest.json"))["assign"]
test_keys = set(k for k, v in assign.items() if v == "test")
print("SPLIT_MANIFEST assign=%d test=%d" % (len(assign), len(test_keys)))

# ---------- laptop / bucket / v1 tables ----------
lp = F + "/laptop_matches.parquet"
bp = F + "/bucket_matches.parquet"
tp = F + "/training_matches_v1.parquet"
want = ["match_key", "map_name", "dup_keeper", "in_scope", "n_ok_rounds", "src"]
lcols, lnames = cols_present(lp, want)
bcols, bnames = cols_present(bp, want)
tcols, tnames = cols_present(tp, want)
print("LAPTOP_SCHEMA", lnames)
print("BUCKET_SCHEMA", bnames)
print("TRAIN_V1_SCHEMA", tnames)
lt = pq.read_table(lp, columns=lcols).to_pydict()
bt = pq.read_table(bp, columns=bcols).to_pydict()
tt = pq.read_table(tp, columns=tcols).to_pydict()
laptop = set(lt["match_key"])
bucket_mt = set(bt["match_key"])
print("LAPTOP_MATCHES rows=%d distinct=%d" % (len(lt["match_key"]), len(laptop)))
print("BUCKET_MATCHES rows=%d distinct=%d" % (len(bt["match_key"]), len(bucket_mt)))
print("TRAIN_V1 rows=%d distinct=%d" % (len(tt["match_key"]), len(set(tt["match_key"]))))
if "src" in tt:
    print("TRAIN_V1 src rows", dict(Counter(tt["src"])))
usable_l = usable_keys(lt)
usable_bm = usable_keys(bt)
usable_t = usable_keys(tt)
print("USABLE laptop=%d bucket_matches=%d train_v1=%d" % (len(usable_l), len(usable_bm), len(usable_t)))
print("TRAIN_V1 usable rows = %d" % sum(1 for a, b, n in zip(tt["dup_keeper"], tt["in_scope"], tt["n_ok_rounds"]) if a and b and n and n > 0))
print("TRAIN_V1 keys_not_usable=%d usable_l_or_b_not_in_train_v1=%d" % (
    len(set(tt["match_key"]) - usable_t), len((usable_l | usable_bm) - set(tt["match_key"]))))

mp = {}
for k, m in zip(lt["match_key"], lt["map_name"]):
    mp.setdefault(k, m)
for k, m in zip(bt["match_key"], bt["map_name"]):
    mp.setdefault(k, m)
mirage = set(k for k in test_keys if mp.get(k) == "de_mirage")
excluded = mirage | hb
print("MIRAGE_TEST=%d HOLDOUT_B=%d overlap=%d excluded_union=%d" % (len(mirage), len(hb), len(mirage & hb), len(excluded)))

# ---------- per chunk ----------
rows = []
bucket_union = set()
usable_b_union = set()
usable_b_sum = 0
mt_rows_sum = 0
status_sum = 0
prep_matches_sum = 0
prep_use_sum = 0
prep_dup_sum = 0
pos_missing = []
print("CHUNK status prep_matches mt_rows mt_distinct dup_keeper in_scope usable prep_use_matches prep_use_rounds use_keys_rounds pos_rows pos_distinct ex_prep ex_use ex_pos v1_in empties_in ho_in hb_in mirage_in oct_in laptop_in prep_dup_removed mt_dup_removed")
for i in range(68):
    c = "c%03d" % i
    d = "%s/data/%s" % (B, c)
    prep = json.load(open(d + "/prep_report.json"))
    st_txt = s3cat("%s/chunks/%s/STATUS_DONE.json" % (S3, c))
    st = json.loads(st_txt)["matches"] if st_txt else None
    mt = pq.read_table(d + "/matches.parquet", columns=["match_key", "dup_keeper", "in_scope", "n_ok_rounds"]).to_pydict()
    kl = mt["match_key"]
    kset = set(kl)
    n_mt = len(kl)
    dup_k = sum(1 for a in mt["dup_keeper"] if a)
    in_sc = sum(1 for a in mt["in_scope"] if a)
    uc = usable_keys(mt)
    rt = pq.read_table(d + "/rounds.parquet", columns=["match_key", "use"]).to_pydict()
    use_keys = set(k for k, u in zip(rt["match_key"], rt["use"]) if u)
    pp = "%s/pos_parts/bucket_%s.parquet" % (F, c)
    if os.path.exists(pp):
        pk = set(pq.read_table(pp, columns=["match_key"]).column("match_key").to_pylist())
        pos_rows = pq.ParquetFile(pp).metadata.num_rows
        pos_distinct = len(pk)
    else:
        pk = None
        pos_rows = None
        pos_distinct = None
        pos_missing.append(c)
    prep_scope = sum(v for k, v in prep.get("scope_by_era", {}).items() if k.endswith("|in_scope"))
    ex_prep = len(export & kset)
    ex_use = len(export & use_keys)
    ex_pos = len(export & pk) if pk is not None else None
    v1_in = len(v1 & kset)
    emp_in = len(empties & kset)
    ho_in = len(ho & kset)
    hb_in = len(hb & kset)
    mir_in = len(mirage & kset)
    oct_in = sum(1 for k in kl if OCT.match(k))
    lap_in = len(laptop & kset)
    prep_dup_removed = prep.get("dup_matches_removed")
    print("%s %s %s %d %d %d %d %d %s %s %d %s %s %d %d %s %d %d %d %d %d %d %d %s %d" % (
        c, st, prep.get("matches"), n_mt, len(kset), dup_k, in_sc, len(uc),
        prep.get("use_matches"), prep.get("use_rounds"), len(use_keys),
        pos_rows, pos_distinct, ex_prep, ex_use, ex_pos, v1_in, emp_in, ho_in, hb_in, mir_in, oct_in, lap_in,
        prep_dup_removed, n_mt - dup_k))
    rows.append(dict(c=c, status=st, prep_matches=prep.get("matches"), mt_rows=n_mt, dup_keeper=dup_k,
                     in_scope=in_sc, usable=len(uc), prep_use_matches=prep.get("use_matches"),
                     use_keys=len(use_keys), pos_rows=pos_rows, pos_distinct=pos_distinct,
                     ex_prep=ex_prep, ex_use=ex_use, empties_in=emp_in, ho_in=ho_in, hb_in=hb_in,
                     mirage_in=mir_in, oct_in=oct_in, prep_dup_removed=prep_dup_removed))
    bucket_union |= kset
    usable_b_union |= uc
    usable_b_sum += len(uc)
    mt_rows_sum += n_mt
    status_sum += st or 0
    prep_matches_sum += prep.get("matches") or 0
    prep_use_sum += prep.get("use_matches") or 0
    prep_dup_sum += prep_dup_removed or 0

print("POS_PARTS_MISSING_CHUNKS n=%d list=%s" % (len(pos_missing), ",".join(pos_missing)))

# ---------- global totals ----------
print("TOTAL status_matches_sum=%d prep_matches_sum=%d mt_rows_sum=%d bucket_union_distinct=%d cross_chunk_dup=%d" % (
    status_sum, prep_matches_sum, mt_rows_sum, len(bucket_union), mt_rows_sum - len(bucket_union)))
print("TOTAL prep_use_matches_sum=%d prep_dup_removed_sum=%d usable_sum=%d usable_union=%d" % (
    prep_use_sum, prep_dup_sum, usable_b_sum, len(usable_b_union)))
print("TOTAL zero_ex_prep_chunks=%s" % ",".join(r["c"] for r in rows if r["ex_prep"] == 0))
print("TOTAL zero_usable_chunks=%s" % ",".join(r["c"] for r in rows if r["usable"] == 0))
print("TOTAL ex_prep_sum=%d ex_use_sum=%d" % (sum(r["ex_prep"] for r in rows), sum(r["ex_use"] for r in rows)))

ex_b = export & bucket_union
ex_nb = export - bucket_union
print("EXPORT in_bucket_union=%d not_in_bucket=%d not_in_bucket_and_in_laptop=%d not_in_laptop_or_bucket=%d" % (
    len(ex_b), len(ex_nb), len(ex_nb & laptop), len(ex_nb - laptop - bucket_union)))
print("EXPORT in_laptop_total=%d in_laptop_and_bucket=%d" % (len(export & laptop), len(export & laptop & bucket_union)))
print("EXPORT in_usable_b=%d in_usable_l=%d" % (len(export & usable_b_union), len(export & usable_l)))

v1_b = v1 & bucket_union
v1_nb = v1 - bucket_union
print("V1 in_bucket=%d not_in_bucket=%d not_in_bucket_and_in_laptop=%d" % (len(v1_b), len(v1_nb), len(v1_nb & laptop)))
print("EMPTIES in_bucket=%d in_laptop=%d in_neither=%d" % (len(empties & bucket_union), len(empties & laptop), len(empties - bucket_union - laptop)))

for name, S in [("HOLDOUT_TOME", ho), ("HOLDOUT_B", hb), ("MIRAGE_TEST", mirage), ("EXCLUDED_UNION", excluded)]:
    print("%s size=%d in_bucket_union=%d in_usable_b=%d in_usable_l=%d in_v1=%d in_export=%d in_laptop=%d" % (
        name, len(S), len(S & bucket_union), len(S & usable_b_union), len(S & usable_l), len(S & v1),
        len(S & export), len(S & laptop)))
print("MIRAGE_AND_HOLDOUTB overlap_all=%d overlap_in_bucket=%d" % (len(mirage & hb), len(mirage & hb & bucket_union)))
print("OCT_WINDOW in_bucket_union=%d in_v1=%d in_export=%d" % (
    sum(1 for k in bucket_union if OCT.match(k)), sum(1 for k in v1 if OCT.match(k)), sum(1 for k in export if OCT.match(k))))

# bucket-side chain using actual sets
ex_excl_b = len(usable_b_union & excluded)
print("CHAIN_BUCKET raw_union=%d minus_nonusable=%d usable_union=%d minus_excluded_in_usable=%d minus_empties_in_bucket=%d expected_bucket_export=%d actual_export_in_bucket=%d" % (
    len(bucket_union), len(bucket_union) - len(usable_b_union), len(usable_b_union), ex_excl_b,
    len(empties & bucket_union), len(usable_b_union) - ex_excl_b - len(empties & bucket_union), len(ex_b)))
ex_excl_l = len(usable_l & excluded)
emp_l = len(empties & laptop)
print("CHAIN_LAPTOP usable_l=%d minus_excluded_in_usable_l=%d minus_empties_in_laptop=%d expected_laptop_export=%d actual_export_in_laptop=%d" % (
    len(usable_l), ex_excl_l, emp_l, len(usable_l) - ex_excl_l - emp_l, len(export & laptop)))
print("CHAIN_TOTAL usable_b+usable_l=%d minus_excluded_union_in_either=%d minus_empties=%d -> %d export=%d" % (
    len(usable_b_union | usable_l), len((usable_b_union | usable_l) & excluded), len(empties),
    len((usable_b_union | usable_l) - excluded) - len(empties), len(export)))

print("USER_CHAIN 67224-3463-653-1033+155-47=%d" % (67224 - 3463 - 653 - 1033 + 155 - 47))
print("RESULT_JSON " + json.dumps(dict(
    status_sum=status_sum, prep_matches_sum=prep_matches_sum, mt_rows_sum=mt_rows_sum,
    bucket_union=len(bucket_union), usable_b_union=len(usable_b_union), prep_use_sum=prep_use_sum,
    prep_dup_sum=prep_dup_sum, export=len(export), v1=len(v1), empties=len(empties),
    pos_missing=pos_missing)))
