import json, glob, os, re, collections
import numpy as np
import pyarrow.parquet as pq

B = "/home/sagemaker-user/adam/builder"
FIN = B + "/final"
EXP = FIN + "/export_clean_v2"


def cno(c):
    return int(c[1:])


# 1. per-chunk prep reports
reports = {}
for d in sorted(glob.glob(B + "/data/c[0-9][0-9][0-9]")):
    c = os.path.basename(d)
    reports[c] = json.load(open(os.path.join(d, "prep_report.json")))
A = sorted(reports)
lo = [c for c in A if cno(c) <= 46]
hi = [c for c in A if cno(c) >= 47]
print("PREP reports:", len(A), A[0], A[-1], "| lo", len(lo), "hi", len(hi))


def sm(keys, f):
    return sum((reports[c].get(f) or 0) for c in keys)


for nm, ks in [("all", A), ("c000-c046", lo), ("c047-c067", hi)]:
    print("PREP", nm, "chunks", len(ks),
          "matches", sm(ks, "matches"),
          "use_matches", sm(ks, "use_matches"),
          "use_rounds", sm(ks, "use_rounds"),
          "rounds_total", sm(ks, "rounds_total"),
          "dup_matches_removed", sm(ks, "dup_matches_removed"))

# 2. chunk matches.parquet rows and keys
chunk_keys = {}
for c in A:
    t = pq.read_table(f"{B}/data/{c}/matches.parquet", columns=["match_key"])
    chunk_keys[c] = t.column("match_key").to_pylist()
bad = [(c, reports[c]["matches"], len(chunk_keys[c])) for c in A if reports[c]["matches"] != len(chunk_keys[c])]
print("CHUNK matches.parquet rows != prep matches (first 10):", bad[:10], "count", len(bad))
all_ck = [k for c in A for k in chunk_keys[c]]
print("CHUNK union rows", len(all_ck), "distinct", len(set(all_ck)))

bm = pq.read_table(FIN + "/bucket_matches.parquet", columns=["match_key"]).column("match_key").to_pylist()
print("BUCKET_MATCHES rows", len(bm), "distinct", len(set(bm)),
      "| in bucket_matches not in chunks", len(set(bm) - set(all_ck)),
      "| in chunks not in bucket_matches", len(set(all_ck) - set(bm)))

# 3. identity recompute log per chunk (matches column)
rec = {}
for line in open(B + "/logs/identity_all.log", errors="replace"):
    m = re.match(r"^(c\d{3}) matches (\d+) skipped (\d+) rows (\d+)", line)
    if m:
        rec[m.group(1)] = (int(m.group(2)), int(m.group(4)))
print("RECOMPUTE log chunks", len(rec), "first", min(rec), "last", max(rec))
print("RECOMPUTE use_matches mismatches vs prep:",
      [(c, rec[c][0], reports[c]["use_matches"]) for c in rec if rec[c][0] != reports[c]["use_matches"]][:10])
print("RECOMPUTE sum matches", sum(v[0] for v in rec.values()), "sum rows", sum(v[1] for v in rec.values()))
print("RECOMPUTE chunks in log but not in lo", sorted(set(rec) - set(lo)), "| lo not in log", sorted(set(lo) - set(rec)))

# 4. pos_parts coverage
pp_rows = {}
pp_keys = {}
for c in A:
    p = f"{FIN}/pos_parts/bucket_{c}.parquet"
    if os.path.exists(p):
        pp_rows[c] = pq.ParquetFile(p).metadata.num_rows
        pp_keys[c] = set(pq.read_table(p, columns=["match_key"]).column("match_key").to_pylist())
print("POS_PARTS files", len(pp_rows), "chunks", sorted(pp_rows)[0], "..", sorted(pp_rows)[-1],
      "rows sum", sum(pp_rows.values()))
print("POS_PARTS chunks missing from pos_parts:", [c for c in A if c not in pp_rows])
pp_all = set().union(*pp_keys.values()) if pp_keys else set()
print("POS_PARTS distinct match_keys", len(pp_all))

# 5. export keys
mk = json.load(open(EXP + "/match_keys.json"))
mks = set(mk)
print("EXPORT match_keys len", len(mk), "distinct", len(mks))

key2chunk = {}
for c in A:
    for k in chunk_keys[c]:
        key2chunk.setdefault(k, c)

grp = collections.Counter()
for k in mk:
    if k not in key2chunk:
        grp["not_in_bucket_chunks"] += 1
    elif cno(key2chunk[k]) <= 46:
        grp["bucket_lo_c000_c046"] += 1
    else:
        grp["bucket_hi_c047_c067"] += 1
print("EXPORT keys by source group", dict(grp))

exp_per_chunk = {c: sum(1 for k in chunk_keys[c] if k in mks) for c in A}
print("EXPORT per-chunk count vs prep use_matches, mismatches (first 10):",
      [(c, exp_per_chunk[c], reports[c]["use_matches"]) for c in A if exp_per_chunk[c] != reports[c]["use_matches"]][:10])
print("EXPORT bucket keys total", sum(exp_per_chunk.values()),
      "lo", sum(exp_per_chunk[c] for c in lo), "hi", sum(exp_per_chunk[c] for c in hi))
print("EXPORT keys in pos_parts keys", len(mks & pp_all), "| export bucket keys not in pos_parts",
      len([k for k in mk if k in key2chunk and k not in pp_all]))

cl = pq.read_table(FIN + "/clean_matches.parquet",
                   columns=["match_key", "src", "in_scope", "n_ok_rounds", "month"]).to_pydict()
ck = {k: (s, i, n, m) for k, s, i, n, m in zip(cl["match_key"], cl["src"], cl["in_scope"], cl["n_ok_rounds"], cl["month"])}
print("CLEAN_matches rows", len(cl["match_key"]), "src", dict(collections.Counter(cl["src"])))
print("EXPORT keys found in clean_matches", sum(k in ck for k in mk), "missing", sum(k not in ck for k in mk))
print("EXPORT src (clean_matches)", dict(collections.Counter(ck[k][0] for k in mk if k in ck)))
print("EXPORT in_scope (clean_matches)", dict(collections.Counter(ck[k][1] for k in mk if k in ck)))
print("EXPORT month (clean_matches)", dict(collections.Counter(ck[k][3] for k in mk if k in ck)))

# 6. exclusions
hb = [l.strip() for l in open(FIN + "/holdout_b_keys.txt") if l.strip()]
print("holdout_b_keys.txt lines", len(hb), "distinct", len(set(hb)), "| export keys in holdout B", len(mks & set(hb)))

mani = json.load(open(FIN + "/split_manifest_v2.json"))
assign = mani["assign"]
print("MANIFEST assign type", type(assign).__name__, "len", len(assign))
if isinstance(assign, dict):
    print("MANIFEST assign label counts", dict(collections.Counter(assign.values())))
    print("EXPORT keys by manifest label", dict(collections.Counter(assign.get(k, "MISSING") for k in mk)))

# 7. npy shapes and internal consistency
for f in sorted(glob.glob(EXP + "/*.npy")):
    a = np.load(f, mmap_mode="r")
    print("NPY", os.path.basename(f), a.shape, a.dtype)

ms = np.load(EXP + "/match_split.npy", mmap_mode="r")
vals, cnts = np.unique(np.asarray(ms), return_counts=True)
print("MATCH_SPLIT values", dict(zip(vals.tolist(), cnts.tolist())))
if isinstance(assign, dict):
    labs = [assign.get(k, "MISSING") for k in mk]
    ct = collections.Counter(zip(np.asarray(ms).tolist(), labs))
    print("CROSSTAB match_split value vs manifest label", dict(ct))

off = np.load(EXP + "/offsets.npy", mmap_mode="r")
print("OFFSETS len", off.shape[0], "first", int(off[0]), "last", int(off[-1]),
      "monotone", bool(np.all(np.diff(np.asarray(off)) >= 0)))
pos_off = np.load(EXP + "/pos_offsets.npy", mmap_mode="r")
print("POS_OFFSETS len", pos_off.shape[0], "first", int(pos_off[0]), "last", int(pos_off[-1]),
      "monotone", bool(np.all(np.diff(np.asarray(pos_off)) >= 0)))
steps = np.load(EXP + "/steps_alive.npy", mmap_mode="r")
print("STEPS_ALIVE len", steps.shape[0], "| offsets last == steps len:", int(off[-1]) == steps.shape[0])
rm = np.load(EXP + "/round_match.npy", mmap_mode="r")
bc = np.bincount(np.asarray(rm), minlength=len(mk))
print("ROUND_MATCH len", rm.shape[0], "min", int(rm.min()), "max", int(rm.max()),
      "matches with zero rounds", int((bc == 0).sum()))
print("ROUND_MATCH rounds per split", dict(collections.Counter(np.asarray(ms)[np.asarray(rm)].tolist())))
