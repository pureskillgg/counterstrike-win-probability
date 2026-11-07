import json, sys, re, glob, os, collections
import pyarrow.parquet as pq
M = json.load(open(sys.argv[1], encoding="utf-8"))
print("RULE:", M["rule"])
print("ASSIGN_N", len(M["assign"]))
print("ASSIGN_LABEL_COUNTS", dict(collections.Counter(M["assign"].values())))
print("FROZEN_FROM", M["frozen_from"])
root = sys.argv[2]
files = sorted(glob.glob(os.path.join(root, "dataframe_*")))
uuid_re = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
shapes = collections.Counter()
hits = collections.Counter()
for f in files:
    t = pq.read_table(f, columns=["match_key"])
    for v in t.column("match_key").to_pylist():
        if v is None:
            shapes["NULL"] += 1; continue
        shapes["len%d" % len(v)] += 1
        if uuid_re.search(v):
            shapes["has_uuid"] += 1
        if v in M["assign"]:
            hits["direct_assign_key"] += 1
        else:
            m = uuid_re.search(v)
            if m:
                cand = [k for k in M["assign"] if m.group(0) in k]
                hits["uuid_in_assign:%d" % min(len(cand), 2)] += 1
            else:
                hits["no_match"] += 1
print("MATCH_KEY_SHAPES", dict(shapes))
print("MATCH_HITS", dict(hits))
