import sys, re, json, collections
import pyarrow.parquet as pq
train = sys.argv[1]; holdout_json = sys.argv[2]; frozen_json = sys.argv[3]
t = pq.read_table(train)
d = t.to_pydict()
n = t.num_rows
hold = set(json.load(open(holdout_json, encoding="utf-8"))["match_keys"])
frozen = set(json.load(open(frozen_json, encoding="utf-8")))
dates = d["match_date"]
def is_oct26_31(s):
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", s or "")
    return bool(m) and m.group(1) == "2026" and m.group(2) == "10" and 26 <= int(m.group(3)) <= 31
keep = [i for i in range(n) if d["match_key"][i] not in hold and d["match_key"][i] not in frozen and not is_oct26_31(dates[i])]
print("total rows:", n, "| excluded holdout/frozen/Oct26-31 rows:", n - len(keep), "| analysed rows:", len(keep))
print("date format sample mask:", collections.Counter(re.sub(r"\d", "9", (dates[i] or "NONE")) for i in keep).most_common(3))
print("match_date year-month counts (analysed rows):", sorted(collections.Counter((dates[i] or "NONE")[:7] for i in keep).items())[:40])
for col in ["map_name", "match_type", "era", "scope_reason"]:
    vals = collections.Counter(d[col][i] for i in keep)
    print("categorical %s distinct=%d top=%s" % (col, len(vals), vals.most_common(12)))
def pattern_summary(col):
    vals = [d[col][i] for i in keep]
    lens = collections.Counter(len(v) if v is not None else -1 for v in vals)
    masks = collections.Counter(re.sub(r"(9+)", lambda m: "9{%d}"%len(m.group(0)), re.sub(r"[0-9]", "9", re.sub(r"[a-z]", "a", re.sub(r"[A-Z]", "A", v or "NONE")))) for v in vals)
    print("col %s: distinct=%d lengths(top)=%s" % (col, len(set(vals)), lens.most_common(5)))
    print("   top masks:", [(m[:120], c) for m, c in masks.most_common(4)])
    rx = {"steamid64": r"7656119\d{10}", "17digit": r"\d{17}", "steam": r"(?i)steam", "url_email": r"(?i)(http|www\.|@)", "uuid": r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"}
    print("   pattern hits:", {k: sum(1 for v in vals if v and re.search(p, v)) for k, p in rx.items()})
for col in ["wseq", "sig_hash", "dup_key", "dup_group", "scope_reason"]:
    if col in d:
        if isinstance(d[col][0], str):
            pattern_summary(col)
        else:
            vals = [d[col][i] for i in keep]
            print("col %s numeric: distinct=%d min/max not printed" % (col, len(set(vals))))
