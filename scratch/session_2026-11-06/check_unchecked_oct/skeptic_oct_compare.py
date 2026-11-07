import os, csv, json, collections

E = "E:/csds_final/csds"
W = "C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/check_unchecked_oct/"

# S3 folders (all of 2026/10) by day
s3 = collections.defaultdict(set)
with open(W + "skeptic_s3_folders.tsv") as f:
    next(f)
    for line in f:
        y, m, d, u = line.rstrip("\n").split("\t")
        if y == "2026" and m == "10":
            s3[d].add(u)

# local E: 2026/10 uuid folders by day
loc = {}
for d in range(1, 32):
    dd = f"{d:02d}"
    p = os.path.join(E, "2026", "10", dd)
    if os.path.isdir(p):
        loc[dd] = set(x for x in os.listdir(p) if os.path.isdir(os.path.join(p, x)))
    else:
        loc[dd] = set()

rows = []
tot = collections.Counter()
for dd in sorted(set(list(s3.keys()) + list(loc.keys()))):
    a = s3.get(dd, set())
    b = loc.get(dd, set())
    row = {
        "day": f"2026-10-{dd}",
        "s3_folders": len(a),
        "local_E_folders": len(b),
        "in_both": len(a & b),
        "s3_only_not_in_E": len(a - b),
        "E_only_not_in_s3": len(b - a),
    }
    rows.append(row)
    for k in ["s3_folders", "local_E_folders", "in_both", "s3_only_not_in_E", "E_only_not_in_s3"]:
        tot[k] += row[k]

o2125 = collections.Counter()
for r in rows:
    if r["day"][-2:] in ("21", "22", "23", "24", "25"):
        for k in ["s3_folders", "local_E_folders", "in_both", "s3_only_not_in_E", "E_only_not_in_s3"]:
            o2125[k] += r[k]

out = {"per_day": rows, "totals_oct": dict(tot), "totals_oct21_25": dict(o2125)}
with open(W + "skeptic_oct_compare.json", "w") as f:
    json.dump(out, f, indent=1)
for r in rows:
    print(r)
print("TOTALS OCT", dict(tot))
print("TOTALS OCT21-25", dict(o2125))
