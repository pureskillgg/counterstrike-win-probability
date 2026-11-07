# verify_A1_1_cohort.py
# Cohort table: flagged (B+C) vs agreeing (A) counts by header (build_num, rushb_version, ppp_version) triple,
# over all 1,752 matches; then, inside the cohort that holds every flagged match, a timing check on match_date.
import os, csv, collections
import polars as pl
ROOT = r"G:\pureskill\csds-cs2"
T = os.path.join(ROOT, "tomes", "2026-10-01_to_08", "tome", "csds")
SCRATCH = r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad"
cat = {r["key"]: (r["cat"], r["day"]) for r in csv.DictReader(open(os.path.join(SCRATCH, "verify_A1_1_permatch.csv"), encoding="utf-8"))}
h = pl.read_parquet(os.path.join(T, "header", "dataframe_00000"), columns=["key", "build_num", "rushb_version", "ppp_version", "match_date"])
print("== verify_A1_1_cohort ==")
print("build_num dtype:", h.schema["build_num"], "| ppp dtype:", h.schema["ppp_version"])
tab = collections.defaultdict(lambda: [0, 0])  # [flagged, agreeing]
rows_c = []
for k, b, rb, pp, md in h.iter_rows():
    c, d = cat[k]
    f = 1 if c in ("B_sum_eq_split_diff", "C_sum_short") else 0
    tab[(str(b), str(rb), str(pp))][0 if f else 1] += 1
    rows_c.append((k, str(b), str(rb), str(pp), str(md), f, d, c))
print("triple (build, rushb, ppp): flagged | agreeing (all days)")
for key, (f, a) in sorted(tab.items(), key=lambda x: -(x[1][0] + x[1][1])):
    print("  ", key, f, "|", a)
T3 = ("10924", "5.4.0", "8.5.1")
print("\nflagged total", sum(v[0] for v in tab.values()), "| flagged inside", T3, ":", tab[T3][0],
      "| agreeing inside", T3, ":", tab[T3][1], "| cohort size", sum(tab[T3]))
inside = [r for r in rows_c if (r[1], r[2], r[3]) == T3]
print("cohort key-days:", dict(collections.Counter(r[6] for r in inside)))
print("cohort OT matches (n not available in header; use category):", dict(collections.Counter(r[7] for r in inside)))
print("\ncohort timing: flagged fraction by match_date day (header):")
bydate = collections.defaultdict(lambda: [0, 0])
for r in inside:
    bydate[r[4][:10]][0 if r[5] else 1] += 1
for d in sorted(bydate):
    print("  ", d, "flagged", bydate[d][0], "| agreeing", bydate[d][1])
print("\ncohort timing: flagged vs agreeing sorted by match_date (F = flagged, . = agreeing):")
seq = "".join("F" if r[5] else "." for r in sorted(inside, key=lambda r: r[4]))
print("  ", seq)
# does the flag sequence look random in time? count runs
runs = 1 + sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
nF = seq.count("F"); nA = seq.count(".")
import math
exp_runs = 1 + 2 * nF * nA / (nF + nA)
print("runs in time order:", runs, "| expected runs if random:", round(exp_runs, 1), "| F", nF, ". ", nA)
