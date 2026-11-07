# verify_A1_1_misjoin.py
# Alternative-explanation test: are the flagged headers copies of another match's round totals (row misjoin)?
# For each match, check whether some OTHER match with the same max_rounds has round-derived totals equal to
# this header's (t, c) pair, in either orientation. Compare the flagged (B+C) rate with the agreeing (A) baseline.
import csv, os, collections
SCRATCH = r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad"
rows = list(csv.DictReader(open(os.path.join(SCRATCH, "verify_A1_1_permatch.csv"), encoding="utf-8")))
for r in rows:
    for c in ["mr", "S", "hdr_t", "hdr_c", "n", "rnd_t", "rnd_c"]:
        r[c] = int(r[c])
print("== verify_A1_1_misjoin ==")
# index: (mr, unordered round-derived pair) -> list of match keys
idx = collections.defaultdict(list)
for r in rows:
    idx[(r["mr"], tuple(sorted((r["rnd_t"], r["rnd_c"]))))].append(r["key"])
def hits(r):
    pair = tuple(sorted((r["hdr_t"], r["hdr_c"])))
    return [k for k in idx.get((r["mr"], pair), []) if k != r["key"]]
flag = [r for r in rows if r["cat"] in ("B_sum_eq_split_diff", "C_sum_short")]
A = [r for r in rows if r["cat"] == "A_agree"]
fh = sum(1 for r in flag if hits(r))
ah = sum(1 for r in A if hits(r))
print("flagged (B+C):", len(flag), "| with another match whose round totals equal this header pair (same max_rounds):", fh)
print("agreeing (A) baseline:", len(A), "| same test:", ah, "| rate A =", round(ah / len(A), 3), "| rate flagged =", round(fh / len(flag), 3))
# For C only, does the header pair equal the round totals of a match on the same key-day?
C = [r for r in rows if r["cat"] == "C_sum_short"]
same_day = 0
for r in C:
    for k in hits(r):
        if k.split("/")[3] == r["day"]:
            same_day += 1
            break
print("C matches with a same-day other match whose round totals equal the header pair:", same_day, "of", len(C))
# Header sum vs rounds of the match the header pair would point to: are those matches' n equal to header S?
print("done")
