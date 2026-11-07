# A2: inventory of the staged 2024 match folders (staging_2024_g junctions -> merged_2024 / source_2024).
# Read-only. Writes per-match summary to a scratch CSV and aggregate text to logs/swarm2/A2_inventory.log.
import os, gzip, json, csv, sys, collections, time
from concurrent.futures import ThreadPoolExecutor
BASE = r"G:\pureskill\csds-cs2"
STAGE = os.path.join(BASE, "staging_2024_g", "csds", "2024", "02")
SCR = sys.argv[1]
OUTCSV = os.path.join(SCR, "A2_inventory_staged.csv")
want = set(l.strip() for l in open(os.path.join(BASE, "scripts", "channels_build_nobom.txt"), encoding="utf-8-sig") if l.strip())
jobs = []
for day in sorted(os.listdir(STAGE)):
    dp = os.path.join(STAGE, day)
    for mid in sorted(os.listdir(dp)):
        jobs.append((day, mid, os.path.join(dp, mid)))
print("staged match folders:", len(jobs), flush=True)
def check(job):
    day, mid, mp = job
    man = os.path.join(mp, "csds")
    rec = {"day": day, "match": mid, "manifest": 0, "declared": "", "n_declared": 0,
           "missing_decl": "", "decl_not_in_want": "", "zero_byte": "", "files": ""}
    if not os.path.isfile(man):
        rec["manifest"] = 0
        return rec
    rec["manifest"] = 1
    raw = open(man, "rb").read()
    if raw[:2] == b"\x1f\x8b": raw = gzip.decompress(raw)
    j = json.loads(raw)
    decl = [c["channel"] for c in j["channels"]]
    rec["declared"] = ";".join(sorted(decl)); rec["n_declared"] = len(decl)
    present = {}
    for e in os.scandir(mp):
        if e.is_file():
            present[e.name] = e.stat().st_size
    rec["files"] = ";".join(sorted(present))
    rec["missing_decl"] = ";".join(sorted(c for c in decl if c not in present and c in want))
    rec["zero_byte"] = ";".join(sorted(c for c in decl if c in present and present[c] == 0))
    rec["decl_not_in_want"] = ";".join(sorted(c for c in decl if c not in want))
    rec["sizes"] = json.dumps({k: v for k, v in present.items() if k in want or k == "header" or k == "csds"}, sort_keys=True)
    return rec
t0 = time.time()
with ThreadPoolExecutor(12) as ex:
    recs = list(ex.map(check, jobs))
print("checked", len(recs), f"{time.time()-t0:.0f}s", flush=True)
keys = ["day", "match", "manifest", "n_declared", "declared", "missing_decl", "decl_not_in_want", "zero_byte", "files", "sizes"]
with open(OUTCSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
    for r in recs: w.writerow({k: r.get(k, "") for k in keys})
print("no manifest:", sum(1 - r["manifest"] for r in recs))
print("manifest count by day:", dict(collections.Counter(r["day"] for r in recs if r["manifest"])))
dec = collections.Counter()
for r in recs:
    for c in r["declared"].split(";"):
        if c: dec[c] += 1
print("declared channel counts (matches declaring each channel):")
for c, n in sorted(dec.items()): print(f"  {c}: {n}")
print("matches with missing declared (in build list):", sum(1 for r in recs if r["missing_decl"]))
mm = collections.Counter(r["missing_decl"] for r in recs if r["missing_decl"])
print("missing_decl patterns:", mm.most_common(10))
print("matches with zero-byte declared channels:", sum(1 for r in recs if r["zero_byte"]))
zb = collections.Counter(c for r in recs for c in r["zero_byte"].split(";") if c)
print("zero-byte channel counts:", dict(zb))
nd = collections.Counter(r["n_declared"] for r in recs)
print("n_declared distribution:", dict(sorted(nd.items())))
