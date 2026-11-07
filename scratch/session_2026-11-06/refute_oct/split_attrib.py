import polars as pl, sys, json, csv, zlib
from collections import Counter, defaultdict
sys.stdout.reconfigure(encoding="utf-8")
S = "C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/"
CK = S + "check_unchecked_oct/"
R = S + "refute_oct/"
G = "G:/pureskill/csds-cs2/data/"
man = json.load(open(G + "combined/split_manifest.json", encoding="utf-8"))["assign"]
frozen = set(json.load(open(S + "pipeline/../frozen_test_keys.json"))) if __import__("os").path.exists(S + "frozen_test_keys.json") else None
if frozen is None:
    print("no local frozen list at scratchpad root; checking pipeline folder", flush=True)
    import os
    p = S + "pipeline/frozen_test_keys.json"
    frozen = set(json.load(open(p))) if os.path.exists(p) else set()
print("frozen keys", len(frozen), flush=True)
batch = set(r["match_key"] for r in csv.DictReader(open(CK + "batch_v4_per_match.csv", encoding="utf-8")))
hb = set(l.strip() for l in open(G + "holdout_b/holdout_keys.txt", encoding="utf-8") if l.strip())
o2631 = set(pl.read_csv(R + "refute_oct2631_keys.csv")["match_key"].to_list())
print("sets batch", len(batch), "hb", len(hb), "oct2631", len(o2631), flush=True)
def code(k):
    if k in frozen: return "test"
    return man.get(k, "none")
def cat(k):
    if k in frozen: return "frozen"
    if k in batch: return "batch"
    if k in hb: return "holdoutB"
    if k in o2631: return "oct26_31"
    return "other"
for cache in ["cache_snapshot_2026.parquet", "cache_snapshot_2026_oct23_31.parquet"]:
    kc = pl.scan_parquet(G + "combined/" + cache).select("match_key").group_by("match_key").len().collect()
    tot = defaultdict(int); keys_by = defaultdict(int)
    for k, n in kc.iter_rows():
        tot[(cat(k), code(k))] += n
        keys_by[(cat(k), code(k))] += 1
    print("==", cache, "keys", kc.height, "rows", int(kc["len"].sum()), flush=True)
    lab = Counter()
    for (c, l), n in sorted(tot.items()):
        print("   cat=%-9s manifest_label=%-5s keys=%6d rows=%10d" % (c, l, keys_by[(c, l)], n), flush=True)
    for split in ["train", "val", "test"]:
        print("   TOTAL label", split, sum(v for (c, l), v in tot.items() if l == split), flush=True)
