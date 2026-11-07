"""Inspect QA mismatches in the sampled schema JSON (read-only, scratchpad only)."""
import collections
import json

SAMPLE = (
    r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
    r"\7308a8e7-a47e-4034-b140-5c511acb0aa4\scratchpad\schema_sample.json"
)
data = json.load(open(SAMPLE, encoding="utf-8"))
f24 = data["folders"]["2024"]
f26 = data["folders"]["2026"]

e = f24[0]["channels"]["bomb_action"]
print("2024 manifest names:", [c[0] for c in e["manifest_columns"]])
print("2024 parquet names: ", [c[0] for c in e.get("fields", [])])
print("2024 manifest types:", [(c[0], c[1]) for c in e["manifest_columns"]][:8])
print("2024 parquet types:", e.get("fields", [])[:8])
print()

e = f26[0]["channels"]["bomb_action"]
print("2026 manifest names:", [c[0] for c in e["manifest_columns"]])
print("2026 parquet names: ", [c[0] for c in e.get("fields", [])])
print("2026 manifest types:", [(c[0], c[1]) for c in e["manifest_columns"]][:8])
print("2026 parquet types:", e.get("fields", [])[:8])
print()

# manifest type diffs: show the first few columns with differing manifest types
mt = collections.defaultdict(lambda: collections.defaultdict(set))
for year, folders in (("2024", f24), ("2026", f26)):
    for f in folders:
        for ch, ent in f["channels"].items():
            for name, mtype, _n, _o in ent["manifest_columns"]:
                mt[(year, ch)][name].add(str(mtype))
ex = []
for ch in sorted({k[1] for k in mt}):
    a = mt.get(("2024", ch), {})
    b = mt.get(("2026", ch), {})
    for name in sorted(set(a) & set(b)):
        if a[name] != b[name]:
            ex.append((ch, name, sorted(a[name]), sorted(b[name])))
print("manifest type diff examples:", len(ex))
for row in ex[:12]:
    print("  ", row)

# within-year parquet type variance examples
pt = collections.defaultdict(lambda: collections.defaultdict(set))
for f in f26:
    for ch, ent in f["channels"].items():
        for name, typ, _n in ent.get("fields", []):
            pt[ch][name].add(typ)
var = [(ch, n, sorted(t)) for ch in pt for n, t in pt[ch].items() if len(t) > 1]
print("2026 within-year parquet type variance:", len(var))
for row in var[:10]:
    print("  ", row)

# how many 2024 folders have a manifest/parquet order or content mismatch, first example
for f in f24[:1]:
    for ch, ent in f["channels"].items():
        m = [c[0] for c in ent["manifest_columns"]]
        p = [c[0] for c in ent.get("fields", [])]
        if m != p:
            print("first mismatch:", ch, "manifest_len", len(m), "parquet_len", len(p))
            print("  only manifest:", sorted(set(m) - set(p))[:10])
            print("  only parquet:", sorted(set(p) - set(m))[:10])
            print("  same set:", set(m) == set(p), " sorted equal:", sorted(m) == sorted(p))
            break
