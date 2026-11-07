import json
import re
import sys
import glob
import collections
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")

SCR = r"C:\Users\13144\AppData\Local\Temp\claude\44708acea14da7d4\047ee5fa-b35b-4b25-97b1-beaf96834ff2\scratchpad"
assign = json.load(open(SCR + r"\split_manifest.json", encoding="utf-8"))["assign"]
ROOT = r"G:\pureskill\csds-cs2\tomes"

def in_window(d):
    return d is not None and "2026-10-26" <= d[:10] <= "2026-10-31"

valve_re = re.compile(r"^Valve Counter-Strike 2 [a-z0-9_\-]+ Server \(srcds\d+[^)]*\)$")
faceit_re = re.compile(r"^FACEIT\.com")

rows = {}
platform_of = {}
for p in sorted(glob.glob(ROOT + r"\*\tome\csds\header\dataframe_*")):
    t = pq.read_table(p, columns=["key", "match_date", "server_name", "platform"])
    for k, d, s, pf in zip(t.column("key").to_pylist(), t.column("match_date").to_pylist(),
                           t.column("server_name").to_pylist(), t.column("platform").to_pylist()):
        if assign.get(k) not in ("train", "val") or in_window(d):
            continue
        rows.setdefault(k, s)
        platform_of.setdefault(k, pf)

other_keys = [k for k, s in rows.items() if not (valve_re.match(s) or faceit_re.match(s))]
print("other-class matches:", len(other_keys))
print("labels of other-class:", dict(collections.Counter(assign[k] for k in other_keys)))
print("platform of other-class (counts):", dict(collections.Counter(platform_of[k] for k in other_keys)))
print("platform of all allowed (counts):", dict(collections.Counter(platform_of[k] for k in rows)))

def tok_shape(v):
    v = re.sub(r"[A-Za-z]{2,}", lambda m: "W" if m.group(0)[0].isupper() else "w", v)
    v = re.sub(r"\d+", "9", v)
    return v

print("\nshape of other-class names (alpha tokens: W=Capitalised, w=lower; digits 9):")
for shp, n in collections.Counter(tok_shape(rows[k]) for k in other_keys).most_common():
    print("  n=%d  %s" % (n, shp))

print("\nshape of FACEIT template names (top 8):")
for shp, n in collections.Counter(tok_shape(v) for v in rows.values() if faceit_re.match(v)).most_common(8):
    print("  n=%d  %s" % (n, shp))
print("shape of Valve template names (top 5):")
for shp, n in collections.Counter(tok_shape(v) for v in rows.values() if valve_re.match(v)).most_common(5):
    print("  n=%d  %s" % (n, shp))
