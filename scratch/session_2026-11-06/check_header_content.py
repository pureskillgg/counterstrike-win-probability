import json
import re
import sys
import collections
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")

SCR = r"C:\Users\13144\AppData\Local\Temp\claude\44708acea14da7d4\047ee5fa-b35b-4b25-97b1-beaf96834ff2\scratchpad"
MANIFEST = SCR + r"\split_manifest.json"
HDR = sys.argv[1]

m = json.load(open(MANIFEST, encoding="utf-8"))
assign = m["assign"]

tbl = pq.read_table(HDR, columns=["key", "match_date", "client_name", "server_name"])
keys = tbl.column("key").to_pylist()
dates = tbl.column("match_date").to_pylist()
cn = tbl.column("client_name").to_pylist()
sn = tbl.column("server_name").to_pylist()

def in_window(d):
    # Oct 26-31 2026 is off limits
    return d is not None and d[:10] >= "2026-10-26" and d[:10] <= "2026-10-31"

allowed = []
for i, k in enumerate(keys):
    lab = assign.get(k)
    if lab not in ("train", "val"):
        continue
    if in_window(dates[i]):
        continue
    allowed.append(i)

print("rows total:", len(keys), "| allowed train/val outside Oct 26-31:", len(allowed))
print("allowed by label:", dict(collections.Counter(assign[keys[i]] for i in allowed)))

def shape(s):
    out = []
    for ch in s:
        if ch.isdigit():
            out.append("9")
        elif ch.isupper():
            out.append("A")
        elif ch.islower():
            out.append("a")
        elif ch == " ":
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)

# ---- client_name ----
cns = [cn[i] for i in allowed]
print("\n[client_name] n=%d distinct=%d" % (len(cns), len(set(cns))))
print("  equals 'SourceTV Demo' (boolean count):", sum(1 for v in cns if v == "SourceTV Demo"))
print("  null count:", sum(1 for v in cns if v is None))
print("  length counts:", dict(collections.Counter(len(v) for v in cns if v is not None)))
print("  shape counts (letters A/a, digits 9):", dict(collections.Counter(shape(v) for v in cns if v is not None)))

# ---- server_name ----
sns = [sn[i] for i in allowed]
print("\n[server_name] n=%d distinct=%d" % (len(sns), len(set(sns))))
valve_re = re.compile(r"^Valve Counter-Strike 2 [a-z0-9_\-]+ Server \(srcds\d+[^)]*\)$")
faceit_re = re.compile(r"^FACEIT\.com")
cls = collections.Counter()
other = []
for v in sns:
    if v is None:
        cls["null"] += 1
    elif valve_re.match(v):
        cls["valve_template"] += 1
    elif faceit_re.match(v):
        cls["faceit_prefix"] += 1
    else:
        cls["other"] += 1
        other.append(v)
print("  class counts:", dict(cls))

# indicator checks (booleans / counts only)
pats = {
    "email_like": re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    "url_like": re.compile(r"https?://|www\.", re.I),
    "ipv4_like": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    "steamid_like": re.compile(r"\b7656119\d{10}\b|\[U:1:\d+\]|STEAM_\d:\d:\d+"),
    "phone_like": re.compile(r"\+?\d[\d\- ]{8,}\d"),
}
for name, p in pats.items():
    print("  %s hits:" % name, sum(1 for v in sns if v and p.search(v)))

print("  other-class shape counts (top 15, shape only):")
for shp, n in collections.Counter(shape(v) for v in other).most_common(15):
    print("    n=%d  %s" % (n, shp))
print("  other-class length quartiles:", sorted(len(v) for v in other)[len(other)//4] if other else None,
      sorted(len(v) for v in other)[len(other)//2] if other else None,
      sorted(len(v) for v in other)[(3*len(other))//4] if other else None)
print("  other-class contains a space (count):", sum(1 for v in other if " " in v))
print("  other-class contains apostrophe (count):", sum(1 for v in other if "'" in v))
print("  other-class contains digit (count):", sum(1 for v in other if any(c.isdigit() for c in v)))
