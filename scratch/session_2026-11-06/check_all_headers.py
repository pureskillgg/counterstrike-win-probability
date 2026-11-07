import json
import re
import sys
import glob
import collections
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")

SCR = r"C:\Users\<user>\AppData\Local\Temp\claude\44708acea14da7d4\047ee5fa-b35b-4b25-97b1-beaf96834ff2\scratchpad"
assign = json.load(open(SCR + r"\split_manifest.json", encoding="utf-8"))["assign"]

ROOT = r"G:\pureskill\csds-cs2\tomes"
paths = sorted(glob.glob(ROOT + r"\*\tome\csds\header\dataframe_*"))
print("header tables found:", len(paths))

def in_window(d):
    return d is not None and "2026-10-26" <= d[:10] <= "2026-10-31"

rows = {}          # key -> (client_name, server_name) for allowed rows, deduped
label_counts = collections.Counter()
window_excluded = 0
unmapped = 0
for p in paths:
    t = pq.read_table(p, columns=["key", "match_date", "client_name", "server_name"])
    ks = t.column("key").to_pylist()
    ds = t.column("match_date").to_pylist()
    cs = t.column("client_name").to_pylist()
    ss = t.column("server_name").to_pylist()
    for k, d, c, s in zip(ks, ds, cs, ss):
        lab = assign.get(k)
        if lab is None:
            unmapped += 1
            continue
        label_counts[lab] += 1
        if lab not in ("train", "val"):
            continue
        if in_window(d):
            window_excluded += 1
            continue
        rows.setdefault(k, (c, s))

print("raw header row labels (all tables, incl. dups):", dict(label_counts))
print("rows not in manifest (excluded):", unmapped)
print("train/val rows in Oct 26-31 window (excluded):", window_excluded)
print("distinct allowed train/val matches:", len(rows))

cns = [v[0] for v in rows.values()]
sns = [v[1] for v in rows.values()]

print("\n[client_name] distinct:", len(set(cns)))
print("  equals 'SourceTV Demo':", sum(1 for v in cns if v == "SourceTV Demo"))
print("  not equal to that constant:", sum(1 for v in cns if v != "SourceTV Demo"))
print("  null:", sum(1 for v in cns if v is None))

def cls_of(v):
    return v

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
print("\n[server_name] distinct:", len(set(sns)), "| class counts:", dict(cls))
pats = {
    "email": re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    "url": re.compile(r"https?://|www\.", re.I),
    "ipv4": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    "steamid": re.compile(r"\b7656119\d{10}\b|\[U:1:\d+\]|STEAM_\d:\d:\d+"),
}
for name, pat in pats.items():
    print("  %s hits:" % name, sum(1 for v in sns if v and pat.search(v)))

# token analysis: alphabetic tokens that occur in exactly one distinct server_name
tok_df = collections.Counter()
names = [v for v in set(sns) if v]
for v in names:
    toks = set(re.findall(r"[A-Za-z]{3,}", v))
    for t_ in toks:
        tok_df[t_] += 1
single = [t_ for t_, n in tok_df.items() if n == 1]
print("\nalphabetic tokens (3+ letters) across distinct server names:", len(tok_df))
print("  tokens occurring in exactly one distinct server_name:", len(single))
multi_dom = tok_df.most_common(12)
print("  document frequencies of 12 most common tokens (values withheld):", [n for t_, n in multi_dom])
# shapes of the names that carry a singleton token (no values printed)
def shape(s):
    o = []
    for ch in s:
        if ch.isdigit(): o.append("9")
        elif ch.isupper(): o.append("A")
        elif ch.islower(): o.append("a")
        else: o.append(ch)
    return "".join(o)
single_set = set(single)
names_with_single = [v for v in names if any(t_ in single_set for t_ in re.findall(r"[A-Za-z]{3,}", v))]
print("  distinct server_names containing a singleton token:", len(names_with_single))
print("  shapes of those (token-collapsed, top 10):")
def tok_shape(v):
    return re.sub(r"[A-Za-z]{3,}", lambda m: ("W" if m.group(0) in single_set else "T"), v)
cnt = collections.Counter(re.sub(r"\d+", "9", tok_shape(v)) for v in names_with_single)
for shp, n in cnt.most_common(10):
    print("    n=%d  %s" % (n, shp))
