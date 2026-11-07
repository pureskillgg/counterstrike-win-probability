import sys, re, collections
import pyarrow.parquet as pq
path = sys.argv[1]
t = pq.read_table(path, columns=["match_key"])
keys = set(x for x in t.column("match_key").to_pylist() if x is not None)
print("file:", path.split("/")[-1], "| rows:", t.num_rows, "| distinct match_key:", len(keys), "| nulls:", t.column("match_key").null_count)
lens = collections.Counter(len(k) for k in keys)
print("length distribution (len: count):", dict(sorted(lens.items())[:10]))
def mask(s):
    out = []
    for ch in s:
        if ch.isdigit(): out.append("9")
        elif ch.islower() and ch.isalpha(): out.append("a")
        elif ch.isupper() and ch.isalpha(): out.append("A")
        else: out.append(ch)
    # collapse runs to show structure without showing values
    return re.sub(r"(9+)", lambda m: "9{%d}" % len(m.group(0)), re.sub(r"(a+)", lambda m: "a{%d}" % len(m.group(0)), re.sub(r"(A+)", lambda m: "A{%d}" % len(m.group(0)), "".join(out))))
masks = collections.Counter(mask(k) for k in keys)
print("distinct structural masks:", len(masks))
for m, c in masks.most_common(8):
    print("  count=%-6d mask=%s" % (c, m))
pats = {
  "steamid64_17digit_7656119": r"7656119\d{10}",
  "any_17digit_run": r"\d{17}",
  "contains_steam_word": r"(?i)steam",
  "contains_url_or_email": r"(?i)(http|www\.|@)",
  "contains_path_sep": r"[\/]",
  "contains_dem_ext": r"(?i)\.dem",
  "contains_name_like_word": r"(?i)(player|nick|user|name)",
}
for name, rx in pats.items():
    n = sum(1 for k in keys if re.search(rx, k))
    print("pattern %-28s matching distinct keys: %d" % (name, n))
