import sys, csv, json, re, collections
csvp, hold_p, froz_p = sys.argv[1], sys.argv[2], sys.argv[3]
hold = set(json.load(open(hold_p, encoding="utf-8"))["match_keys"])
froz = set(json.load(open(froz_p, encoding="utf-8")))
def oct26_31(s):
    m = re.search(r"2026-10-(\d{2})", s or "")
    return bool(m) and 26 <= int(m.group(1)) <= 31
rows = list(csv.DictReader(open(csvp, encoding="utf-8", errors="replace")))
kept = [r for r in rows if r["match_key"] not in hold and r["match_key"] not in froz and not oct26_31(r["date"])]
print("total rows:", len(rows), "| off-limit excluded:", len(rows) - len(kept), "| analysed:", len(kept))
print("split values:", dict(collections.Counter(r["split"] for r in kept)))
print("src distinct:", len(set(r["src"] for r in kept)))
def mask(s):
    s = re.sub(r"[0-9]", "9", s); s = re.sub(r"[a-z]", "a", s); s = re.sub(r"[A-Z]", "A", s)
    return re.sub(r"(.)\1{3,}", lambda m: m.group(1) + "{%d+}" % len(m.group(0)), s)[:90]
print("src masks (top):", collections.Counter(mask(r["src"]) for r in kept).most_common(5))
print("reason distinct:", len(set(r["reason"] for r in kept)))
print("reason masks (top):", collections.Counter(mask(r["reason"]) for r in kept).most_common(5))
pats = {"steamid64": r"7656119\d{10}", "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "ipv4": r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "url": r"https?://", "win_user_path": r"(?i)[A-Z]:\+Users\+[^\\s]+", "unix_home": r"/(?:home|Users)/[^/\s]+", "steam": r"(?i)steam", "uuid": r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"}
for k, rx in pats.items():
    n = sum(1 for r in kept for c in r.values() if c and re.search(rx, c))
    print("pattern %-14s cell hits=%d" % (k, n))
# UUID-embedded src check: do src values contain a match key or a path with any non-uuid name token?
print("src contains csds-style match key path:", sum(1 for r in kept if re.search(r"csds/", r["src"])))
