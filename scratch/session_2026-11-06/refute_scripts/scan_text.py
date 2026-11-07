import sys, re, json, csv, collections, os
PATS = {
 "steamid64": r"7656119\d{10}",
 "steam_id_legacy": r"STEAM_[0-5]:[01]:\d+",
 "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
 "ipv4": r"\b(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}\b",
 "url": r"(?i)https?://",
 "win_user_path": r"(?i)[A-Z]:\\?Users\\?[^\\s\"']+",
 "unix_home_path": r"/home/[^/\s\"']+|/Users/[^/\s\"']+",
 "phone_like": r"\+?\d[\d\s\-()]{8,}\d",
 "steam_word": r"(?i)steam",
 "name_field_word": r"(?i)(player_?name|nickname|username|real_?name|display_?name)",
}
CRX = {k: re.compile(v) for k, v in PATS.items()}
def walk_strings(o, out, keys_out):
    if isinstance(o, dict):
        for k, v in o.items():
            keys_out.add(k)
            walk_strings(v, out, keys_out)
    elif isinstance(o, list):
        for v in o: walk_strings(v, out, keys_out)
    elif isinstance(o, str):
        out.append(o)
def scan_strings(name, strings):
    hits = collections.Counter()
    for s in strings:
        for k, rx in CRX.items():
            if rx.search(s): hits[k] += 1
    print("  %s: strings scanned=%d | pattern hit counts=%s" % (name, len(strings), dict(hits) if hits else "none"))
for p in sys.argv[1:]:
    base = os.path.basename(p)
    print("=== %s" % base)
    if p.endswith(".json"):
        obj = json.load(open(p, encoding="utf-8"))
        strs, keys = [], set()
        if isinstance(obj, dict) and "assign" in obj:
            # split map: scan keys (match_key strings) and values (split labels)
            strs = list(obj["assign"].keys()) + [str(v) for v in obj["assign"].values()]
            keys = set(obj.keys())
        else:
            walk_strings(obj, strs, keys)
        print("  key names:", sorted(keys)[:40] if len(keys) < 80 else "(many: %d)" % len(keys))
        scan_strings(base, strs)
    elif p.endswith(".csv"):
        rdr = csv.DictReader(open(p, encoding="utf-8", errors="replace"))
        cols = rdr.fieldnames
        print("  columns:", cols)
        per = {c: [] for c in cols}
        nrows = 0
        for row in rdr:
            nrows += 1
            for c in cols:
                per[c].append(row[c] or "")
        print("  rows:", nrows)
        for c in cols:
            scan_strings("col:%s" % c, per[c])
