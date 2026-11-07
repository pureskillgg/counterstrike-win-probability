import os

BASE = "C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad"
VOLUME_LIST = os.path.join(BASE, "volume_cs2data_objects.txt")
CANDIDATES = [
    ("staging_cs2data_gz", os.path.join(BASE, "local_staging_cs2data_gz_files.txt")),
    ("source_2024", os.path.join(BASE, "local_source_2024_files.txt")),
    ("merged_2024", os.path.join(BASE, "local_merged_2024_files.txt")),
    ("evac_2026-10-13", os.path.join(BASE, "local_evac_2026-10-13_files.txt")),
]
PREFIXES = ["cs2data", "cs2data_gz"]
SHOW = 30


def read_pairs(path):
    out = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        for raw in fh:
            line = raw.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise SystemExit("bad line in %s: %r" % (path, line[:200]))
            out.append((parts[0], int(parts[1])))
    return out


def split_prefix(key):
    head, _, rest = key.partition("/")
    return head, rest


volume = []
for key, size in read_pairs(VOLUME_LIST):
    prefix, rel = split_prefix(key)
    volume.append((prefix, rel, key, size))

print("VOLUME ROWS:", len(volume))
for p in PREFIXES:
    rows = [v for v in volume if v[0] == p]
    print("  prefix %-12s files=%d bytes=%d" % (p, len(rows), sum(v[3] for v in rows)))

for cname, cpath in CANDIDATES:
    pairs = read_pairs(cpath)
    local = dict(pairs)
    local_base = {}
    for rel in local:
        local_base.setdefault(os.path.basename(rel), []).append(rel)
    print("\n== CANDIDATE %s  (local list rows=%d, bytes=%d)" % (cname, len(local), sum(local.values())))
    for p in PREFIXES:
        vol_rows = [v for v in volume if v[0] == p]
        matched = []
        name_only = []
        unmatched = []
        for prefix, rel, key, size in vol_rows:
            if rel in local:
                if local[rel] == size:
                    matched.append((key, size))
                else:
                    name_only.append((key, size, local[rel]))
            else:
                unmatched.append((key, size))
        matched_bytes = sum(s for _, s in matched)
        print("  [%s] volume_files=%d matched_name_and_size=%d bytes_matched=%d" % (
            p, len(vol_rows), len(matched), matched_bytes))
        if name_only:
            print("    name match but size differs: %d" % len(name_only))
            for key, vs, ls in name_only[:SHOW]:
                print("      %s volume=%d local=%d" % (key, vs, ls))
        print("    unmatched volume files: %d" % len(unmatched))
        for key, _ in unmatched[:SHOW]:
            print("      " + key)
        if len(unmatched) > SHOW:
            print("      ... and %d more" % (len(unmatched) - SHOW))
        # basename-only hits (informational, not counted as matches)
        base_hits = []
        for key, _ in unmatched:
            b = os.path.basename(key)
            if b in local_base:
                base_hits.append((key, local_base[b]))
        if base_hits:
            print("    basename-only hits (not counted): %d" % len(base_hits))
            for key, lp in base_hits[:SHOW]:
                print("      %s -> local %s" % (key, ", ".join(lp)))
        # local files not claimed by any matched volume row
        claimed = set(key for key, _ in matched)
        leftover = []
        for rel, size in pairs:
            ok = False
            for p2 in PREFIXES:
                pass
            # leftover relative to this prefix only
            vol_rels = set(v[1] for v in vol_rows)
            if rel not in vol_rels:
                leftover.append(rel)
        if p == PREFIXES[-1]:
            pass
        print("    local files with no path match under %s: %d" % (p, len(leftover)))
