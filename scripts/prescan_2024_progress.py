import os, gzip, json, sys, csv, collections, time
ROOT = sys.argv[1]
CHANNELS = [l.strip() for l in open(sys.argv[2], encoding="utf8") if l.strip()] + ["header"]
OUT = sys.argv[3]
def parquet_ok(path):
    size = os.path.getsize(path)
    if size < 12:
        return f"size {size}"
    with open(path, "rb") as f:
        head = f.read(4)
        f.seek(-4, 2)
        tail = f.read(4)
    if head != b"PAR1" or tail != b"PAR1":
        return f"bad_magic head={head!r} tail={tail!r} size={size}"
    return None
rows = []
t0 = time.time()
n = 0
for day in sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))):
    dp = os.path.join(ROOT, day)
    mids = sorted(os.listdir(dp))
    print(f"day {day}: {len(mids)} entries, t={time.time()-t0:.0f}s", flush=True)
    for mid in mids:
        mp = os.path.join(dp, mid)
        n += 1
        if n % 100 == 0:
            print(f"  progress {n} entries, t={time.time()-t0:.0f}s", flush=True)
        if not os.path.isdir(mp):
            continue
        man = os.path.join(mp, "csds")
        if not os.path.isfile(man):
            rows.append((day, mid, "no_manifest", "", 0))
            continue
        try:
            raw = open(man, "rb").read()
            if raw[:2] == b"\x1f\x8b":
                raw = gzip.decompress(raw)
            declared = [c["channel"] for c in json.loads(raw)["channels"]]
        except Exception as e:
            rows.append((day, mid, "bad_manifest", str(e)[:120], 0))
            continue
        problems = []
        for ch in CHANNELS:
            if ch not in declared:
                continue
            p = os.path.join(mp, ch)
            if not os.path.isfile(p):
                problems.append(f"{ch}:missing")
                continue
            why = parquet_ok(p)
            if why:
                problems.append(f"{ch}:{why}")
        rows.append((day, mid, "problem" if problems else "ok", ";".join(problems)[:400], len(declared)))
with open(OUT, "w", newline="", encoding="utf8") as f:
    w = csv.writer(f)
    w.writerow(["day", "match_folder", "status", "detail", "declared_channels"])
    w.writerows(rows)
print("entries:", len(rows), dict(collections.Counter(r[2] for r in rows)), f"{time.time()-t0:.0f}s", flush=True)
bad = collections.Counter()
for r in rows:
    if r[2] == "problem":
        for item in r[3].split(";"):
            bad[item.split(":")[0]] += 1
print("problem channels:", dict(bad.most_common(15)), flush=True)
