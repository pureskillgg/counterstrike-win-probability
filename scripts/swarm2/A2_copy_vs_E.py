# A2: G: staged copy vs E: source. Read-only on both trees.
#  Pass 1 (all days 02-25 ok matches + all day-01 merged): file list + size for wanted files, G vs E.
#  Pass 2 (sample): SHA-256 of every wanted file for a stratified sample.
# Day 01 E sources: A = E:\csds_extracted\csds\2024\02\01 ; C = E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02\01
# Day 02-25 E source: E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02\<DD>
import os, sys, csv, random, hashlib, time, collections
from concurrent.futures import ThreadPoolExecutor
BASE = r"G:\pureskill\csds-cs2"
G_STAGE = os.path.join(BASE, "staging_2024_g", "csds", "2024", "02")
E_TAIL = r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02"
E_A = r"E:\csds_extracted\csds\2024\02\01"
E_C = r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02\01"
SCR = sys.argv[1]; MODE = sys.argv[2]  # MODE = sizes | sample
want = set(l.strip() for l in open(os.path.join(BASE, "scripts", "channels_build_nobom.txt"), encoding="utf-8-sig") if l.strip()) | {"header", "csds"}
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
def listing(d):
    out = {}
    if not os.path.isdir(d): return None
    for e in os.scandir(d):
        if e.is_file() and e.name in want:
            out[e.name] = e.stat().st_size
    return out
jobs = []
for day in sorted(os.listdir(G_STAGE)):
    for mid in sorted(os.listdir(os.path.join(G_STAGE, day))):
        jobs.append((day, mid))
def e_dirs(day, mid):
    if day == "01":
        return [os.path.join(E_A, mid), os.path.join(E_C, mid)]
    return [os.path.join(E_TAIL, day, mid)]
if MODE == "sizes":
    def cmp(job):
        day, mid = job
        g = listing(os.path.join(G_STAGE, day, mid))
        es = [listing(d) for d in e_dirs(day, mid)]
        if day == "01":
            # A is primary; C used for add_from_C. Report A-equality and whether each G file matches A or C.
            ea, ec = es
            bad = []
            for n, sz in g.items():
                ok_a = ea is not None and ea.get(n) == sz
                ok_c = ec is not None and ec.get(n) == sz
                if not (ok_a or ok_c): bad.append(f"{n}:g={sz}:A={None if ea is None else ea.get(n)}:C={None if ec is None else ec.get(n)}")
            return day, mid, len(g), (";".join(bad))[:500], 1 if ea is None else 0
        e = es[0]
        if e is None:
            return day, mid, len(g), "E_DIR_MISSING", 0
        bad = []
        for n in set(g) | set(e):
            if g.get(n) != e.get(n):
                bad.append(f"{n}:g={g.get(n)}:e={e.get(n)}")
        return day, mid, len(g), (";".join(sorted(bad)))[:500], len(e)
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(cmp, jobs))
    print("compared matches:", len(res), f"{time.time()-t0:.0f}s", flush=True)
    bad = [r for r in res if r[3]]
    print("matches with any size/name difference (G vs E wanted files):", len(bad))
    print("by day:", dict(collections.Counter(r[0] for r in bad)))
    with open(os.path.join(SCR, "A2_size_compare.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["day", "match", "n_wanted_G", "diff", "E_count"]); w.writerows(res)
    print("DONE sizes", flush=True)
else:
    rng = random.Random(20261013)
    picks = []
    tail_days = [d for d in sorted({j[0] for j in jobs}) if d != "01"]
    for d in tail_days:
        ms = [j for j in jobs if j[0] == d]
        picks += rng.sample(ms, min(6, len(ms)))
    d1 = [j for j in jobs if j[0] == "01"]
    picks += rng.sample(d1, 12)
    print("sample matches:", len(picks), flush=True)
    rows = []
    for day, mid in picks:
        g_root = os.path.join(G_STAGE, day, mid)
        for n in sorted(os.listdir(g_root)):
            gp = os.path.join(g_root, n)
            if not os.path.isfile(gp) or n not in want: continue
            gs = os.path.getsize(gp); gh = sha(gp)
            if day == "01":
                cands = {"A": os.path.join(E_A, mid, n), "C": os.path.join(E_C, mid, n)}
            else:
                cands = {"E": os.path.join(E_TAIL, day, mid, n)}
            info = {}
            for tag, p in cands.items():
                if os.path.isfile(p):
                    es = os.path.getsize(p)
                    info[tag] = (es, sha(p) if es == gs else None)
                else:
                    info[tag] = None
            # equality: sha compare if sizes equal; otherwise mismatch
            status = []
            for tag, v in info.items():
                if v is None: status.append(f"{tag}:absent")
                elif v[0] != gs: status.append(f"{tag}:size_diff({v[0]} vs {gs})")
                else: status.append(f"{tag}:{'sha_equal' if v[1] == gh else 'SHA_DIFF'}")
            rows.append((day, mid, n, gs, ";".join(status)))
    with open(os.path.join(SCR, "A2_sample_sha.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["day", "match", "file", "g_size", "status"]); w.writerows(rows)
    tot = len(rows)
    sha_eq = sum(1 for r in rows if "sha_equal" in r[4])
    diff = [r for r in rows if "SHA_DIFF" in r[4] or "size_diff" in r[4]]
    print("files compared:", tot, "sha_equal_or_matched:", sha_eq, "diff_or_absent_rows:", len(diff), flush=True)
    for r in diff[:30]: print("  ", r, flush=True)
    print("DONE sample", flush=True)
