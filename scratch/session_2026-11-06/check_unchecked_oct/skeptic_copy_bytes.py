import os, csv, json, hashlib, random, collections

E = "E:/csds_final/csds"
ROOT = "G:/pureskill/csds-cs2"
W = "C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/check_unchecked_oct/"

targets = [r["match_key"] for r in csv.DictReader(open(W + "undetermined_matches.csv", encoding="utf-8"))]
staging_dirs = sorted(d for d in os.listdir(ROOT) if d.startswith("staging") and os.path.isdir(os.path.join(ROOT, d)))


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


# file counts per target folder on E:
nfiles = collections.Counter()
for mk in targets:
    p = mk.split("/")
    d = os.path.join(E, p[1], p[2], p[3], p[4])
    nfiles[len([x for x in os.listdir(d) if os.path.isfile(os.path.join(d, x))])] += 1

random.seed(11)
sample = random.sample(targets, 80)
c = collections.Counter()
ex = []
for mk in sample:
    p = mk.split("/")
    y, m, d, u = p[1], p[2], p[3], p[4]
    edir = os.path.join(E, y, m, d, u)
    copies = [os.path.join(ROOT, sd, "csds", y, m, d, u) for sd in staging_dirs
              if os.path.isdir(os.path.join(ROOT, sd, "csds", y, m, d, u))]
    e_names = set(x for x in os.listdir(edir) if os.path.isfile(os.path.join(edir, x)))
    for cdir in copies:
        g_names = set(x for x in os.listdir(cdir) if os.path.isfile(os.path.join(cdir, x)))
        c["copies_compared"] += 1
        if g_names == e_names:
            c["same_file_name_set"] += 1
        else:
            c["file_name_set_differs"] += 1
        common = [n for n in e_names & g_names]
        same_bytes = all(md5(os.path.join(edir, n)) == md5(os.path.join(cdir, n)) for n in common[:43])
        if same_bytes:
            c["all_common_files_byte_identical"] += 1
        else:
            c["some_common_files_differ"] += 1
            if len(ex) < 5:
                ex.append({"key": mk, "copy": cdir})
        if len(ex) >= 0:
            pass
        break  # first copy only per key for speed

out = {"E_files_per_target_folder": dict(nfiles), "sample_keys": len(sample), "counters": dict(c), "examples": ex}
with open(W + "skeptic_copy_bytes.json", "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1), flush=True)
print("DONE", flush=True)
