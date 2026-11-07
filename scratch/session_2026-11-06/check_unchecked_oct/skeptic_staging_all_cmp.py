import os, csv, json, collections, time
import pyarrow.parquet as pq

E = "E:/csds_final/csds"
ROOT = "G:/pureskill/csds-cs2"
W = "C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/check_unchecked_oct/"

targets = [r["match_key"] for r in csv.DictReader(open(W + "undetermined_matches.csv", encoding="utf-8"))]
staging_dirs = sorted(d for d in os.listdir(ROOT) if d.startswith("staging") and os.path.isdir(os.path.join(ROOT, d)))
print("staging dirs", len(staging_dirs), staging_dirs, flush=True)

COLS = ["round", "player_id", "team_code"]


def rows_of(path):
    t = pq.read_table(path, columns=COLS)
    return sorted((tuple(str(v) for v in z) for z in zip(*[t.column(c).to_pylist() for c in COLS])))


t0 = time.time()
stats = collections.Counter()
per_key = []
diff_examples = []
for i, mk in enumerate(targets):
    p = mk.split("/")
    y, m, d, u = p[1], p[2], p[3], p[4]
    e_pi = os.path.join(E, y, m, d, u, "player_info")
    e_rows = rows_of(e_pi) if os.path.isfile(e_pi) else None
    copies = []
    for sd in staging_dirs:
        fdir = os.path.join(ROOT, sd, "csds", y, m, d, u)
        if os.path.isdir(fdir):
            copies.append((sd, fdir))
    stats["targets_with_any_G_copy"] += 1 if copies else 0
    stats["copies_found_total"] += len(copies)
    n_ident = 0
    n_diff = 0
    n_nopi = 0
    for sd, fdir in copies:
        g_pi = os.path.join(fdir, "player_info")
        if not os.path.isfile(g_pi):
            n_nopi += 1
            continue
        try:
            g_rows = rows_of(g_pi)
        except Exception as ex:
            stats["copy_player_info_unreadable"] += 1
            continue
        if e_rows is not None and g_rows == e_rows:
            n_ident += 1
        else:
            n_diff += 1
            if len(diff_examples) < 8:
                diff_examples.append({"key": mk, "copy": sd, "E_rows": len(e_rows) if e_rows else None, "G_rows": len(g_rows)})
    stats["copies_identical_to_E"] += n_ident
    stats["copies_differ_from_E"] += n_diff
    stats["copies_missing_player_info"] += n_nopi
    if n_ident > 0:
        stats["targets_with_identical_G_copy"] += 1
    elif n_diff > 0:
        stats["targets_with_only_differing_G_copy"] += 1
    else:
        stats["targets_with_no_comparable_G_copy"] += 1
    if i % 1000 == 0:
        print("progress", i, round(time.time() - t0, 1), dict(stats), flush=True)

out = {"targets": len(targets), "staging_dirs": staging_dirs, "stats": dict(stats),
       "diff_examples": diff_examples, "seconds": round(time.time() - t0, 1)}
with open(W + "skeptic_staging_all_cmp.json", "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1), flush=True)
print("DONE", flush=True)
