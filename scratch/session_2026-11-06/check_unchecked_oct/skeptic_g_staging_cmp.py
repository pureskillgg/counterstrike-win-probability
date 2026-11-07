import os, csv, json, collections
import pyarrow.parquet as pq
import pyarrow as pa

E = "E:/csds_final/csds"
ROOT = "G:/pureskill/csds-cs2"
W = "C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/check_unchecked_oct/"

targets = []
for r in csv.DictReader(open(W + "undetermined_matches.csv", encoding="utf-8")):
    targets.append(r["match_key"])
tset = set(targets)

stg = os.path.join(ROOT, "staging", "csds")
c = collections.Counter()
mismatch_examples = []
staging_keys_ok = []


def tuples(path, cols):
    t = pq.read_table(path, columns=cols)
    return sorted(zip(*[t.column(x).to_pylist() for x in cols]), key=lambda z: tuple(str(v) for v in z))


# 1) staging folders present for targets
for mk in targets:
    p = mk.split("/")
    sdir = os.path.join(stg, p[2], p[3], p[4], p[5] if False else p[4])
    # staging layout: staging/csds/YYYY/MM/DD/<uuid>/...
    sdir = os.path.join(stg, p[1], p[2], p[3], p[4])
    if os.path.isdir(sdir):
        c["staging_folder_present"] += 1
        c["staging_folder_for_" + p[1] + "/" + p[2]] += 1
    else:
        c["staging_folder_absent"] += 1

# 2) compare staging player_info vs E: player_info on the staging-covered keys
for mk in targets:
    p = mk.split("/")
    sdir = os.path.join(stg, p[1], p[2], p[3], p[4])
    if not os.path.isdir(sdir):
        continue
    spi = os.path.join(sdir, "player_info")
    epi = os.path.join(E, p[1], p[2], p[3], p[4], "player_info")
    if not (os.path.isfile(spi) and os.path.isfile(epi)):
        c["stg_or_E_player_info_missing"] += 1
        continue
    try:
        sch = pq.read_schema(spi)
    except Exception as e:
        c["stg_player_info_unreadable"] += 1
        continue
    if "team_code" not in sch.names:
        c["stg_player_info_no_team_code"] += 1
        continue
    cols = ["round", "player_id", "team_code"]
    a = tuples(spi, cols)
    b = tuples(epi, cols)
    c["stg_vs_E_compared"] += 1
    if a == b:
        c["stg_vs_E_identical_round_player_team"] += 1
    else:
        c["stg_vs_E_differs"] += 1
        if len(mismatch_examples) < 5:
            mismatch_examples.append({"key": mk, "n_stg": len(a), "n_E": len(b)})

# 3) swarm E1 cache player_info.parquet vs E:
sw = os.path.join(ROOT, "logs", "swarm", "E1", "cache", "player_info.parquet")
t = pq.read_table(sw, columns=["match_key", "round", "player_id", "team_code"])
by_key = collections.defaultdict(list)
for mk, rd, pid, tc in zip(t.column("match_key").to_pylist(), t.column("round").to_pylist(),
                           t.column("player_id").to_pylist(), t.column("team_code").to_pylist()):
    if mk in tset:
        by_key[mk].append((rd, pid, tc))
c["swarm_target_keys"] = len(by_key)
for mk, rows in by_key.items():
    p = mk.split("/")
    epi = os.path.join(E, p[1], p[2], p[3], p[4], "player_info")
    if not os.path.isfile(epi):
        c["swarm_key_no_E"] += 1
        continue
    b = tuples(epi, ["round", "player_id", "team_code"])
    a = sorted(rows, key=lambda z: tuple(str(v) for v in z))
    if a == b:
        c["swarm_vs_E_identical"] += 1
    else:
        c["swarm_vs_E_differs"] += 1
        if len(mismatch_examples) < 10:
            mismatch_examples.append({"key": mk, "swarm_rows": len(a), "E_rows": len(b), "src": "swarm"})

# 4) which target days does swarm cover, and which do staging cover
sw_days = collections.Counter(mk.split("/")[2] + "/" + mk.split("/")[3] for mk in by_key)
out = {
    "counters": dict(c),
    "swarm_target_keys_by_day": dict(sorted(sw_days.items())),
    "mismatch_examples": mismatch_examples,
}
with open(W + "skeptic_g_staging_cmp.json", "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1), flush=True)
print("DONE", flush=True)
