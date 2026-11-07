import os, csv, json, re, collections
import pyarrow.parquet as pq

ROOT = "G:/pureskill/csds-cs2"
W = "C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/check_unchecked_oct/"
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")

targets_full = set()
target_uuids = set()
for r in csv.DictReader(open(W + "undetermined_matches.csv", encoding="utf-8")):
    targets_full.add(r["match_key"])
    target_uuids.add(r["match_key"].split("/")[4])

path_hits = []
uuid_path_hit_set = set()
n_paths = 0
for dp, dn, fn in os.walk(ROOT):
    if ".pixi" in dp or "holdout" in dp.lower():
        dn[:] = []
        continue
    for name in list(dn) + list(fn):
        n_paths += 1
        for u in UUID_RE.findall(name):
            if u in target_uuids:
                uuid_path_hit_set.add(u)
                if len(path_hits) < 20:
                    path_hits.append(os.path.join(dp, name))
    for name in fn:
        if False:
            pass
print("paths walked", n_paths, flush=True)
print("target uuids in any G: path name or dir name", len(uuid_path_hit_set), flush=True)

cands = [
    os.path.join(ROOT, "logs", "swarm", "E1", "cache", "player_info.parquet"),
    os.path.join(ROOT, "evac_2026-10-13", "data", "state_sample", "players_1hz.parquet"),
    os.path.join(ROOT, "data", "state_sample", "players_1hz.parquet"),
]
per_file = {}
for p in cands:
    if not os.path.isfile(p):
        per_file[p] = "missing"
        continue
    sch = pq.read_schema(p)
    info = {"cols": sch.names, "rows": pq.ParquetFile(p).metadata.num_rows}
    if "match_key" in sch.names:
        vals = set(v for v in pq.read_table(p, columns=["match_key"]).column("match_key").to_pylist() if v is not None)
        info["distinct_match_keys"] = len(vals)
        info["target_full_hits"] = len(vals & targets_full)
        info["target_uuid_hits"] = len(set(v.split("/")[4] for v in vals if v.count("/") >= 4) & target_uuids)
    per_file[p] = info
    print(p, info, flush=True)

out = {
    "paths_walked": n_paths,
    "target_uuids_in_G_path_names": len(uuid_path_hit_set),
    "path_hit_examples": path_hits,
    "team_code_candidates": per_file,
}
with open(W + "skeptic_g_coverage.json", "w") as f:
    json.dump(out, f, indent=1, default=str)
print("DONE", flush=True)
