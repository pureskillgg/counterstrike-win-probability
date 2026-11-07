import os, re, csv, collections, json, sys
UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
ROOT = "E:/csds_final/csds"
SCR = "C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/check_unchecked_oct/"
uuid_dirs = {}   # uuid -> list of full paths
n_nonuuid = 0
def walk(path, depth):
    global n_nonuuid
    try:
        it = list(os.scandir(path))
    except Exception as e:
        return
    n_nonuuid += 1
    for e in it:
        if not e.is_dir(follow_symlinks=False): continue
        if UUID.match(e.name):
            uuid_dirs.setdefault(e.name, []).append(e.path)
        elif depth < 8:
            walk(e.path, depth + 1)
walk(ROOT, 0)
print("non-uuid dirs visited", n_nonuuid, "distinct uuid dirs", len(uuid_dirs), "total uuid paths", sum(len(v) for v in uuid_dirs.values()), flush=True)
rows = list(csv.DictReader(open(SCR + "undetermined_matches.csv")))
hits = []; by_src = collections.Counter(); paths = {}
for r in rows:
    k = r["match_key"]; u = k.split("/")[4]; y, m, d = k.split("/")[1:4]
    if u in uuid_dirs:
        hits.append(k); paths[k] = uuid_dirs[u]
        for p in uuid_dirs[u]:
            by_src[p.replace("\\", "/").split("/csds_final/csds/")[1].split("/")[0] + "/" + ("/".join(p.replace("\\","/").split("/csds_final/csds/")[1].split("/")[1:3]) if "/" in p.split("csds_final/csds/")[1] else "")] += 1
print("target keys with uuid folder in E:", len(hits), flush=True)
print("by-src:", by_src.most_common(20), flush=True)
json.dump({"n_uuid_dirs": len(uuid_dirs), "target_hits": len(hits), "hit_keys_sample": hits[:20],
           "paths_sample": {k: paths[k] for k in hits[:20]}, "by_src": by_src.most_common(50)},
          open(SCR + "local_e_walk.json", "w"), indent=1)
print("WROTE", flush=True)
