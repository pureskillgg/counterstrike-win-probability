import json, os
from collections import Counter
S = "C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/loadout_fix/"
root = "E:/csds_final/csds/"
keys = json.load(open(S + "match_keys.json")); K = set(keys)
export_month = Counter(k[5:12] for k in keys)
found = {}   # key -> dict(files)
dirs_seen = 0; no_ps = []; sizes = {}
for dpath, dnames, fnames in os.walk(root):
    rel = os.path.relpath(dpath, root).replace("\\", "/")
    parts = rel.split("/")
    # match dir: 2026/MM/DD/<uuid>
    if len(parts) == 4 and parts[0] == "2026":
        dirs_seen += 1
        key = f"csds/{parts[0]}/{parts[1]}/{parts[2]}/{parts[3]}/csds"
        files = {f: os.path.getsize(os.path.join(dpath, f)) for f in fnames}
        found[key] = files
    elif rel.startswith("csds_tail_recovered"):
        for d in dnames:
            pass
        if len(fnames):
            print("tail files", rel, fnames[:5])
keys_in_raw = [k for k in found if k in K]
ps_ok = [k for k in keys_in_raw if found[k].get("player_status", 0) > 0]
print("match dirs under 2026 walk:", dirs_seen, "dir keys (unique):", len(found))
print("export keys with raw dir:", len(keys_in_raw), "with non-empty player_status:", len(ps_ok))
print("raw dir keys not in export:", len([k for k in found if k not in K]))
print("export keys with dir but no csds file:", len([k for k in keys_in_raw if "csds" not in found[k]]))
print("export keys with dir but no player_status:", len([k for k in keys_in_raw if "player_status" not in found[k]]))
print("dirs with player_status size 0:", len([k for k in found if found[k].get("player_status", None) == 0]))
print("file name set counts:", Counter(f for k in found for f in found[k]).most_common(40))
cov_m = Counter(k[5:12] for k in ps_ok)
print("covered by month (raw player_status):", {m: [cov_m.get(m, 0), export_month[m]] for m in sorted(export_month) if m >= "2026/09" or cov_m.get(m, 0)})
print("export months with raw dirs:", sorted(set(k[5:12] for k in keys_in_raw)))
print("ps size stats (bytes) min/median/max:", min(found[k]['player_status'] for k in ps_ok), sorted(found[k]['player_status'] for k in ps_ok)[len(ps_ok)//2], max(found[k]['player_status'] for k in ps_ok))
json.dump({"raw_dirs": len(found), "export_keys_with_dir": len(keys_in_raw), "export_keys_with_player_status": len(ps_ok), "by_month": {m: [cov_m.get(m, 0), export_month[m]] for m in sorted(export_month)}, "ps_keys": ps_ok}, open(S + "raw_e_coverage.json", "w"))
