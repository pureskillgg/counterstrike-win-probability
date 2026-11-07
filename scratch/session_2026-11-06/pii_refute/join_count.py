import json, os, collections
ROOT = "G:/pureskill/csds-cs2"
man = json.load(open(ROOT + "/deliverables/email_package_2026-10-19_review/results/split_manifest.json", encoding="utf-8"))
assign = man["assign"]
scope = [k for k, v in assign.items() if v in ("train", "val")]
total_by_year = collections.Counter(k.split("/")[1] for k in scope)
key_path_exists = 0
folder_exists = 0
personal_exists = 0
by_year = collections.Counter()
for k in scope:
    p = "E:/csds_final/" + k            # .../<match id>/csds (channel file)
    folder = os.path.dirname(p)         # .../<match id>
    if os.path.exists(p):
        key_path_exists += 1
        by_year[k.split("/")[1]] += 1
    if os.path.isdir(folder):
        folder_exists += 1
        if os.path.isdir(os.path.join(folder, "player_personal")):
            personal_exists += 1
print("SCOPE_KEYS", len(scope), "by_year", dict(total_by_year))
print("KEY_PATH_EXISTS_ON_E", key_path_exists, "by_year", dict(by_year))
print("MATCH_FOLDER_EXISTS_ON_E", folder_exists)
print("PLAYER_PERSONAL_DIR_PRESENT", personal_exists)
