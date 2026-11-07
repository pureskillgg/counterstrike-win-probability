import os, json, hashlib, collections
E = "E:/csds_final/csds"
W = "C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/check_unchecked_oct/"
sample = json.load(open(W + "skeptic_oct21_sample.json"))
c = collections.Counter()
examples = []
for s in sample:
    folder = os.path.join(E, "2026", "10", s["day"], s["uuid"])
    if not os.path.isdir(folder):
        c["folder_missing_in_E"] += 1
        continue
    local = {}
    for name in os.listdir(folder):
        p = os.path.join(folder, name)
        if os.path.isfile(p):
            local[name] = os.path.getsize(p)
    s3f = s["files"]
    c["folders_compared"] += 1
    if set(local) == set(s3f):
        c["same_file_name_set"] += 1
    else:
        c["file_name_set_differs"] += 1
        if len(examples) < 3:
            examples.append((s["uuid"], sorted(set(s3f) ^ set(local))[:5]))
    size_mismatch = [n for n in s3f if n in local and local[n] != s3f[n]]
    if not size_mismatch:
        c["all_common_file_sizes_equal"] += 1
    else:
        c["size_mismatch_folders"] += 1
    if s["player_info_md5"] is not None and "player_info" in local:
        with open(os.path.join(folder, "player_info"), "rb") as fh:
            md5 = hashlib.md5(fh.read()).hexdigest()
        if md5 == s["player_info_md5"]:
            c["player_info_md5_equal"] += 1
        else:
            c["player_info_md5_differs"] += 1
            if len(examples) < 6:
                examples.append(("md5diff", s["uuid"]))
out = {"counts": dict(c), "examples": examples, "sample_size": len(sample)}
print(json.dumps(out, indent=1))
