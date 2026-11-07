import json, os, hashlib
import numpy as np
B2 = "/home/sagemaker-user/adam/builder/final/export_clean_v2"
B3 = "/home/sagemaker-user/adam/builder/final/export_clean_v3_loadout"
print("== v2 README.json ==")
print(open(os.path.join(B2, "README.json")).read())
print("== v2 loadout_names.json ==")
print(open(os.path.join(B2, "loadout_names.json")).read())
print("== v2 maps.json ==")
print(open(os.path.join(B2, "maps.json")).read())
print("== npy headers v2 ==")
for f in ["round_match.npy", "round_freeze.npy", "round_num.npy", "round_label.npy", "round_map.npy", "round_plant.npy",
          "offsets.npy", "steps_alive.npy", "loadout_round.npy", "loadout_flag.npy", "match_split.npy"]:
    a = np.load(os.path.join(B2, f), mmap_mode="r")
    print(f, a.shape, a.dtype)
print("== npy headers v3 ==")
for f in ["loadout_step_i16.npy", "loadout_equip_i32.npy", "offsets.npy"]:
    a = np.load(os.path.join(B3, f), mmap_mode="r")
    print(f, a.shape, a.dtype)
print("== v3 keys ==")
print(sorted(os.listdir(B3)))
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()
print("v2 match_keys.json sha256", sha(os.path.join(B2, "match_keys.json")))
print("v3 match_keys.json sha256", sha(os.path.join(B3, "match_keys.json")))
print("v2 offsets sha256", sha(os.path.join(B2, "offsets.npy")))
print("v3 offsets sha256", sha(os.path.join(B3, "offsets.npy")))
