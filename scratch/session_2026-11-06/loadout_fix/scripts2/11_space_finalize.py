import hashlib, os, shutil, json
F = "/home/sagemaker-user/adam/builder/final/"
OUT = F + "export_clean_v3_loadout/"
STAGE = F + "export_clean_v3_loadout_stage/"
shutil.copy(F + "export_clean_v2/offsets.npy", OUT + "offsets.npy")
shutil.copy(STAGE + "parity_s3_rows.parquet", OUT + "parity_s3_rows.parquet")
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()
res = {}
for name in ["loadout_step_i16.npy", "loadout_equip_i32.npy", "offsets.npy", "match_keys.json", "built_match_keys.json", "match_loadout_status.csv", "README.json", "parity_s3_rows.parquet"]:
    res[name] = {"bytes": os.path.getsize(OUT + name), "sha256": sha(OUT + name)}
json.dump(res, open(OUT + "files_sha256.json", "w"), indent=1)
print(json.dumps(res, indent=1))
shutil.rmtree(STAGE)
print("stage removed")
