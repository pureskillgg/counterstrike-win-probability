import json, numpy as np, polars as pl
FIN = "/home/sagemaker-user/adam/builder/final"
V2 = FIN + "/export_clean_v2"
for c in ["c050", "c060"]:
    df = pl.read_parquet(f"{FIN}/pos_parity/bucket_{c}.parquet")
    print(c, "rows", df.height, "matches", df["match_key"].n_unique(), "cols", df.width)
    print(" schema:", {k: str(v) for k, v in df.schema.items()})
    print(" nulls:", {k: int(df[k].null_count()) for k in df.columns})
    print(" dup (match_key,round,tick):", df.group_by(["match_key", "round", "tick"]).len().filter(pl.col("len") > 1).height)
for f in ["pos_int8.npy", "pos_uint16.npy", "pos_uint8.npy", "pos_float16.npy"]:
    a = np.load(f"{V2}/{f}", mmap_mode="r")
    print(f, a.shape, a.dtype)
for f in ["pos_offsets.npy", "offsets.npy", "round_match.npy", "round_num.npy", "round_freeze.npy"]:
    a = np.load(f"{V2}/{f}", mmap_mode="r")
    print(f, a.shape, a.dtype)
keys = json.load(open(V2 + "/match_keys.json"))
print("match_keys", len(keys), "unique", len(set(keys)))
