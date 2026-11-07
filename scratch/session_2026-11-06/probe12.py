import polars as pl, numpy as np, json, glob, os, time
t0 = time.time()
F = "/home/sagemaker-user/adam/builder/final"; E = F + "/export_clean_v2"
KEYS = ["match_key", "round", "tick"]
keys = json.load(open(E + "/match_keys.json"))
RM = np.load(E + "/round_match.npy"); RN = np.load(E + "/round_num.npy")
ekr = pl.DataFrame({"match_key": [keys[m] for m in RM.tolist()], "round": RN.astype(np.int64)})
ekm = pl.DataFrame({"match_key": keys})
tot = dict(rows_a=0, rows_b=0, not_ek=0, ek_not_round=0, in_round=0, bad_files=0)
fails = []
for pa in sorted(glob.glob(F + "/pos_parts/bucket_c*.parquet")):
    name = os.path.basename(pa)
    pb = F + "/pos_check_all/" + name
    a = pl.read_parquet(pa).sort(KEYS)
    b = pl.read_parquet(pb).sort(KEYS)
    same_cols = a.columns == b.columns
    same_shape = a.height == b.height
    same_keys = same_shape and bool((a.select(KEYS).to_numpy() == b.select(KEYS).to_numpy()).all())
    bad = {}
    if same_keys:
        for col in a.columns:
            if col in KEYS: continue
            x = a[col].cast(pl.Float64).to_numpy(); y = b[col].cast(pl.Float64).to_numpy()
            d = int((~((np.isnan(x) & np.isnan(y)) | (x == y))).sum())
            if d: bad[col] = d
    ok = same_cols and same_shape and same_keys and not bad
    if not ok:
        tot["bad_files"] += 1; fails.append((name, same_cols, same_shape, same_keys, bad))
    a2 = a.with_columns(pl.col("round").cast(pl.Int64))
    n_ek = a2.join(ekm, on="match_key", how="semi").height
    n_rd = a2.join(ekr, on=["match_key", "round"], how="semi").height
    tot["rows_a"] += a.height; tot["rows_b"] += b.height
    tot["not_ek"] += a.height - n_ek; tot["ek_not_round"] += n_ek - n_rd; tot["in_round"] += n_rd
print("files compared", len(glob.glob(F + "/pos_parts/bucket_c*.parquet")), "bad files", tot["bad_files"], "fails", fails[:5])
print("totals", tot, "time", round(time.time() - t0, 1))
print("breakdown check: in_round + ek_not_round + not_ek =", tot["in_round"] + tot["ek_not_round"] + tot["not_ek"], "(should equal rows_a)")
print("expected in_round (19504876 covered + 0) ; not_ek should be 25327 ; ek_not_round should be 6274")
