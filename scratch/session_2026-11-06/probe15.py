import polars as pl, numpy as np, json, glob, os, time
t0 = time.time()
F = "/home/sagemaker-user/adam/builder/final"; E = F + "/export_clean_v2"
keys = json.load(open(E+"/match_keys.json"))
RM = np.load(E+"/round_match.npy"); RN = np.load(E+"/round_num.npy"); FZ = np.load(E+"/round_freeze.npy").astype(np.int64)
PO = np.load(E+"/pos_offsets.npy").astype(np.int64)
spec = json.load(open(E+"/pos_README.json"))
INT8, U16, U8, F16 = spec["int8"], spec["uint16"], spec["uint8_tenth_seconds"], spec["float16"]
pi = np.load(E+"/pos_int8.npy", mmap_mode="r"); pu = np.load(E+"/pos_uint16.npy", mmap_mode="r")
pu8 = np.load(E+"/pos_uint8.npy", mmap_mode="r"); pf = np.load(E+"/pos_float16.npy", mmap_mode="r")
N = len(RM)
ex = pl.DataFrame({"match_key": [keys[m] for m in RM.tolist()], "round": RN.astype(np.int64), "ridx": np.arange(N, dtype=np.int64), "freeze": FZ, "po": PO[:-1], "cnt": PO[1:]-PO[:-1]})
tot = dict(rows=0, unmapped=0, badgrid=0, badj=0, mism=0)
colbad = {}
for pa in sorted(glob.glob(F+"/pos_parts/bucket_c*.parquet"))[:5]:
    d = pl.read_parquet(pa).with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64))
    j = d.join(ex, on=["match_key","round"], how="left")
    um = j.filter(pl.col("ridx").is_null()).height
    m = j.filter(pl.col("ridx").is_not_null())
    tick = m["tick"].to_numpy(); frz = m["freeze"].to_numpy(); po = m["po"].to_numpy(); cnt = m["cnt"].to_numpy(); ridx = m["ridx"].to_numpy()
    dt = tick - frz
    badgrid = int((dt % 128 != 0).sum())
    jj = dt // 128
    badj = int(((jj < 0) | (jj >= cnt)).sum())
    pos_idx = po + jj
    ok = (dt % 128 == 0) & (jj >= 0) & (jj < cnt)
    mism = 0
    for c in INT8:
        a = m[c].cast(pl.Float64).to_numpy(); exp = np.where(np.isnan(a), -1, np.clip(np.round(a), 0, 5)).astype(np.int8)
        act = np.asarray(pi[pos_idx, INT8.index(c)])
        n = int((ok & (exp != act)).sum()); mism += n
        if n: colbad[c] = colbad.get(c, 0) + n
    for c in U16:
        a = m[c].cast(pl.Float64).to_numpy(); exp = np.where(np.isnan(a), 65535, np.clip(np.round(a), 0, 65534)).astype(np.uint16)
        act = np.asarray(pu[pos_idx, U16.index(c)])
        n = int((ok & (exp != act)).sum()); mism += n
        if n: colbad[c] = colbad.get(c, 0) + n
    for c in U8:
        a = m[c].cast(pl.Float64).to_numpy(); exp = np.where(np.isnan(a), 255, np.clip(np.round(a*10), 0, 254)).astype(np.uint8)
        act = np.asarray(pu8[pos_idx, U8.index(c)])
        n = int((ok & (exp != act)).sum()); mism += n
        if n: colbad[c] = colbad.get(c, 0) + n
    for c in F16:
        a = m[c].cast(pl.Float64).to_numpy(); exp = a.astype(np.float16)
        act = np.asarray(pf[pos_idx])
        same = (np.isnan(exp.astype(np.float32)) & np.isnan(act.astype(np.float32))) | (exp == act)
        n = int((ok & ~same).sum()); mism += n
        if n: colbad[c] = colbad.get(c, 0) + n
    os.path.basename(pa)
    tot["rows"] += d.height; tot["unmapped"] += um; tot["badgrid"] += badgrid; tot["badj"] += badj; tot["mism"] += mism
    print(os.path.basename(pa), "rows", d.height, "unmapped", um, "off-grid", badgrid, "bad j", badj, "value mismatches (all 24 feats)", mism, round(time.time()-t0,1), flush=True)
print("TOTAL (c000-c004)", tot, "column mismatches", colbad)
