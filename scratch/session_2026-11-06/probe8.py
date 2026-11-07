import numpy as np, json, polars as pl, collections, os, time
t0 = time.time()
E = "/home/sagemaker-user/adam/builder/final/export_clean_v2"
F = "/home/sagemaker-user/adam/builder/final"
B = "/home/sagemaker-user/adam/builder"
S = np.load(E+"/steps_alive.npy", mmap_mode='r'); O = np.load(E+"/offsets.npy")
RM = np.load(E+"/round_match.npy"); RN = np.load(E+"/round_num.npy")
L = np.load(E+"/loadout_round.npy", mmap_mode='r')
keys = json.load(open(E+"/match_keys.json")); EKset = set(keys)
names = json.load(open(E+"/loadout_names.json"))
# example rounds
for pref in ["9d26354b3", "ba1358958"]:
    mi = [j for j, k in enumerate(keys) if pref in k]
    print("match idx for", pref, mi, [keys[j] for j in mi])
    for j in mi:
        ridx = np.nonzero(np.asarray(RM) == j)[0]
        for i in ridx.tolist():
            print("  round_idx", i, "round_num", int(RN[i]), "loadout CT/T", float(L[i, 0]), float(L[i, 22]), "grid k0 CT/T", int(S[O[i],0]), int(S[O[i],1]), "steps", int(O[i+1]-O[i]), "T_alive_grid_min_max", int(np.asarray(S[O[i]:O[i+1],1]).min()), int(np.asarray(S[O[i]:O[i+1],1]).max()), "T_loadout", float(L[i,22]))
cr = pl.read_parquet(F+"/clean_rounds.parquet", columns=["match_key","round","use","src"])
print("clean_rounds rows", cr.height, "use", int(cr["use"].sum()), "use by src", cr.filter(pl.col("use")).group_by("src").len().rows())
ex_round_idx = np.arange(len(RM))
ex = pl.DataFrame({"match_key": [keys[j] for j in np.asarray(RM).tolist()], "round": np.asarray(RN).astype(np.int64), "steps": (O[1:]-O[:-1]).astype(np.int64), "ridx": ex_round_idx.astype(np.int64)})
print("export rounds", ex.height, "steps sum", int(ex["steps"].sum()), "unique (match,round)", ex.select(["match_key","round"]).unique().height)
m = ex.join(cr.select(["match_key","round","use","src"]), on=["match_key","round"], how="left")
print("export rounds matched to clean_rounds", int(m.filter(pl.col("use").is_not_null()).height), "use True", int(m.filter(pl.col("use")==True).height), "use False", int(m.filter(pl.col("use")==False).height), "no row", int(m.filter(pl.col("use").is_null()).height))
print("export round src", m.group_by("src").len().rows())
cg = pl.scan_parquet(F+"/clean_grid.parquet").group_by(["match_key","round"]).agg(pl.len().alias("grid_n")).collect()
print("clean_grid (match,round) groups", cg.height, "time", round(time.time()-t0,1))
print("grid total rows", int(cg["grid_n"].sum()))
j = ex.join(cg, on=["match_key","round"], how="left")
print("export rounds with grid group", int(j.filter(pl.col("grid_n").is_not_null()).height), "missing", int(j.filter(pl.col("grid_n").is_null()).height))
jj = j.filter(pl.col("grid_n").is_not_null())
diff = jj.filter(pl.col("grid_n") != pl.col("steps"))
print("export rounds where grid_n != steps", diff.height, "sum(grid_n) on export rounds", int(jj["grid_n"].sum()), "sum(steps)", int(jj["steps"].sum()))
print("sample diffs", diff.head(5).rows())
print("grid rows for export (match,round) minus export steps", int(jj["grid_n"].sum()) - int(jj["steps"].sum()))
# grid rows on non-export matches or rounds
ekm = pl.DataFrame({"match_key": keys})
cg_ek = cg.join(ekm, on="match_key", how="inner")
print("grid rows in export matches", int(cg_ek["grid_n"].sum()), "export matches' rounds not exported (grid rows)", int(cg_ek["grid_n"].sum()) - int(jj["grid_n"].sum()))
print("grid rows total", int(cg["grid_n"].sum()), "grid rows in non-export matches", int(cg["grid_n"].sum()) - int(cg_ek["grid_n"].sum()))
print("time", round(time.time()-t0,1))
