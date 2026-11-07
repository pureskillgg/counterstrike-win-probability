import json, numpy as np, collections, glob, os, polars as pl, pyarrow.parquet as pq
E = "/home/sagemaker-user/adam/builder/final/export_clean_v2"; F = "/home/sagemaker-user/adam/builder/final"; B = "/home/sagemaker-user/adam/builder"
keys = json.load(open(E+"/match_keys.json"))
def ym(k): return "/".join(k.split("/")[1:3])
print("export key months", sorted(collections.Counter(ym(k) for k in keys).items()))
win = [k for k in keys if k.split("/")[1:4] in (["2026","10","01"],["2026","10","02"],["2026","10","03"],["2026","10","04"],["2026","10","05"],["2026","10","06"],["2026","10","07"],["2026","10","08"])]
print("export keys in 2026/10/01-08 window", len(win))
print("parity dir files", sorted(os.listdir(F+"/pos_parity")) if os.path.isdir(F+"/pos_parity") else "NA")
for f in sorted(glob.glob(F+"/pos_parity/*.parquet")):
    md = pq.ParquetFile(f).metadata
    cols = pq.ParquetFile(f).schema_arrow.names
    print("parity", os.path.basename(f), "rows", md.num_rows, "cols", len(cols), cols[:8])
lp = glob.glob("/home/sagemaker-user/adam/laptop_pos/every2_sites_v1/*.parquet") + glob.glob(B+"/../laptop_pos/every2_sites_v1/*.parquet")
print("laptop_pos day files", len(set(lp)))
print("pos_parts row total (metadata)", sum(pq.ParquetFile(f).metadata.num_rows for f in glob.glob(F+"/pos_parts/bucket_c*.parquet")), "files", len(glob.glob(F+"/pos_parts/bucket_c*.parquet")))
print("pos_check_all row total (metadata)", sum(pq.ParquetFile(f).metadata.num_rows for f in glob.glob(F+"/pos_check_all/bucket_c*.parquet")), "files", len(glob.glob(F+"/pos_check_all/bucket_c*.parquet")))
O = np.load(E+"/offsets.npy"); PO = np.load(E+"/pos_offsets.npy")
RM = np.load(E+"/round_match.npy")
cm = pl.read_parquet(F+"/clean_matches.parquet", columns=["match_key","src"])
src = dict(zip(cm["match_key"].to_list(), cm["src"].to_list()))
rsrc = np.array([src.get(keys[m], "NA") for m in RM.tolist()])
n = (O[1:]-O[:-1]).astype(np.int64)
ceil = (n+1)//2
print("expected even steps (sum ceil(n/2))", int(ceil.sum()), "E=PO[-1]", int(PO[-1]))
bmask = rsrc == "bucket"
print("expected bucket positions (bucket-src rounds)", int(ceil[bmask].sum()), "laptop-src", int(ceil[rsrc=="laptop"].sum()))
pi = np.load(E+"/pos_int8.npy", mmap_mode="r")
cnt = (PO[1:]-PO[:-1]).astype(np.int64)
rid = np.repeat(np.arange(len(cnt)), cnt)
c0 = np.asarray(pi[:,0])
neg = (c0 == -1)
print("c_site_hi == -1 total", int(neg.sum()), "in bucket-src rounds", int((neg & bmask[rid]).sum()), "in laptop-src rounds", int((neg & (rsrc[rid]=="laptop")).sum()))
