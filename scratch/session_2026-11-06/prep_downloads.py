import csv, random
import polars as pl
OUT = "/home/sagemaker-user/adam/builder/fixes/fix332/"
rec = [r for r in csv.DictReader(open(OUT + "recover_list.csv")) if r["s3_folder_present"] == "True"]
# control: tome-covered laptop matches (2026/10) that also have an S3 per-match folder
rf = pl.read_parquet(OUT + "round_flags.parquet", columns=["match_key", "src", "has_roster"])
cov = (rf.filter((pl.col("src") == "laptop") & pl.col("has_roster"))
       .select("match_key").unique())
cov_keys = set(cov["match_key"].to_list())
s3f = set()
for line in open(OUT + "s3_folders_2026_10.txt"):
    dd, uu = line.split()
    s3f.add(("2026/10/" + dd, uu.rstrip("/")))
ctrl_pool = []
for mk in sorted(cov_keys):
    p = mk.split("/")
    if p[1] == "2026" and p[2] == "10" and (p[1] + "/" + p[2] + "/" + p[3], p[4]) in s3f:
        ctrl_pool.append(mk)
random.seed(332)
ctrl = set(random.sample(ctrl_pool, min(150, len(ctrl_pool))))
print("control pool", len(ctrl_pool), "control sample", len(ctrl))
lines = []
for r in rec:
    lines.append((r["match_key"], "recover"))
for mk in sorted(ctrl):
    lines.append((mk, "control"))
with open(OUT + "dl_list.tsv", "w") as fh:
    for mk, tag in lines:
        p = mk.split("/")
        folder = "/".join(p[:5])  # csds/YYYY/MM/DD/uuid
        fh.write(f"{folder}\t{'csds_' + '_'.join(p[1:4]) + '_' + p[4]}\t{tag}\n")
print("download matches", len(lines), "recover", len(rec), "control", len(ctrl))
