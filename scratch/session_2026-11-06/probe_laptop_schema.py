import polars as pl
import glob, os
B = "/home/sagemaker-user/adam/builder/"
for d in ["laptop_status/status_1hz", "laptop_pos/every2_sites_v1"]:
    fs = sorted(glob.glob(B + d + "/*.parquet"))
    print(d, "files", len(fs), fs[0].split("/")[-1], fs[-1].split("/")[-1])
    f = pl.read_parquet(fs[0])
    print("  rows", f.height, "cols", f.columns)
    print("  dtypes", dict(zip(f.columns, [str(t) for t in f.dtypes])))
    print(f.head(3))
lap_dirs = ["/home/sagemaker-user/adam/builder/final/export_clean_v4/../../final/laptop", "/home/sagemaker-user/adam/builder/data/laptop"]
m = pl.read_parquet(B + "data/laptop/matches.parquet")
print("data/laptop/matches cols", m.columns, m.height)
r = pl.read_parquet(B + "data/laptop/rounds.parquet")
print("data/laptop/rounds cols", r.columns, r.height)
print("laptop_status cols earlier check", "keys?")
print("glob other laptop-ish dirs:", sorted(os.listdir(B + "final"))[:80])
