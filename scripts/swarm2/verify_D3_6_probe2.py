import os, sys, time, glob
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
E = Path(r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02")
days = sorted(os.listdir(E))
print("days", days)
d1 = E / "01"
uu = sorted(os.listdir(d1))
print("day01 folders", len(uu))
empties = [u for u in uu if not os.listdir(d1 / u)]
print("day01 empty", len(empties))
nonempty = [u for u in uu if os.listdir(d1 / u)]
print("day01 nonempty", len(nonempty))
for u in nonempty[:5]:
    print("  nonempty sample", u, sorted(os.listdir(d1 / u))[:40])
t = time.time()
cnt = 0
for u in sorted(os.listdir(E / "02"))[:200]:
    p = E / "02" / u / "round_end"
    if os.path.exists(p):
        pl.read_parquet(str(p), columns=["round"])
        cnt += 1
print("200 raw round_end reads (day02) ok:", cnt, "secs", round(time.time() - t, 2))
