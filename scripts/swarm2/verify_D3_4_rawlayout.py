"""D3-4 verifier: shallow listing of one raw E: match folder to learn the layout (READ-ONLY, listdir only)."""
import os, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
RAW = Path(r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02")
uuid = "m_06f6a3ca"
day_dirs = sorted(os.listdir(RAW))[:3]
print("day dirs (first 3):", day_dirs)
d = RAW / "12"
print("entries under day 12 (count):", len(os.listdir(d)))
match_dirs = [x for x in os.listdir(d) if x.startswith(uuid)]
print("matching dirs for prefix:", match_dirs)
for mdn in match_dirs:
    md = d / mdn
    print("match dir listing:", sorted(os.listdir(md))[:40])
    for sub in sorted(os.listdir(md))[:3]:
        p = md / sub
        if p.is_dir():
            print(f"  {sub}/ ->", sorted(os.listdir(p))[:40])
