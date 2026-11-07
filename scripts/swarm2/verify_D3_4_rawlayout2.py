"""D3-4 verifier: list round_end / round_state / header folders under one raw E: match (READ-ONLY, listdir only)."""
import os, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
md = Path(r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02\12\0e458e20-b8d7-4af6-b6e8-5df25d2a5fee")
for sub in ["round_end", "round_state", "header", "csds"]:
    p = md / sub
    if p.exists():
        names = sorted(os.listdir(p))
        print(f"{sub}/ count={len(names)} first={names[:8]}")
    else:
        print(f"{sub}/ missing")
