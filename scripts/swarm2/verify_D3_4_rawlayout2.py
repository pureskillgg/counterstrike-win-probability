"""D3-4 verifier: list round_end / round_state / header folders under one raw E: match (READ-ONLY, listdir only)."""
import os, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
md = Path(r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02\12\m_06f6a3ca51dfd6ba3de2d2ae")
for sub in ["round_end", "round_state", "header", "csds"]:
    p = md / sub
    if p.exists():
        names = sorted(os.listdir(p))
        print(f"{sub}/ count={len(names)} first={names[:8]}")
    else:
        print(f"{sub}/ missing")
