"""D3-4 verifier: file sizes and magic bytes of raw E: channel files for one match (READ-ONLY)."""
import os, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
md = Path(r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02\12\0e458e20-b8d7-4af6-b6e8-5df25d2a5fee")
for name in ["round_end", "round_state", "header", "round_start", "bomb_defuse"]:
    p = md / name
    if p.is_file():
        with open(p, "rb") as f:
            magic = f.read(4)
        print(f"{name}: file size={p.stat().st_size} magic={magic!r}")
    else:
        print(f"{name}: not a file (is_dir={p.is_dir()})")
csds = md / "csds"
print("csds entry is_dir:", csds.is_dir(), "is_file:", csds.is_file())
if csds.is_file():
    print("csds size:", csds.stat().st_size)
    with open(csds, "rb") as f: print("csds magic:", f.read(4))
