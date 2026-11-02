"""S2 step 3: per-slot loadout features of the holdout matches, one map at a time (build_loadout_map.py, imported).

Redirects, in this process only: round-state input and output -> data/holdout_b/round_state; status -> data/holdout_b/status_1hz;
positions -> data/holdout_b/positions_1hz. Writes data/holdout_b/round_state/<map>_ext_loadout.npy and .json.
"""
import os
import sys
from pathlib import Path

ROOT = r"G:\pureskill\csds-cs2"
sys.path.insert(0, os.path.join(ROOT, "scripts", "model"))
import build_loadout_map as BL  # noqa: E402
import embed_round_state as ers  # noqa: E402

HB = Path(ROOT) / "data" / "holdout_b"
BL.OUT = HB / "round_state"
BL.STATUS_DIR = HB / "status_1hz"
ers.POSITIONS_DIR = HB / "positions_1hz"
MAPS = ["de_mirage", "de_dust2", "de_ancient", "de_anubis", "de_inferno", "de_nuke", "de_cache"]
for m in sys.argv[1:] or MAPS:
    sys.argv = ["build_loadout_map.py", "--map", m, "--in-stem", f"{m}_ext", "--out-stem", f"{m}_ext", "--maps-from", "catalog"]
    print(f"=== loadout {m}", flush=True)
    BL.main()
