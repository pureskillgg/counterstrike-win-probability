"""S2 step 2: per-second round-state arrays of the holdout matches, one map at a time.

Drives build_round_state_map.py (imported). Three changes, all inside this process only:
  - OUT_DIR -> data/holdout_b/round_state (nothing is written under data/round_state)
  - --fixed-window: the window is the one stored in data/round_state/<map>_ext.json (the frozen extended arrays' window); a copy
    is placed at data/holdout_b/round_state/<map>.json because fixed_window() reads <OUT_DIR>/<map>.json
  - --day-dir -> data/holdout_b/positions_1hz (holdout positions only); --maps-from catalog (matches.parquet has the holdout keys)
"""
import json
import os
import sys
from pathlib import Path

ROOT = r"G:\pureskill\csds-cs2"
sys.path.insert(0, os.path.join(ROOT, "scripts", "model"))
import build_round_state_map as B  # noqa: E402

HB = Path(ROOT) / "data" / "holdout_b"
RS = HB / "round_state"
POS = HB / "positions_1hz"
RS.mkdir(parents=True, exist_ok=True)
B.OUT_DIR = RS
MAPS = ["de_mirage", "de_dust2", "de_ancient", "de_anubis", "de_inferno", "de_nuke", "de_cache"]
only = sys.argv[1:] or MAPS
for m in only:
    win = json.loads((Path(ROOT) / "data" / "round_state" / f"{m}_ext.json").read_text(encoding="utf-8"))["window"]
    (RS / f"{m}.json").write_text(json.dumps({"window": win, "copied_from": f"data/round_state/{m}_ext.json"}), encoding="utf-8")
    sys.argv = ["build_round_state_map.py", "--map", m, "--day-dir", str(POS), "--fixed-window", "--maps-from", "catalog",
                "--out-name", f"{m}_ext"]
    print(f"=== round state {m}", flush=True)
    B.main()
