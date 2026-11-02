"""Screening rule for the learning-rate sweep (pre-registered in the Meridian pin for the sweep). Read-only apart from its own output.

For each arm, compare the EMA final-step validation log loss at learning rate 1e-3 and 3e-4 on the screening seed (0).
Keep 1e-3 unless 3e-4 is lower by more than 0.0003. Prints the arm:lr pairs for the confirmation run and writes selection.json
next to the screening outputs.

  ...tabicl python.exe -I -X utf8 scripts/model/select_lr.py data/diagnostics/lr_sweep_screen
"""
import json
import math
import sys
from pathlib import Path

ARMS = ("gru", "tcn", "tf")
LRS = (1e-3, 3e-4)
MARGIN = 0.0003

screen = Path(sys.argv[1])
STEPS, WARMUP, EMA = 30000, 500, 0.999  # the screen's settings; a metrics file with other settings is refused
stale = screen / "selection.json"
if stale.exists():
    stale.unlink()  # a failed run must not leave an earlier selection behind
chosen = {}
record = {}
for arm in ARMS:
    vals = {}
    for lr in LRS:
        path = screen / f"{arm}_lr{lr:g}_seed0_metrics.json"
        m = json.loads(path.read_text(encoding="utf-8"))
        found = (m.get("arm"), m.get("seed"), m.get("steps"), m.get("warmup"), m.get("ema_decay"))
        if found != (arm, 0, STEPS, WARMUP, EMA) or abs(m.get("lr", -1) - lr) > 1e-12:
            raise SystemExit(f"{path.name} does not match this screen: arm, seed, steps, warmup, EMA, lr = {found}, {m.get('lr')}")
        value = m["final_ema_val_log_loss"]
        if not math.isfinite(value):
            raise SystemExit(f"{path.name} has a non-finite final EMA log loss ({value}); the rule does not pick a diverged run")
        vals[lr] = value
    pick = 3e-4 if vals[3e-4] < vals[1e-3] - MARGIN else 1e-3
    chosen[arm] = pick
    record[arm] = {"final_ema_val_log_loss": {f"{lr:g}": vals[lr] for lr in LRS}, "chosen_lr": pick}

(screen / "selection.json").write_text(json.dumps({"rule": f"keep 1e-3 unless 3e-4 is lower by more than {MARGIN}",
                                                   "arms": record}, indent=1), encoding="utf-8")
print(",".join(f"{arm}:{chosen[arm]:g}" for arm in ARMS))
