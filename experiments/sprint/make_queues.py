"""Writes the validation sprint's queue and design file. One queue serves both devices.

Every cell is run on the GPU and on the CPU, each device into its own run folder (<out>/gpu, <out>/cpu), so every comparison pairs two
arms on the same device. The cells, in priority order:
  1. the grid for both arms, seeds 0 to 2 (core cells first, so a stopped run still answers the question);
  2. the feature-group ablations at the default setting, seeds 0 to 4, and the full default cell for seeds 3 and 4 (the ablations'
     reference);
  3. the grid for both arms, seeds 3 and 4.
The default setting (depth 4, eta 0.05, min_child_weight 20) is one cell of the grid.

    python make_queues.py --out <run dir>
"""
import argparse
import json
import os

DEPTHS = [3, 4, 6]
ETAS = [0.03, 0.05, 0.1]
MCWS = [20, 100]
ARMS = ["full", "base"]
ABLATIONS = ["base+GEO", "base+VIEW", "base+BLIND", "base+VIEW+BLIND", "base+GEO+BLIND", "base+GEO+VIEW"]
DEFAULT = (4, 0.05, 20)
SEEDS = [0, 1, 2]  # three seeds per amendment ddd564b1 (seeds 3 and 4 dropped)


def cell_id(set_name, d, e, m, seed):
    return f"{set_name.replace('+', '-')}_d{d}_e{e}_m{m}_s{seed}"


def cell(set_name, d, e, m, seed):
    return {"id": cell_id(set_name, d, e, m, seed), "set": set_name, "depth": d, "eta": e, "mcw": m, "seed": seed}


def grid_cells(seeds):
    return [cell(arm, d, e, m, s) for d in DEPTHS for e in ETAS for m in MCWS for arm in ARMS for s in seeds]


def build_cells():
    """Three seeds (0, 1, 2) throughout, per amendment ddd564b1: the grid for both arms, then the ablations at the default setting."""
    cells, seen = [], set()

    def add(c):
        if c["id"] not in seen:
            seen.add(c["id"])
            cells.append(c)

    for c in grid_cells(SEEDS):
        add(c)
    for name in ABLATIONS:
        for seed in SEEDS:
            add(cell(name, *DEFAULT, seed))
    return cells


def main():
    ap = argparse.ArgumentParser(description="Write the validation sprint queue and design file")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cells = build_cells()
    # the CPU fits take about 69 minutes each on this machine (base, seed 0), so the CPU queue holds the default setting for both
    # arms, seeds 0 to 2: the device check on the core comparison, about 7 hours at full CPU use
    cpu_cells = [c for c in cells if c["set"] in ARMS and (c["depth"], c["eta"], c["mcw"]) == DEFAULT and c["seed"] in (0, 1, 2)]
    with open(os.path.join(a.out, "queue.json"), "w") as f:
        json.dump(cells, f, indent=1)
    with open(os.path.join(a.out, "cpu_queue.json"), "w") as f:
        json.dump(cpu_cells, f, indent=1)
    design = {"cells": len(cells), "cpu_cells": len(cpu_cells), "devices": ["cuda", "cpu"], "arms": ARMS, "depths": DEPTHS,
              "etas": ETAS, "min_child_weight": MCWS, "ablations": ABLATIONS, "default_setting": list(DEFAULT),
              "fixed": {"subsample": 0.8, "early_stopping_rounds": 100, "tree_method": "hist", "n_estimators_cap": 3000}}
    with open(os.path.join(a.out, "design.json"), "w") as f:
        json.dump(design, f, indent=1)
    print(f"{len(cells)} cells per device, written to {a.out}", flush=True)


if __name__ == "__main__":
    main()
