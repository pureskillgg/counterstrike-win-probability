"""Tables for the drop-group ablations (scripts/explain/xgb83_drop_ablation.py).

For each set: validation log loss per seed, the paired difference from the full 83-input model with the same seed (positive
means the set is worse without the inputs), the mean and spread over seeds, and the mean stopping iteration. Also checks that
the full model's seed-0 result matches the sprint's saved result for the same seed, as a check on the fit path.

Run with the pixi environment after the ablation run has finished:
  G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe scripts\\explain\\xgb83_ablation_report.py
"""
import json
import os
import statistics as st
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xgb83_explain import DOCS  # noqa: E402

RUN = "C:/Users/13144/sprint/explain83_ablation"
TABLES = os.path.join(DOCS, "tables")
SEEDS = [0, 1, 2]


def main():
    rows = []
    for f in sorted(os.listdir(os.path.join(RUN, "results"))):
        if f.endswith(".json"):
            rows.append(json.load(open(os.path.join(RUN, "results", f))))
    df = pl.DataFrame(rows)
    assert not df.is_empty(), "no results yet"
    sets = list(json.load(open(os.path.join(RUN, "sets.json"))).keys())
    full = {r["seed"]: r["val_logloss"] for r in rows if r["set"] == "full"}
    assert all(s in full for s in SEEDS), f"full model missing seeds: {sorted(set(SEEDS) - set(full))}"

    # reproduction check: this environment's full model against the sprint's saved result, same seed
    repro = [{"seed": r["seed"], "this_run": r["val_logloss"], "sprint_saved": r["sprint_val_logloss_same_seed"],
              "difference": r["val_logloss"] - r["sprint_val_logloss_same_seed"]}
             for r in rows if r["set"] == "full" and "sprint_val_logloss_same_seed" in r]
    pl.DataFrame(repro).write_csv(os.path.join(TABLES, "drop_ablation_reproduction.csv"))

    per_seed, summary = [], []
    for name in sets:
        rs = {r["seed"]: r for r in rows if r["set"] == name and r["seed"] in SEEDS}
        if len(rs) < len(SEEDS):
            continue
        lls = [rs[s]["val_logloss"] for s in SEEDS]
        deltas = [rs[s]["val_logloss"] - full[s] for s in SEEDS]
        for s in SEEDS:
            per_seed.append({"set": name, "seed": s, "n_inputs": rs[s]["n_inputs"], "val_logloss": rs[s]["val_logloss"],
                             "delta_vs_full_same_seed": rs[s]["val_logloss"] - full[s],
                             "best_iteration": rs[s]["best_iteration"]})
        summary.append({"set": name, "n_inputs": rs[SEEDS[0]]["n_inputs"], "mean_val_logloss": st.mean(lls),
                        "mean_delta_vs_full": st.mean(deltas), "sd_delta_over_seeds": st.stdev(deltas) if len(deltas) > 1 else None,
                        "min_delta": min(deltas), "max_delta": max(deltas),
                        "mean_best_iteration": st.mean(rs[s]["best_iteration"] for s in SEEDS),
                        "xgboost": rs[SEEDS[0]]["xgboost"]})
    s_df = pl.DataFrame(summary).sort("mean_delta_vs_full", descending=True)
    s_df.write_csv(os.path.join(TABLES, "drop_ablation_83.csv"))
    pl.DataFrame(per_seed).write_csv(os.path.join(TABLES, "drop_ablation_83_per_seed.csv"))
    print(s_df.to_pandas().to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print("reproduction (full, same seed):")
    print(pl.DataFrame(repro).to_pandas().to_string(index=False, float_format=lambda v: f"{v:.6f}") if repro else "none")


if __name__ == "__main__":
    main()
