"""Regularisation test vs the GRU log-loss reference (validation, final-step, five seeds).

Arms:
  A = 30k steps, dropout 0.2, weight decay 1e-4
  B = 25k steps, no regularisation
Reference = GRU log-loss runs at 30k steps (same seeds). Rule from pin c1f31a9d:
  a setting is a lever if its five-seed mean is >= 0.0005 below the reference mean
  AND it beats the reference on >= 4 of 5 seeds.
Paired by seed. Interval = match-cluster bootstrap is not available from metrics alone,
so this reports seed-level differences only (no bootstrap).
"""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).parent / "reg_results"
SEEDS = [0, 1, 2, 3, 4]


def final(path):
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    return m["final_val_log_loss"], m["steps"], m.get("dropout"), m.get("weight_decay"), m.get("objective"), m.get("best_step"), m.get("best_val_log_loss")


ref = {}
for s in SEEDS:
    v, steps, *_ = final(ROOT / "ref" / f"gru_seed{s}_metrics.json")
    ref[s] = v
    print(f"ref seed{s}: final {v:.6f} steps {steps}")

ref_mean = statistics.mean(ref.values())
print(f"\nreference five-seed mean (final step): {ref_mean:.6f}  [pinned reference 0.451061]")

summary = {}
for arm, label in [("A", "A: 30k, dropout 0.2, wd 1e-4"), ("B", "B: 25k, no reg")]:
    rows = []
    for s in SEEDS:
        v, steps, dropout, wd, obj, best_step, best = final(ROOT / f"reg_{arm}" / f"gru_seed{s}_metrics.json")
        rows.append((s, v, steps, dropout, wd, obj, best_step, best))
    vals = [r[1] for r in rows]
    mean = statistics.mean(vals)
    diffs = [ref[r[0]] - r[1] for r in rows]  # positive = arm better
    wins = sum(1 for d in diffs if d > 0)
    print(f"\n== {label}")
    print(f"   objective {rows[0][5]}, steps {rows[0][2]}, dropout {rows[0][3]}, wd {rows[0][4]}")
    for r, d in zip(rows, diffs):
        print(f"   seed{r[0]}: final {r[1]:.6f}  best {r[7]:.6f} @ {r[6]}  ref {ref[r[0]]:.6f}  gain {d:+.6f}")
    print(f"   five-seed mean {mean:.6f}  gain vs ref mean {ref_mean - mean:+.6f}  wins {wins}/5")
    lever = (ref_mean - mean) >= 0.0005 and wins >= 4
    print(f"   LEVER under rule (>=0.0005 and >=4/5): {lever}")
    summary[arm] = {"mean": mean, "gain": ref_mean - mean, "wins": wins, "lever": lever,
                    "seeds": {r[0]: {"final": r[1], "best": r[7], "best_step": r[6], "ref": ref[r[0]]} for r in rows}}

(ROOT / "reg_compare_summary.json").write_text(json.dumps({"ref_mean": ref_mean, "ref": ref, **summary}, indent=2), encoding="utf-8")
print("\nwrote", ROOT / "reg_compare_summary.json")
