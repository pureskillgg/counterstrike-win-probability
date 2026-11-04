"""Pin 79da6d09 comparison, computed as pinned: gru_L60 minus gru_L20, EMA final, seeds 0 to 2, paired per second on validation.
Validation predictions only. paired() from scripts/model/spatial_feature_eval.py, 1,000 match-cluster resamples, RNG seed 0."""
import sys
import numpy as np
import polars as pl

sys.path.insert(0, "G:/pureskill/csds-cs2/scripts/model")
from spatial_feature_eval import paired  # noqa: E402

D = "G:/pureskill/csds-cs2/data/diagnostics/window_gru_final"
KEY = ["match_key", "round", "second"]

def load(seed, L):
    df = pl.read_parquet(f"{D}/gru_L{L}_lr0.001_seed{seed}_ema_final_val_preds.parquet")
    lab = "label" if "label" in df.columns else "label_ct_win"
    pc = "p" if "p" in df.columns else [c for c in df.columns if c.startswith("p")][0]
    return df.select(KEY + [lab, pc]).rename({lab: "y", pc: "p"})

rows = []
for s in (0, 1, 2):
    a = load(s, 60)
    b = load(s, 20)
    j = a.join(b, on=KEY, how="inner", suffix="_20")
    assert j.height == a.height == b.height, f"seed {s}: row sets differ"
    assert (j["y"] == j["y_20"]).all(), f"seed {s}: labels differ"
    r = paired(j["match_key"].to_numpy(), j["y"].to_numpy().astype(np.float64),
               j["p"].to_numpy().astype(np.float64), j["p_20"].to_numpy().astype(np.float64), seed=0)
    rows.append((s, float(r["diff"]), [float(v) for v in r["ci95"]]))
    print(f"seed {s}: L60 minus L20 {r['diff']:+.6f}, 95% [{r['ci95'][0]:+.6f}, {r['ci95'][1]:+.6f}]", flush=True)

pooled = float(np.mean([d for _, d, _ in rows]))
all_upper = all(c[1] < 0 for _, _, c in rows)
all_lower = all(c[0] > 0 for _, _, c in rows)
if pooled <= -0.0005 and all_upper:
    v = "60-second window beats 20-second"
elif pooled >= 0.0005 and all_lower:
    v = "20-second window beats 60-second"
else:
    v = "no claim"
print(f"POOLED {pooled:+.6f}; {v}")
