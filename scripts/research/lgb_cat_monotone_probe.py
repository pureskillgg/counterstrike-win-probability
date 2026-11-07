"""Probe LightGBM monotone_constraints with a categorical column (synthetic, n=3000).

Usage: python -I lgb_cat_monotone_probe.py <variant>
  variant "x0"  -> constraints [1, 0, 0] (x0 = +1, categorical = 0)
  variant "cat" -> constraints [0, 0, 1] (constraint placed on the categorical)
"""
import os
import sys

os.environ["OMP_NUM_THREADS"] = "2"

import numpy as np
import pandas as pd
import lightgbm as lgb

variant = sys.argv[1]
levels = ["a", "b", "c", "d", "e"]
r = np.random.default_rng(0)
n = 3000
x0 = r.uniform(-1, 1, n)
x1 = r.uniform(-1, 1, n)
cat = r.choice(levels, size=n)
eff = {"a": -1.5, "b": -0.5, "c": 0.0, "d": 0.5, "e": 1.5}
logit = x0 + 0.5 * x1 + np.array([eff[v] for v in cat])
y = (r.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
df = pd.DataFrame({"x0": x0, "x1": x1, "cat": pd.Categorical(cat, categories=levels)})
cons = {"x0": [1, 0, 0], "cat": [0, 0, 1]}[variant]
print(f"variant={variant} constraints={cons} start", flush=True)
m = lgb.LGBMClassifier(n_estimators=50, num_leaves=15, monotone_constraints=cons,
                       n_jobs=2, random_state=0, verbose=-1)
m.fit(df, y)
print(f"variant={variant} fit returned OK", flush=True)
print("monotone in model:", m.booster_.dump_model().get("monotone_constraints"), flush=True)
