"""Check whether isotonic non-monotonicity comes from float32 inputs. Synthetic data."""
import numpy as np
from sklearn.isotonic import IsotonicRegression

r = np.random.default_rng(0)
n = 4000
s64 = np.sort(r.uniform(0.0, 1.0, n))            # float64 raw scores
y = (r.uniform(size=n) < 1 / (1 + np.exp(-(4 * s64 - 2)))).astype(float)
grid = np.linspace(0.0, 1.0, 20001)

for label, dtype in (("float64 scores", np.float64), ("float32 scores", np.float32)):
    iso = IsotonicRegression(out_of_bounds="clip").fit(s64.astype(dtype), y)
    out = iso.predict(grid)
    d = np.diff(out)
    print(f"{label:<16} dtype(X_thr)={iso.X_thresholds_.dtype} dtype(y_thr)={iso.y_thresholds_.dtype} "
          f"steps_down={int(np.sum(d < -1e-12))} max_drop={float(max(0.0, -d.min())):.3e}", flush=True)
