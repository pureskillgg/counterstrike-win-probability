"""M2 scale check: how big is the map term next to the alive-difference and score-lead terms.

Fits, on the kept states of each era (in-sample, same as the M2 analysis):
  constant, alive difference only, alive + score lead, alive + score lead + map.
Reports log loss per state and the gain of each term. Cluster-robust Wald for the map block.

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\M2_scale.py
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.special import expit
from scipy.stats import chi2 as chi2_dist

sys.stdout.reconfigure(encoding="utf-8")

SCRATCH = Path("C:/Users/<user>/AppData/Local/Temp/claude/C--Users-<user>-Documents-pureskill/"
               "e0a18ad2-aa9e-4a3c-9946-29641b2cde82/scratchpad")
REF_MAP = "de_mirage"
MIN_MATCHES = 20


def fit_logit(X, y, clusters):
    n, k = X.shape
    beta = np.zeros(k)
    p0 = float(y.mean())
    beta[0] = np.log(p0 / (1 - p0))
    for _ in range(100):
        p = expit(X @ beta)
        w = p * (1 - p)
        g = X.T @ (y - p)
        H = (X * w[:, None]).T @ X + 1e-9 * np.eye(k)
        step = np.linalg.solve(H, g)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = expit(X @ beta)
    w = p * (1 - p)
    H = (X * w[:, None]).T @ X + 1e-9 * np.eye(k)
    Hinv = np.linalg.inv(H)
    eps = 1e-15
    ll = float(np.sum(y * np.log(np.clip(p, eps, 1)) + (1 - y) * np.log(np.clip(1 - p, eps, 1))))
    s = X * (y - p)[:, None]
    _, inv = np.unique(clusters, return_inverse=True)
    C = int(inv.max()) + 1
    S = np.column_stack([np.bincount(inv, weights=s[:, j], minlength=C) for j in range(k)])
    V = Hinv @ (S.T @ S) @ Hinv * (C / (C - 1)) * ((n - 1) / (n - k))
    return beta, V, ll


def levels_for(rounds_kept):
    cnt = rounds_kept.group_by("map_name").agg(pl.col("match_key").n_unique().alias("m"))
    d = {r["map_name"]: r["m"] for r in cnt.iter_rows(named=True)}
    lv = sorted([m for m, n in d.items() if n >= MIN_MATCHES and m != REF_MAP])
    levels = [REF_MAP] + lv + (["other"] if any(n < MIN_MATCHES for n in d.values()) else [])
    return levels


def run(era):
    r = pl.read_parquet(SCRATCH / f"M2_rounds_{era}.parquet").filter(pl.col("kept"))
    s = pl.read_parquet(SCRATCH / f"M2_states_{era}.parquet").sort(["match_key", "round", "tick"])
    levels = levels_for(r)
    mlv = s["map_name"].to_numpy()
    mlv = np.where(np.isin(mlv, levels), mlv, "other")
    y = s["label_ct_win"].to_numpy().astype(np.float64)
    cl = s["match_key"].to_numpy()
    alive = s["ct_alive_minus_t"].to_numpy().astype(np.float64)
    lead = s["ct_score_minus_t_score"].to_numpy().astype(np.float64)
    n = len(y)
    one = np.ones(n)
    X_const = one[:, None]
    X_alive = np.column_stack([one, alive])
    X_al = np.column_stack([one, alive, lead])
    map_cols = [(mlv == lv).astype(np.float64) for lv in levels if lv != REF_MAP]
    X_map = np.column_stack([X_al] + map_cols)
    out = {}
    fits = {}
    for name, X in [("constant", X_const), ("alive", X_alive), ("alive+lead", X_al), ("alive+lead+map", X_map)]:
        beta, V, ll = fit_logit(X, y, cl)
        fits[name] = (beta, V, ll, X.shape[1])
        out[name] = {"ll_per_state": ll / n, "k": X.shape[1], "ll": ll}
    base = fits["constant"][2]
    print(f"[{era}] states {n}, matches {len(np.unique(cl))}, levels {levels}")
    prev = None
    for name in ["constant", "alive", "alive+lead", "alive+lead+map"]:
        ll = fits[name][2] / n
        if prev is None:
            print(f"  {name:<16} LL/state {ll:.6f}")
        else:
            print(f"  {name:<16} LL/state {ll:.6f}  gain over {prev[0]:<12} {ll - prev[1]:+.6f}")
        prev = (name, ll)
    gain_alive = (fits["alive"][2] - fits["constant"][2]) / n
    gain_lead = (fits["alive+lead"][2] - fits["alive"][2]) / n
    gain_map = (fits["alive+lead+map"][2] - fits["alive+lead"][2]) / n
    print(f"  gains per state: alive {gain_alive:+.6f}; score lead {gain_lead:+.6f}; map {gain_map:+.6f}; "
          f"map as share of alive gain {100 * gain_map / gain_alive:.2f}%, of lead gain {100 * gain_map / gain_lead:.2f}%")
    beta, V, ll, k = fits["alive+lead+map"]
    k_al = 3
    idx = list(range(k_al, k))
    b = beta[idx]
    Vb = V[np.ix_(idx, idx)]
    W = float(b @ np.linalg.solve(Vb, b))
    p = float(chi2_dist.sf(W, len(idx)))
    print(f"  cluster-robust Wald for map block: chi2 {W:.2f}, df {len(idx)}, p {p:.4g}")
    out["gains"] = {"alive": gain_alive, "lead": gain_lead, "map": gain_map,
                    "wald_chi2": W, "wald_df": len(idx), "wald_p": p}
    return out


def main():
    res = {}
    for era in ["2026", "2024"]:
        res[era] = run(era)
    import json
    (SCRATCH / "M2_scale.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(f"wrote {SCRATCH / 'M2_scale.json'}")


if __name__ == "__main__":
    main()
