"""M5: can batch (date) and source (FACEIT-hosted server) be separated in the map interaction?

  X-a. Joint model: round + map + batch + faceit + batch x map + faceit x map (core maps).
       Wald tests of each interaction block conditional on the other. Identifiability is limited:
       1,008 of 1,064 (kept) 10-08 matches are FACEIT-hosted; 24 FACEIT matches fall in 1001-1007.
  X-b. Round-bucket profile by batch: CT% by round number, 10-08 minus 1001-1007 (pp), with a
       cluster-robust Wald test of the round x batch interaction (all clean rounds, 25 buckets).
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\M5_separate.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(r"G:\pureskill\csds-cs2\scripts\swarm2")
_spec = importlib.util.spec_from_file_location("M5_map_batch", HERE / "M5_map_batch.py")
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)


def xa_joint(d):
    print("\n== X-a. Batch and FACEIT source inside the map interaction (core maps) ==")
    dc = d.filter(pl.col("map_pool") != "rare")
    print("Cells (matches): batch x faceit")
    print(dc.group_by(["batch", "faceit"]).agg(pl.col("match_key").n_unique().alias("matches"),
                                               pl.len().alias("rounds")).sort(["batch", "faceit"]))
    full = M.run_model(dc, ["round", "map", "batch", "faceit", "batch_x_map", "faceit_x_map"],
                       "X-a joint: batch + faceit + both interactions")
    for lab, key in [("batch x map | faceit x map", "batch_x_map"), ("faceit x map | batch x map", "faceit_x_map")]:
        W, df, p = M.wald(full["beta"], full["V"], full["idx"][key])
        print(f"  {lab:<32}: chi2={W:.2f} df={df} p={p:.4g}")
    # condition-free comparison: batch x map only vs faceit x map only already in M3 / M5
    b, se = full["beta"][full["idx"]["batch"][0]], np.sqrt(full["V"][full["idx"]["batch"][0], full["idx"]["batch"][0]])
    f_b, f_se = full["beta"][full["idx"]["faceit"][0]], np.sqrt(full["V"][full["idx"]["faceit"][0], full["idx"]["faceit"][0]])
    print(f"  batch main (given faceit and maps): b={b:+.3f} se={se:.3f}")
    print(f"  faceit main (given batch and maps): b={f_b:+.3f} se={f_se:.3f}")
    print("  Reading: with both terms the within-cell information is thin (56 non-FACEIT 10-08 matches,"
          " 24 FACEIT 1001-1007 matches), so the separate blocks have wide uncertainty.")


def xb_round_profile(d):
    print("\n== X-b. CT% by round bucket, 10-08 minus 1001-1007 (all clean rounds, both batches) ==")
    cells = (d.group_by(["rb", "batch"]).agg(pl.len().alias("n"), pl.col("ct_win").mean().alias("p"))
             .sort(["rb", "batch"]))
    a = cells.filter(pl.col("batch") == "b1001_1007").select(["rb", "n", "p"]).rename({"n": "n1001", "p": "p1001"})
    b = cells.filter(pl.col("batch") == "b1008").select(["rb", "n", "p"]).rename({"n": "n1008", "p": "p1008"})
    j = a.join(b, on="rb").with_columns(((pl.col("p1008") - pl.col("p1001")) * 100).round(2).alias("diff_pp"))
    print(j.sort("rb").with_columns(pl.col(["p1001", "p1008"]).round(4)))
    # cluster-robust Wald for round x batch interaction (rounds 1..24 vs OT pooled)
    base = M.design(d, ["round", "batch"])
    X0, names0, idx0 = base
    rb = d["rb"].to_numpy()
    b1008 = d["b1008"].to_numpy().astype(float)
    cols = []
    for lvl in range(2, 26):
        cols.append(b1008 * (rb == lvl))
    X = np.column_stack([X0] + cols)
    y = d["ct_win"].to_numpy().astype(float)
    cl = np.unique(d["match_key"].to_numpy(), return_inverse=True)[1].astype(np.int64)
    beta, V, ll, G = M.fit_logit(X, y, cl)
    k0 = X0.shape[1]
    idx = list(range(k0, X.shape[1]))
    W = float(beta[idx] @ np.linalg.solve(V[np.ix_(idx, idx)], beta[idx]))
    print(f"round x batch interaction (cluster-robust Wald, df={len(idx)}): chi2={W:.2f} p={stats.chi2.sf(W, len(idx)):.4g}")


def main():
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_fmt_str_lengths(30)
    d = M.load()
    xa_joint(d)
    xb_round_profile(d)


if __name__ == "__main__":
    main()
