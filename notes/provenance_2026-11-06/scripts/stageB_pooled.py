"""Stage B of pin c807884d: GDT difference vs the GRU control, pooled over seeds 0 to 4.

Uses the grader's own functions (score_gdt.py) for the datapoint draw, situations, table and log loss, so the datapoints are
exactly the grader's. The pooled bootstrap resamples matches (default_rng(0), 1,000 reps, the same resamples for every seed and
arm), and the pooled statistic is the mean over seeds of the per-seed weighted difference.

Check: with SEEDS = [0] the point estimates and intervals must match score_gdt.py's own vs_first output for seed 0.
"""
import importlib.util
import sys

import numpy as np
import polars as pl

SCORE = "C:/Users/<user>/Documents/pureskill/counterstrike-win-probability/evaluation/intern_gdt/score_gdt.py"
TABLE = "C:/Users/<user>/Documents/pureskill/counterstrike-win-probability/evaluation/intern_gdt/table.csv"
ROWS = "G:/pureskill/csds-cs2/data/diagnostics/gdt_seq/rows_all_arms.parquet"
REF = "gru"
ARMS = ["xlstm_fast", "xlstm_blk", "xlstm_sig", "rwkv7", "mingru", "tcn"]
SEEDS = [0, 1, 2, 3, 4] if "--check" not in sys.argv else [0]
REPS = 1000

spec = importlib.util.spec_from_file_location("sg", SCORE)
sg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sg)

models = [f"{REF}_s{s}" for s in SEEDS] + [f"{a}_s{s}" for a in ARMS for s in SEEDS]
df = pl.read_parquet(ROWS)
df = df.select(sg.NEED + models)
df = df.with_columns(pl.when(pl.col("planted") == 0).then(pl.col("since_plant").fill_null(0)).otherwise(pl.col("since_plant"))
                     .alias("since_plant"))
df = df.drop_nulls()
df = sg.situations(df)
table = pl.read_csv(TABLE, schema_overrides={"tb": pl.Utf8, "econ_ct": pl.Utf8, "econ_t": pl.Utf8})
table = table.select(sg.CELL + [pl.col("p_ct").alias("p_table")])
pts = sg.draw(df, models, 0).join(table, on=sg.CELL, how="left")
pts = pts.filter(pl.col("p_table").is_not_null())
pts = pts.join(pts.group_by(sg.CELL).agg(pl.len().alias("n_s")), on=sg.CELL, how="left")

y = pts["y"].to_numpy().astype("float64")
mids = pts["match"].to_numpy()
ids, inv = np.unique(mids, return_inverse=True)
M = len(ids)
n_s = pts["n_s"].to_numpy().astype("float64")
w_rounds = np.ones(len(y))
w_sit = np.where(n_s >= sg.SIT_MIN, 1.0 / n_s, 0.0)
L = {m: sg.logloss(pts[m].to_numpy(), y) for m in models}
L_table = sg.logloss(pts["p_table"].to_numpy(), y)
print(f"datapoints {len(y):,}, matches {M}, seeds {SEEDS}", flush=True)

# the same resamples for every seed, arm and statistic (grader's wboot draws them the same way: default_rng(0), one per rep)
rng = np.random.default_rng(0)
PICKS = [rng.integers(0, M, M) for _ in range(REPS)]


def per_match(l_ref, l_arm, w):
    d = np.bincount(inv, weights=w * (l_ref - l_arm), minlength=M)  # sum of w * (reference loss - arm loss) per match
    sw = np.bincount(inv, weights=w, minlength=M)                  # sum of w per match
    return d, sw


def pooled(arm, ref_loss, arm_loss_key, w):
    """Mean over seeds of the per-seed weighted difference (ref minus arm; positive = arm better), with bootstrap over matches."""
    Ds, Ss = [], []
    for s in SEEDS:
        d, sw = per_match(L[f"{ref_loss}_s{s}"], L[f"{arm_loss_key}_s{s}"], w)
        Ds.append(d)
        Ss.append(sw)
    point = float(np.mean([d.sum() / sw.sum() for d, sw in zip(Ds, Ss)]))
    boot = np.array([np.mean([d[p].sum() / sw[p].sum() for d, sw in zip(Ds, Ss)]) for p in PICKS])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    per_seed = [float(d.sum() / sw.sum()) for d, sw in zip(Ds, Ss)]
    return point, float(lo), float(hi), per_seed


if "--check" in sys.argv:
    arm = "xlstm_fast"
    pt, lo, hi, _ = pooled(arm, REF, arm, w_rounds)
    print(f"CHECK rounds weights {arm} seed 0: {pt:+.6f} [{lo:+.6f}, {hi:+.6f}] (grader: +0.002355 [+0.000608, +0.004296])")
    pt, lo, hi, _ = pooled(arm, REF, arm, w_sit)
    print(f"CHECK situations weights {arm} seed 0: {pt:+.6f} [{lo:+.6f}, {hi:+.6f}] (grader: -0.000295)")
    sys.exit(0)

print("\nPOOLED over seeds 0 to 4, GRU control as reference (positive = arm better than GRU)")
print(f"{'arm':<12} {'GDT diff':>10} {'95% interval':>26} {'GDT_sit':>10} {'95% interval':>26} {'log loss diff':>14} {'95% interval':>26}")
rows = []
for arm in ARMS:
    g = pooled(arm, REF, arm, w_rounds)
    gs = pooled(arm, REF, arm, w_sit)
    # log loss alongside: per datapoint, rounds weights, same resamples
    ll = pooled(arm, REF, arm, w_rounds)  # placeholder replaced below
    d_ll, s_ll = [], []
    for s in SEEDS:
        d, sw = per_match(L[f"{REF}_s{s}"], L[f"{arm}_s{s}"], w_rounds)
        d_ll.append(d)
        s_ll.append(sw)
    ll_point = float(np.mean([d.sum() / sw.sum() for d, sw in zip(d_ll, s_ll)]))
    ll_boot = np.array([np.mean([d[p].sum() / sw[p].sum() for d, sw in zip(d_ll, s_ll)]) for p in PICKS])
    ll_lo, ll_hi = np.percentile(ll_boot, [2.5, 97.5])
    bar = "PASS" if (g[1] > 0 and gs[0] > 0) else ("LOSE" if g[2] < 0 else "no claim")
    rows.append((g[0], arm, bar))
    print(f"{arm:<12} {g[0]:+10.6f} [{g[1]:+.6f}, {g[2]:+.6f}] {gs[0]:+10.6f} [{gs[1]:+.6f}, {gs[2]:+.6f}] "
          f"{ll_point:+14.6f} [{ll_lo:+.6f}, {ll_hi:+.6f}]  BAR {bar}")
    print(f"             per seed GDT diff: " + ", ".join(f"s{s} {v:+.5f}" for s, v in zip(SEEDS, g[3])))

passing = [r for r in rows if r[2] == "PASS"]
if passing:
    best = max(passing)
    print(f"\nSELECTION (pin c807884d): {best[1]} (pooled GDT diff {best[0]:+.6f}); ties reported, not decided")
else:
    print("\nSELECTION (pin c807884d): no arm passes the BAR")
