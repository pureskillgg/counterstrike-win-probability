# X15 models: 5-fold match-grouped OOF XGBoost (CUDA) for four feature sets, paired match-cluster bootstrap,
# and hero-death analysis from out-of-fold predictions.
import os
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
import time
import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.model_selection import GroupKFold

SCR = "C:/Users/13144/AppData/Local/Temp/claude/C--Users-13144-Documents-pureskill/e0a18ad2-aa9e-4a3c-9946-29641b2cde82/scratchpad/"
t0 = time.time()
g = pl.read_parquet(SCR + "X15_grid.parquet")
d = pl.read_parquet(SCR + "X15_deaths.parquet")
print("grid", g.shape, "deaths", d.shape, flush=True)

BASE = ["alive_ct", "alive_t", "ct_alive_minus_t", "bomb_planted", "seconds_since_plant",
        "seconds_since_live", "round_number", "score_lead", "pistol"]
STATIC = ["ct_eq_fe", "t_eq_fe"]
DYN = ["ct_eq", "t_eq", "ct_ra", "t_ra", "ct_awp", "t_awp", "ct_arm", "t_arm", "ct_hp", "t_hp"]
SETS = {"a": BASE, "b": BASE + STATIC, "c": BASE + DYN, "d": BASE + STATIC + DYN}
print("feature counts:", {k: len(v) for k, v in SETS.items()})

g = g.with_columns(pl.col("seconds_since_live").cast(pl.Int32).alias("sec_i"))
uniq, mcode = np.unique(g["match_key"].to_numpy(), return_inverse=True)
M = len(uniq)
y = g["y"].to_numpy().astype(np.float64)
sec = g["sec_i"].to_numpy()
print("matches", M, "grid rows", len(y), "ct_win rate", round(y.mean(), 4), flush=True)

def fit_pred(Xtr, ytr, Xte):
    last = None
    for attempt in range(3):
        try:
            m = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                                  min_child_weight=20, random_state=0, device="cuda",
                                  tree_method="hist", objective="binary:logistic", eval_metric="logloss")
            m.fit(Xtr, ytr)
            return m.predict_proba(Xte)[:, 1]
        except Exception as e:
            last = e
            print("fit error attempt", attempt, repr(e)[:300], flush=True)
            time.sleep(30)
    raise last

folds = list(GroupKFold(n_splits=5).split(np.zeros(len(y)), y, groups=mcode))
print("fold sizes (rows, matches):", [(len(te), len(np.unique(mcode[te]))) for _, te in folds], flush=True)

oof = {}
for name, cols in SETS.items():
    X = g.select(cols).to_numpy().astype(np.float32)
    pr = np.zeros(len(y))
    for k, (tr, te) in enumerate(folds):
        pr[te] = fit_pred(X[tr], y[tr], X[te])
    oof[name] = pr
    print("model", name, "done at", round(time.time() - t0, 1), "s", flush=True)

def ll(p, yy):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(yy * np.log(p) + (1 - yy) * np.log(1 - p))

LL = {k: ll(v, y) for k, v in oof.items()}

BINS = [("0-20s", sec < 20), ("20-45s", (sec >= 20) & (sec < 45)),
        ("45-70s", (sec >= 45) & (sec < 70)), ("70s+", sec >= 70), ("all", np.ones(len(y), dtype=bool))]

rng = np.random.default_rng(0)
B = 1000
IDX = rng.integers(0, M, size=(B, M))

def boot_mean(vals, codes, mask):
    c_m = np.bincount(codes[mask], minlength=M).astype(np.float64)
    s_m = np.bincount(codes[mask], weights=vals[mask], minlength=M)
    S = s_m[IDX].sum(axis=1)
    C = c_m[IDX].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = S / C
    point = s_m.sum() / c_m.sum() if c_m.sum() > 0 else float("nan")
    return point, np.nanpercentile(R, 2.5), np.nanpercentile(R, 97.5), int(c_m.sum()), int((c_m > 0).sum())

print("\n=== Table 1: out-of-fold log loss (row level; matches in bin) ===", flush=True)
print("%-8s %-8s %8s %8s %8s %8s %8s" % ("bin", "model", "LL", "rows", "matches", "", ""))
for bn, mask in BINS:
    for name in ["a", "b", "c", "d"]:
        v = LL[name][mask]
        nm = len(np.unique(mcode[mask]))
        print("%-8s %-8s %8.4f %8d %8d" % (bn, name, v.mean(), mask.sum(), nm))
    print()

print("=== Table 2: paired log-loss differences (negative = second model better), match-cluster bootstrap 95% CI, B=1000 seed 0 ===", flush=True)
PAIRS = [("c-a", "c", "a"), ("c-b", "c", "b"), ("d-b", "d", "b")]
print("%-8s %-6s %9s %22s %8s %8s" % ("bin", "pair", "diff", "95% CI", "rows", "matches"))
for bn, mask in BINS:
    for label, m1, m0 in PAIRS:
        diff = LL[m1] - LL[m0]
        pt, lo, hi, nr, nm = boot_mean(diff, mcode, mask)
        print("%-8s %-6s %9.5f [%9.5f, %9.5f] %8d %8d" % (bn, label, pt, lo, hi, nr, nm))
    print()

# ---------------- hero-death analysis ----------------
g = g.with_columns([pl.Series("p_a", oof["a"]), pl.Series("p_c", oof["c"])])
last_sec = g.group_by(["match_key", "round"]).agg(pl.col("sec_i").max().alias("ls"))
lastrows = g.join(last_sec, on=["match_key", "round"]).filter(pl.col("sec_i") == pl.col("ls"))
print("alive players on final snapshot (should equal absences dropped by round-end rule):",
      int((lastrows["ct_n"] + lastrows["t_n"]).sum()), flush=True)

BC = ["ct_ra", "t_ra", "ct_n", "t_n", "p_a", "p_c"]
gt = g.select(["match_key", "round", pl.col("sec_i").alias("sec")] + [pl.col(c).alias(c + "_t") for c in BC])
gt1 = g.select(["match_key", "round", pl.col("sec_i").alias("sec1")] + [pl.col(c).alias(c + "_t1") for c in BC])
dd = (d.with_columns(pl.col("seconds_since_live").cast(pl.Int32).alias("sec"))
      .join(gt, on=["match_key", "round", "sec"], how="inner")
      .join(gt1, on=["match_key", "round", "sec1"], how="inner"))
print("deaths with both t and t+1 predictions:", dd.height, flush=True)

dd = dd.with_columns([
    pl.when(pl.col("team_code") == 3).then(pl.col("ct_ra_t")).otherwise(pl.col("t_ra_t")).alias("side_ra_t"),
    pl.when(pl.col("team_code") == 3).then(pl.col("ct_ra_t1")).otherwise(pl.col("t_ra_t1")).alias("side_ra_t1"),
    pl.when(pl.col("team_code") == 3).then(pl.col("p_a_t") - pl.col("p_a_t1")).otherwise(pl.col("p_a_t1") - pl.col("p_a_t")).alias("drop_a"),
    pl.when(pl.col("team_code") == 3).then(pl.col("p_c_t") - pl.col("p_c_t1")).otherwise(pl.col("p_c_t1") - pl.col("p_c_t")).alias("drop_c"),
])
dd = dd.with_columns((pl.col("side_ra_t1") - pl.col("side_ra_t")).alias("d_ra"))
mcode_d = np.searchsorted(uniq, dd["match_key"].to_numpy())
vc = dd["victim_class"].to_numpy()
drop_a = dd["drop_a"].to_numpy()
drop_c = dd["drop_c"].to_numpy()
dsec = dd["sec"].to_numpy()
ALLM = np.ones(dd.height, dtype=bool)

print("\n=== Table 3: mean signed change in predicted victim-side win probability, t to t+1 (positive = worse for victim side) ===", flush=True)
print("%-14s %-6s %7s %8s %22s %8s %22s" % ("victim_class", "model", "deaths", "matches", "mean drop 95% CI", "", ""))
for cl in ["AWP", "rifle", "other_primary", "pistol_only", "ALL"]:
    mask = ALLM if cl == "ALL" else (vc == cl)
    for mname, vals in [("a", drop_a), ("c", drop_c)]:
        pt, lo, hi, nr, nm = boot_mean(vals, mcode_d, mask)
        print("%-14s %-6s %7d %8d  %8.5f [%8.5f, %8.5f]" % (cl, mname, nr, nm, pt, lo, hi))
print()

print("=== Table 3b: same, by time bin (model c drop and model a drop) ===", flush=True)
DBINS = [("0-20s", dsec < 20), ("20-45s", (dsec >= 20) & (dsec < 45)), ("45-70s", (dsec >= 45) & (dsec < 70)), ("70s+", dsec >= 70)]
for bn, bm in DBINS:
    for cl in ["AWP", "rifle", "other_primary", "pistol_only"]:
        mask = bm & (vc == cl)
        na = int(mask.sum())
        if na == 0:
            print("%-7s %-14s n=0" % (bn, cl))
            continue
        ma = float(drop_a[mask].mean())
        mc = float(drop_c[mask].mean())
        print("%-7s %-14s n=%5d  mean drop a=%8.5f  c=%8.5f" % (bn, cl, na, ma, mc))
    print()

print("=== Table 4: rifle/AWP holders on victim side, t to t+1 (d_ra = count(t+1) - count(t), victim included in t) ===", flush=True)
for cl in ["AWP", "rifle"]:
    mask = vc == cl
    dr = dd["d_ra"].to_numpy()[mask]
    vals_, cnts_ = np.unique(dr, return_counts=True)
    print(cl, "deaths:", int(mask.sum()), " distribution of d_ra:", dict(zip(vals_.tolist(), cnts_.tolist())))
    for label, cond in [("d_ra == -1 (count minus one, weapon not recovered)", dd["d_ra"].to_numpy() == -1),
                        ("d_ra == 0 (same count: pickup or another holder)", dd["d_ra"].to_numpy() == 0),
                        ("d_ra <= -2 (another rifle/AWP holder also died)", dd["d_ra"].to_numpy() <= -2),
                        ("d_ra >= +1 (count rose)", dd["d_ra"].to_numpy() >= 1)]:
        pt, lo, hi, nr, nm = boot_mean(cond.astype(np.float64), mcode_d, mask)
        print("  %-55s share %6.2f%% [%6.2f, %6.2f]  n=%d matches=%d" % (label, 100 * pt, 100 * lo, 100 * hi, nr, nm))
    print()

g.select(["match_key", "round", "sec_i", "y", "p_a", "p_c", "ct_n", "t_n", "ct_ra", "t_ra"]).write_parquet(SCR + "X15_oof.parquet")
dd.write_parquet(SCR + "X15_hero_dd.parquet")
print("done in", round(time.time() - t0, 1), "s", flush=True)
