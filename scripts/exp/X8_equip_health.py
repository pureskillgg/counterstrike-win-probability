"""X8: equipment and health gain on the 150-match state sample (per-second rows).

Builds one row per (match_key, round, tick) from players_1hz.parquet + rounds.parquet.
5-fold GroupKFold by match, XGBoost CUDA, depth 4, eta 0.05, 300 trees, subsample 0.8.
Feature sets: base, base + equipment, base + equipment + health/armor.
Paired differences with a match-cluster bootstrap (1000 resamples, seed 0).
"""
import os

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import time

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.model_selection import GroupKFold

SAMPLE = r"G:\pureskill\csds-cs2\data\state_sample"
RIFLE = [7, 8, 9, 10, 11, 13, 16, 38, 39, 60]
TICKS_PER_SEC = 64.0
EPS = 1e-15
N_BOOT = 1000
NROUNDS = 300

BASE = ["alive_ct", "alive_t", "ct_alive_minus_t", "bomb_planted", "seconds_since_round_live",
        "seconds_since_plant", "round_number", "ct_score_minus_t_score", "is_pistol_round", "is_overtime"]
MONO_BASE = [1, -1, 1, 0, 0, 0, 0, 1, 0, 0]
EQUIP = ["ct_equip", "t_equip", "ct_rifles", "t_rifles"]
HEALTH = ["ct_health", "t_health", "ct_armor", "t_armor"]
SETS = {
    "base": BASE,
    "base_equip": BASE + EQUIP,
    "base_equip_health": BASE + EQUIP + HEALTH,
}


def mono_for(cols):
    vals = []
    for c in cols:
        vals.append(str(MONO_BASE[BASE.index(c)]) if c in BASE else "0")
    return "(" + ",".join(vals) + ")"


def ll_vec(p, y):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_predict(Xtr, ytr, Xte, cols, mono, tries=4):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": "cuda",
        "nthread": 4,
        "max_depth": 4,
        "eta": 0.05,
        "min_child_weight": 20,
        "subsample": 0.8,
        "monotone_constraints": mono,
        "seed": 0,
    }
    for k in range(tries):
        try:
            dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=cols)
            bst = xgb.train(params, dtr, num_boost_round=NROUNDS)
            return bst.predict(xgb.DMatrix(Xte, feature_names=cols))
        except xgb.core.XGBoostError as e:
            msg = str(e).lower()
            if ("out of memory" in msg or "cuda" in msg or "oom" in msg) and k < tries - 1:
                print("CUDA error, waiting 30s then retry:", str(e)[:200], flush=True)
                time.sleep(30)
                continue
            raise


def main():
    print("xgboost", xgb.__version__, "USE_CUDA", xgb.build_info().get("USE_CUDA"), flush=True)
    P = pl.read_parquet(os.path.join(SAMPLE, "players_1hz.parquet"))
    R = pl.read_parquet(os.path.join(SAMPLE, "rounds.parquet"))
    print("players", P.shape, "rounds", R.shape, flush=True)

    P = P.with_columns([
        pl.col("round").cast(pl.Int64),
        pl.col("tick").cast(pl.Int64),
        pl.col("health").cast(pl.Float64),
        pl.col("armor").cast(pl.Float64),
        pl.col("equipment_value_calc").cast(pl.Float64),
        pl.col("inv_primary").cast(pl.Int64),
        pl.col("team_code").cast(pl.Int64),
        pl.col("seconds_since_live").cast(pl.Float64),
    ])
    R = R.select([
        pl.col("match_key"),
        pl.col("round").cast(pl.Int64),
        pl.col("ct_win").cast(pl.Int64),
        pl.col("round_number").cast(pl.Int64),
        pl.col("ct_score").cast(pl.Int64),
        pl.col("t_score").cast(pl.Int64),
        pl.col("plant_tick").cast(pl.Int64),
        pl.col("live_start_tick").cast(pl.Int64),
        pl.col("round_end_tick").cast(pl.Int64),
    ])

    P = P.join(R.select(["match_key", "round", "live_start_tick", "round_end_tick"]), on=["match_key", "round"], how="inner")
    n0 = P.height
    P = P.filter((pl.col("tick") >= pl.col("live_start_tick")) & (pl.col("tick") <= pl.col("round_end_tick")))
    print("player rows", n0, "dropped outside live window", n0 - P.height, flush=True)

    # freeze-end snapshot = first tick of each round
    ftick = P.group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("ftick"))
    chk = ftick.join(R.select(["match_key", "round", "live_start_tick"]), on=["match_key", "round"])
    print("rounds where first tick != live_start_tick:", chk.filter(pl.col("ftick") != pl.col("live_start_tick")).height,
          "of", chk.height, flush=True)
    P = P.join(ftick, on=["match_key", "round"])
    fz = (P.filter(pl.col("tick") == pl.col("ftick"))
          .group_by(["match_key", "round", "team_code"])
          .agg([pl.col("equipment_value_calc").sum().alias("eq"),
                pl.col("inv_primary").is_in(RIFLE).cast(pl.Int64).sum().alias("rifles")]))

    def side(code, prefix):
        return fz.filter(pl.col("team_code") == code).select([
            "match_key", "round",
            pl.col("eq").alias(f"{prefix}_equip"),
            pl.col("rifles").alias(f"{prefix}_rifles"),
        ])

    EQ = side(3, "ct").join(side(2, "t"), on=["match_key", "round"], how="full", coalesce=True)

    # per-second aggregates over alive players
    ct = pl.col("team_code") == 3
    tt = pl.col("team_code") == 2
    S = P.group_by(["match_key", "round", "tick"]).agg([
        ct.cast(pl.Int64).sum().alias("alive_ct"),
        tt.cast(pl.Int64).sum().alias("alive_t"),
        pl.when(ct).then(pl.col("health")).otherwise(0.0).sum().alias("ct_health"),
        pl.when(tt).then(pl.col("health")).otherwise(0.0).sum().alias("t_health"),
        pl.when(ct & (pl.col("armor") > 0)).then(1).otherwise(0).sum().alias("ct_armor"),
        pl.when(tt & (pl.col("armor") > 0)).then(1).otherwise(0).sum().alias("t_armor"),
        pl.col("seconds_since_live").max().alias("sec_live"),
    ])
    S = S.join(R, on=["match_key", "round"], how="left")
    S = S.join(EQ, on=["match_key", "round"], how="left")
    S = S.with_columns([
        pl.col("ct_equip").fill_null(0.0),
        pl.col("t_equip").fill_null(0.0),
        pl.col("ct_rifles").fill_null(0).cast(pl.Float64),
        pl.col("t_rifles").fill_null(0).cast(pl.Float64),
    ])

    planted = pl.col("plant_tick").is_not_null() & (pl.col("tick") >= pl.col("plant_tick"))
    S = S.with_columns([
        (pl.col("alive_ct") - pl.col("alive_t")).cast(pl.Float64).alias("ct_alive_minus_t"),
        planted.fill_null(False).cast(pl.Float64).alias("bomb_planted"),
        pl.col("sec_live").alias("seconds_since_round_live"),
        pl.when(planted.fill_null(False))
          .then((pl.col("tick") - pl.col("plant_tick")).cast(pl.Float64) / TICKS_PER_SEC)
          .otherwise(-1.0).alias("seconds_since_plant"),
        (pl.col("ct_score") - pl.col("t_score")).cast(pl.Float64).alias("ct_score_minus_t_score"),
        pl.col("round_number").is_in([1, 13]).cast(pl.Float64).alias("is_pistol_round"),
        (pl.col("round_number") > 24).cast(pl.Float64).alias("is_overtime"),
        pl.col("round_number").cast(pl.Float64).alias("round_number_f"),
        pl.col("ct_win").cast(pl.Float64).alias("y"),
        pl.len().over(["match_key", "round"]).cast(pl.Float64).alias("round_rows"),
    ])
    S = S.with_columns(pl.col("round_number_f").alias("round_number"))
    S = S.sort(["match_key", "round", "tick"])

    n = S.height
    print("rows (match,round,tick)", n, "matches", S["match_key"].n_unique(), "rounds",
          S.select(["match_key", "round"]).unique().height, flush=True)
    print("nulls in feature cols:", S.select(BASE + EQUIP + HEALTH).null_count().row(0), flush=True)
    print("CT win rate", float(S["y"].mean()), flush=True)

    uniq, mid = np.unique(S["match_key"].to_numpy(), return_inverse=True)
    mid = mid.astype(np.int64)
    M_n = len(uniq)
    y = S["y"].to_numpy().astype(np.float64)
    wr = 1.0 / S["round_rows"].to_numpy().astype(np.float64)

    all_cols = BASE + EQUIP + HEALTH
    A = S.select([pl.col(c).cast(pl.Float64) for c in all_cols]).to_numpy()
    colidx = {c: i for i, c in enumerate(all_cols)}
    mats = {k: A[:, [colidx[c] for c in cols]].astype(np.float32) for k, cols in SETS.items()}

    folds = list(GroupKFold(n_splits=5).split(np.zeros(n), y, groups=mid))
    for k, (tr, te) in enumerate(folds):
        print(f"fold {k}: train rows {len(tr)} test rows {len(te)} test matches {len(np.unique(mid[te]))}", flush=True)

    oof = {}
    for name, cols in SETS.items():
        X = mats[name]
        p = np.zeros(n, dtype=np.float64)
        t0 = time.time()
        for k, (tr, te) in enumerate(folds):
            p[te] = fit_predict(X[tr], y[tr], X[te], cols, mono_for(cols))
        oof[name] = p
        ll = ll_vec(p, y)
        print(f"{name}: {len(cols)} features, OOF logloss {ll.mean():.6f}, time {time.time() - t0:.1f}s", flush=True)

    # match-cluster bootstrap
    rng = np.random.default_rng(0)
    idx = rng.integers(0, M_n, size=(N_BOOT, M_n))
    cnt = np.bincount(mid, minlength=M_n).astype(np.float64)
    cnt_b = cnt[idx].sum(axis=1)

    def pct(a):
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    def boot_ratio(v, w):
        s = np.bincount(mid, weights=v * w, minlength=M_n)
        d = np.bincount(mid, weights=w, minlength=M_n)
        point = float(s.sum() / d.sum())
        boots = s[idx].sum(axis=1) / d[idx].sum(axis=1)
        return point, pct(boots)

    ones = np.ones(n)
    lls = {k: ll_vec(v, y) for k, v in oof.items()}
    out = {"n_rows": int(n), "n_matches": int(M_n), "models": {}, "paired": {}}
    for k in SETS:
        pt, ci = boot_ratio(lls[k], ones)
        ptr, cir = boot_ratio(lls[k], wr)
        brier = float(np.mean((oof[k] - y) ** 2))
        out["models"][k] = {"logloss": pt, "ci95": ci, "round_weighted": ptr, "round_weighted_ci95": cir,
                            "brier": brier}
        print(f"MODEL {k}: logloss {pt:.5f} CI {ci[0]:.5f},{ci[1]:.5f} | round-weighted {ptr:.5f} "
              f"CI {cir[0]:.5f},{cir[1]:.5f} | brier {brier:.5f}", flush=True)

    for a, b in [("base_equip", "base"), ("base_equip_health", "base"), ("base_equip_health", "base_equip")]:
        d = lls[a] - lls[b]
        pt, ci = boot_ratio(d, ones)
        ptr, cir = boot_ratio(d, wr)
        key = f"{a} minus {b}"
        out["paired"][key] = {"delta": pt, "ci95": ci, "round_weighted_delta": ptr, "round_weighted_ci95": cir}
        print(f"PAIRED {key}: delta {pt:+.5f} CI [{ci[0]:+.5f}, {ci[1]:+.5f}] | round-weighted {ptr:+.5f} "
              f"CI [{cir[0]:+.5f}, {cir[1]:+.5f}]", flush=True)

    print("JSON", __import__("json").dumps(out), flush=True)


if __name__ == "__main__":
    main()
