"""Tick-rate experiment: training on 1, 2, 4, 8 and 16 random ticks per second (the 16-per-second status store), against every second.

The random-tick sweep (exp_random_ticks.py / exp_random_ticks59.py) drew 1 to 64 ticks per ROUND, which is 0.015 to about 0.94 rows per second.
This script tests rates above one per second: for each second of a training match, r of its quarter-second ticks are kept (random, nested:
the first r of a random order within each second). The grid holds 4 quarter-second ticks per second (every 16 ticks), so r = 1, 2, 4;
r = 4 keeps every quarter-second tick. Rates 8 and 16 need new grids and status extraction.

Features: the 58-input set (59 minus heavy, plus score lead), built for the 16-per-second rows from grid_step16 (event inputs) and
data/status_16hz (loadout inputs, read one day file at a time through features_status.loadout_features). Parity is checked first: the
16-per-second rows at second boundaries must equal the cache's rows (cache_snapshot_2026) on every feature.

Training matches: a fixed random 25% of the frozen split's training matches (seed 0), the same matches for every rate and for the
every-second reference. Validation (early stopping) and test (scored once) are the every-second rows of the frozen split, as before.
Settings: winprob_v4 settings on GPU (depth 4, eta 0.05, min child weight 20, subsample 0.8, early stopping 100, monotone alive counts,
score lead unconstrained). Each rate is paired with the every-second model on the same matches.

  ...python.exe -I -X utf8 scripts\\model\\exp_tick_rates.py
Writes data/combined/exp_tick_rates.json.
"""
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl

import features_status as FS
import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")
HEAVY = ["st_heavy_ct", "st_heavy_t"]
RATES = [1, 2, 4]  # data/combined/grid_step16.parquet is every 16 ticks = 4 rows per second; 8 and 16 per second need new grids and status extraction
FRAC = 0.25
MAPS = ["ancient", "anubis", "cache", "dust2", "inferno", "mirage", "nuke"]
EVENT = ["alive_ct", "alive_t", "ct_alive_minus_t", "bomb_planted", "seconds_since_round_live", "seconds_since_plant"] + [f"map_is_{m}" for m in MAPS]


def main():
    t0 = time.time()
    d = os.path.join(W.ROOT, "data", "combined")
    status16 = os.path.join(W.ROOT, "data", "status_16hz")
    ref_meta = json.load(open(os.path.join(W.ROOT, "models", "winprob_v4.meta.json"), encoding="utf-8"))
    feats59 = json.load(open(os.path.join(W.ROOT, "models", "winprob_v3.meta.json"), encoding="utf-8"))["features"]
    mono59 = {f: int(ref_meta["monotone_constraints"][f]) for f in feats59}
    feats = [f for f in feats59 if f not in HEAVY] + ["score_lead_diag"]
    st_names = [f for f in feats if f.startswith("st_")]
    assert set(EVENT) <= set(feats) and len(st_names) == 44 and len(feats) == 58, "58 inputs = 13 event + 44 loadout + score lead"
    mono = [mono59.get(f, 0) for f in feats]

    # frozen split and every-second validation / test rows (the cache), as in the other experiments
    need = list(dict.fromkeys(["match_key", "round", "tick", "second", "label_ct_win", "w_round"] + feats59))
    G = pl.scan_parquet(os.path.join(d, "cache_snapshot_2026.parquet")).select(need).with_columns(pl.col(feats59).cast(pl.Float32)).collect()
    lead = (pl.scan_parquet(os.path.join(d, "grid.parquet")).filter(pl.col("era") == "2026").select(["match_key", "round", "score_lead_diag"])
            .unique(subset=["match_key", "round"]).collect())
    G = G.join(lead, on=["match_key", "round"], how="left").with_columns(pl.col("score_lead_diag").fill_null(0).cast(pl.Float32))
    kdf = G.select("match_key").unique().with_columns(pl.lit(1.0).alias("w_round"), pl.lit(0).alias("label_ct_win"),
                                                      pl.lit("2026").alias("era"), pl.lit("").alias("match_date"))
    sp = W.make_split("random", kdf, seed=0)
    tr, va, te = sorted(sp["train"]), sorted(sp["val"]), sorted(sp["test"])
    del kdf
    sub_va, Xva, yva = W.arrays(G, va, feats, stride=1)
    sub_te, Xte, yte = W.arrays(G, te, feats, stride=1)

    rng = np.random.default_rng(0)
    S = sorted(rng.choice(np.array(tr), size=int(round(FRAC * len(tr))), replace=False).tolist())
    print(f"training matches {len(S)} of {len(tr)} ({time.time() - t0:.0f}s)", flush=True)

    # 16-per-second rows of the training matches, with the event inputs from grid_step16 and the loadout inputs from the status store
    # grid_step16 stores match_key as an integer id (mk_id); grid_step16_matches.parquet maps it to the match-key string
    mk_map = pl.read_parquet(os.path.join(d, "grid_step16_matches.parquet"), columns=["match_key", "mk_id"])
    S_ids = mk_map.filter(pl.col("match_key").is_in(S))["mk_id"].to_list()
    g16 = (pl.scan_parquet(os.path.join(d, "grid_step16.parquet")).filter((pl.col("era") == "2026") & pl.col("match_key").is_in(S_ids))
           .select(["match_key", "round", "tick", "second", "alive_ct", "alive_t", "ct_alive_minus_t", "bomb_planted",
                    "seconds_since_round_live", "seconds_since_plant", "label_ct_win", "map_name"]).collect())
    g16 = g16.rename({"match_key": "mk_id"}).join(mk_map, on="mk_id", how="inner").drop("mk_id")
    g16 = g16.with_columns([(pl.col("map_name") == f"de_{m}").cast(pl.Float32).alias(f"map_is_{m}") for m in MAPS])
    g16 = g16.join(lead, on=["match_key", "round"], how="left").with_columns(pl.col("score_lead_diag").fill_null(0).cast(pl.Float32))
    t_first = pl.col("tick").min().over(["match_key", "round"])
    g16 = g16.with_columns((t_first).alias("t0")).with_columns(((pl.col("tick") - pl.col("t0")) % 64 == 0).alias("boundary"))
    day_of = pl.read_parquet(os.path.join(d, "grid.parquet"), columns=["match_key", "upload_day"]).unique(subset=["match_key"])
    g16 = g16.join(day_of, on="match_key", how="left")
    parts = []
    for day in sorted(g16["upload_day"].drop_nulls().unique().to_list()):
        f = os.path.join(status16, day.replace("/", "-") + ".parquet")
        sub = g16.filter(pl.col("upload_day") == day)
        if not os.path.exists(f):
            print(f"  {day}: no status file ({sub.height:,} rows) skipped", flush=True)
            continue
        st = FS.loadout_features(sub.select(["match_key", "round", "tick"]), status_dir=status16, files=[f])
        missing = [c for c in st_names if c not in st.columns]
        if missing:
            sys.exit(f"loadout columns missing: {missing[:5]}")
        parts.append(sub.select(["match_key", "round", "tick", "second", "t0", "boundary", "label_ct_win"] + EVENT + ["score_lead_diag"])
                     .hstack(st.select(st_names)))
    big = pl.concat(parts)
    # whole-second index from the freeze-end tick (the grid's own "second" column counts rows, not seconds)
    big = big.with_columns(pl.Series("u", rng.random(big.height)), ((pl.col("tick") - pl.col("t0")) // 64).alias("sec"))
    big = big.with_columns((pl.col("u").rank("ordinal").over(["match_key", "round", "sec"]) - 1).cast(pl.Int32).alias("rank"))
    print(f"16-per-second training rows {big.height:,}; second-boundary rows {int(big['boundary'].sum()):,} ({time.time() - t0:.0f}s)", flush=True)

    # parity: boundary rows against the cache rows of the same matches and ticks, feature by feature
    cache_s = G.filter(pl.col("match_key").is_in(S)).select(["match_key", "round", "tick", "label_ct_win"] + feats)
    bnd = big.filter(pl.col("boundary")).select(["match_key", "round", "tick"] + feats + ["label_ct_win"])
    j = cache_s.join(bnd, on=["match_key", "round", "tick"], how="inner", suffix="_16")
    print(f"parity: {j.height:,} of {cache_s.height:,} cache rows matched on the boundary ticks", flush=True)
    worst = {}
    for f in feats:
        diff = (j[f].cast(pl.Float64) - j[f + "_16"].cast(pl.Float64)).abs().max()
        worst[f] = float(diff) if diff is not None else float("nan")
    bad = {f: v for f, v in worst.items() if not (v <= 1e-5)}
    print(f"parity max abs difference over {len(feats)} features: {max(worst.values()):.2e}; features differing: {bad if bad else 'none'}", flush=True)
    label_diff = int((j["label_ct_win"] != j["label_ct_win_16"]).sum())
    print(f"parity labels differing: {label_diff}", flush=True)
    del cache_s, bnd, j

    out = {"matches": len(S), "training_rows_every_second": None, "rates": []}
    ref_bnd = big.filter(pl.col("boundary"))
    Xr = ref_bnd.select(feats).to_numpy().astype("float32")
    yr = ref_bnd["label_ct_win"].to_numpy()
    m, best = W.fit_xgb(Xr, yr, Xva, yva, mono, seed=0, device="cuda")
    p_ref_te, p_ref_va = W.predict(m, best, Xte), W.predict(m, best, Xva)
    ev_ref = W.evaluate(sub_te, yte, p_ref_te)["logloss"]
    out["every_second"] = {"rows": int(len(yr)), "best_iteration": int(best), "test_logloss": ev_ref,
                           "validation_logloss": W.evaluate(sub_va, yva, p_ref_va)["logloss"]}
    out["training_rows_every_second"] = int(len(yr))
    print(f"every second (same {len(S)} matches): rows {len(yr):,}; best {best}; test {ev_ref:.6f} ({time.time() - t0:.0f}s)", flush=True)
    W.dump_json(out, os.path.join(d, "exp_tick_rates.json"))
    del Xr, yr, ref_bnd

    for r in RATES:
        sel = big.filter(pl.col("rank") < r)
        Xr = sel.select(feats).to_numpy().astype("float32")
        yr = sel["label_ct_win"].to_numpy()
        m, best = W.fit_xgb(Xr, yr, Xva, yva, mono, seed=0, device="cuda")
        p_te, p_va = W.predict(m, best, Xte), W.predict(m, best, Xva)
        ev = W.evaluate(sub_te, yte, p_te, ref=p_ref_te)
        evv = W.evaluate(sub_va, yva, p_va, ref=p_ref_va)
        lo, hi = ev["delta_vs_ref_ci95"]
        row = {"ticks_per_second": r, "rows": int(len(yr)), "rows_vs_every_second": round(len(yr) / out["training_rows_every_second"], 3),
               "best_iteration": int(best), "validation_logloss": evv["logloss"], "test_logloss": ev["logloss"],
               "test_delta_vs_every_second": ev["delta_vs_ref"], "test_delta_ci95": [lo, hi],
               "validation_delta_vs_every_second": evv["delta_vs_ref"]}
        out["rates"].append(row)
        print(f"{r} ticks/s: rows {len(yr):,} ({row['rows_vs_every_second']:.2f} of every second); best {best}; "
              f"test {ev['logloss']:.6f} (delta {ev['delta_vs_ref']:+.5f} [{lo:+.5f}, {hi:+.5f}]); "
              f"validation delta {evv['delta_vs_ref']:+.5f} ({time.time() - t0:.0f}s)", flush=True)
        W.dump_json(out, os.path.join(d, "exp_tick_rates.json"))
        del Xr, yr, sel, m
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
