"""Tick-rate experiment at up to 8 ticks per second (within each second, random sub-ticks), on the 58-input set.

Rows: the 8-per-second grid (grid_step8_sub.parquet: every 8 ticks after the freeze end) for the same 2,644 training matches as
exp_tick_rates.py. Loadout features come from data/status_8hz (the 4-per-second rows already extracted plus the new 8-per-second rows,
merged by tick8_prepare.py). Within each second, r of its 8 ticks are kept (random, nested), r = 1, 2, 4, 8.
Reference: every second (the boundary ticks, freeze end + 64 k) on the same matches. Validation (early stopping) and test (scored once)
are the every-second rows of the frozen split, as before.

Checks before training: the 8-per-second features at the boundary ticks equal the cache rows on every feature; every training tick lies
inside its round; no duplicate (match, round, tick); training rows come only from the training subset; validation and test rows come from
the cache, never from the tick store.

  ...python.exe -I -X utf8 scripts\\model\\exp_tick8.py
Writes data/combined/exp_tick8.json.
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
RATES = [1, 2, 4, 8]
MAPS = ["ancient", "anubis", "cache", "dust2", "inferno", "mirage", "nuke"]
EVENT = ["alive_ct", "alive_t", "ct_alive_minus_t", "bomb_planted", "seconds_since_round_live", "seconds_since_plant"] + [f"map_is_{m}" for m in MAPS]


def main():
    t0 = time.time()
    d = os.path.join(W.ROOT, "data", "combined")
    status8 = os.path.join(W.ROOT, "data", "status_8hz")
    ref_meta = json.load(open(os.path.join(W.ROOT, "models", "winprob_v4.meta.json"), encoding="utf-8"))
    feats59 = json.load(open(os.path.join(W.ROOT, "models", "winprob_v3.meta.json"), encoding="utf-8"))["features"]
    mono59 = {f: int(ref_meta["monotone_constraints"][f]) for f in feats59}
    feats = [f for f in feats59 if f not in HEAVY] + ["score_lead_diag"]
    st_names = [f for f in feats if f.startswith("st_")]
    assert set(EVENT) <= set(feats) and len(st_names) == 44 and len(feats) == 58, "58 inputs = 13 event + 44 loadout + score lead"
    mono = [mono59.get(f, 0) for f in feats]

    # frozen split, every-second validation and test rows (the cache), as in the other experiments
    need = list(dict.fromkeys(["match_key", "round", "tick", "second", "label_ct_win", "w_round"] + feats59))
    G = pl.scan_parquet(os.path.join(d, "cache_snapshot_2026.parquet")).select(need).with_columns(pl.col(feats59).cast(pl.Float32)).collect()
    grid = pl.read_parquet(os.path.join(d, "grid.parquet"), columns=["match_key", "round", "score_lead_diag", "era", "upload_day"]).filter(pl.col("era") == "2026")
    lead = grid.select(["match_key", "round", "score_lead_diag"]).unique(subset=["match_key", "round"])
    G = G.join(lead, on=["match_key", "round"], how="left").with_columns(pl.col("score_lead_diag").fill_null(0).cast(pl.Float32))
    kdf = G.select("match_key").unique().with_columns(pl.lit(1.0).alias("w_round"), pl.lit(0).alias("label_ct_win"),
                                                      pl.lit("2026").alias("era"), pl.lit("").alias("match_date"))
    sp = W.make_split("random", kdf, seed=0)
    tr, va, te = sorted(sp["train"]), sorted(sp["val"]), sorted(sp["test"])
    del kdf
    sub_va, Xva, yva = W.arrays(G, va, feats, stride=1)
    sub_te, Xte, yte = W.arrays(G, te, feats, stride=1)
    S = set(pl.read_parquet(os.path.join(d, "tick_subset_matches.parquet"))["match_key"].to_list())
    assert S <= set(tr), "the training subset must be inside the frozen training split"

    # the 8-per-second grid of the subset, with keys, day and lead
    g8 = pl.read_parquet(os.path.join(d, "grid_step8_sub.parquet"))
    mapping = pl.read_parquet(os.path.join(d, "grid_step8_matches.parquet"), columns=["match_key", "mk_id"])
    g8 = g8.with_columns(pl.col("map_name").cast(pl.Utf8)).rename({"match_key": "mk_id"}).join(mapping, on="mk_id", how="left")
    g8 = g8.join(grid.select(["match_key", "upload_day"]).unique(subset=["match_key"]), on="match_key", how="left")
    g8 = g8.join(lead, on=["match_key", "round"], how="left").with_columns(pl.col("score_lead_diag").fill_null(0).cast(pl.Float32))
    g8 = g8.with_columns([(pl.col("map_name") == f"de_{m}").cast(pl.Float32).alias(f"map_is_{m}") for m in MAPS])
    g8 = g8.with_columns((pl.col("tick") - pl.col("tick").min().over(["match_key", "round"])).alias("off"))
    g8 = g8.with_columns(pl.col("off").alias("off_"), ((pl.col("tick") - pl.col("tick").min().over(["match_key", "round"])) // 64).alias("sec"))
    g8 = g8.with_columns(((pl.col("off") % 64) == 0).alias("boundary"))
    dup = g8.height - g8.unique(["match_key", "round", "tick"]).height
    assert dup == 0, f"{dup} duplicate (match, round, tick) rows in the 8-per-second grid"
    print(f"8-per-second rows {g8.height:,} for {g8['match_key'].n_unique()} matches ({time.time() - t0:.0f}s)", flush=True)

    # loadout features per day file of the merged tick store
    parts = []
    for day in sorted(g8["upload_day"].drop_nulls().unique().to_list()):
        f = os.path.join(status8, day.replace("/", "-") + ".parquet")
        sub = g8.filter(pl.col("upload_day") == day)
        if not os.path.exists(f):
            sys.exit(f"no merged status file for {day}")
        st = FS.loadout_features(sub.select(["match_key", "round", "tick"]), status_dir=status8, files=[f])
        parts.append(sub.select(["match_key", "round", "tick", "sec", "off", "boundary", "label_ct_win"] + EVENT + ["score_lead_diag"]).hstack(st.select(st_names)))
    big = pl.concat(parts)
    print(f"loadout features built ({big.height:,} rows, {time.time() - t0:.0f}s)", flush=True)

    # checks: boundary ticks equal the cache rows on every feature, and labels agree
    cache_s = G.filter(pl.col("match_key").is_in(list(S))).select(["match_key", "round", "tick", "label_ct_win"] + feats)
    bnd = big.filter(pl.col("boundary")).select(["match_key", "round", "tick", "label_ct_win"] + feats)
    j = cache_s.join(bnd, on=["match_key", "round", "tick"], how="inner", suffix="_8")
    worst = max(float((j[f].cast(pl.Float64) - j[f + "_8"].cast(pl.Float64)).abs().max() or 0.0) for f in feats)
    lab = int((j["label_ct_win"] != j["label_ct_win_8"]).sum())
    print(f"parity: {j.height:,} of {cache_s.height:,} cache rows matched; max abs feature difference {worst:.2e}; labels differing {lab}", flush=True)
    assert j.height == cache_s.height and worst <= 1e-5 and lab == 0, "8-per-second features do not reproduce the cache"
    n_before = int((big["off"] < 0).sum())
    n_off_grid = int(((big["off"] % 8) != 0).sum())
    assert n_before == 0 and n_off_grid == 0, f"ticks outside the freeze-end + 8k grid: before freeze {n_before}, off grid {n_off_grid}"
    print("in-round check: every tick is on the freeze-end + 8k grid and at or after the freeze end", flush=True)

    # training rows: the subset's 8-per-second rows
    rng = np.random.default_rng(1)
    big = big.with_columns(pl.Series("u", rng.random(big.height)))
    big = big.with_columns((pl.col("u").rank("ordinal").over(["match_key", "round", "sec"]) - 1).cast(pl.Int32).alias("rank"))
    ref_bnd = big.filter(pl.col("boundary"))
    Xr, yr = ref_bnd.select(feats).to_numpy().astype("float32"), ref_bnd["label_ct_win"].to_numpy()
    m, best = W.fit_xgb(Xr, yr, Xva, yva, mono, seed=0, device="cuda")
    p_ref_te, p_ref_va = W.predict(m, best, Xte), W.predict(m, best, Xva)
    ev_ref = W.evaluate(sub_te, yte, p_ref_te)["logloss"]
    out = {"matches": len(S), "every_second": {"rows": int(len(yr)), "best_iteration": int(best), "test_logloss": ev_ref,
                                                 "validation_logloss": W.evaluate(sub_va, yva, p_ref_va)["logloss"]}, "rates": []}
    print(f"every second (same {len(S)} matches): rows {len(yr):,}; best {best}; test {ev_ref:.6f} (earlier 0.457776) ({time.time() - t0:.0f}s)", flush=True)
    W.dump_json(out, os.path.join(d, "exp_tick8.json"))
    del Xr, yr, ref_bnd

    for r in RATES:
        sel = big.filter(pl.col("rank") < r)
        assert sel["match_key"].is_in(list(S)).all(), "training rows must come from the subset only"
        Xr, yr = sel.select(feats).to_numpy().astype("float32"), sel["label_ct_win"].to_numpy()
        m, best = W.fit_xgb(Xr, yr, Xva, yva, mono, seed=0, device="cuda")
        p_te, p_va = W.predict(m, best, Xte), W.predict(m, best, Xva)
        ev = W.evaluate(sub_te, yte, p_te, ref=p_ref_te)
        evv = W.evaluate(sub_va, yva, p_va, ref=p_ref_va)
        lo, hi = ev["delta_vs_ref_ci95"]
        row = {"ticks_per_second": r, "rows": int(len(yr)), "rows_vs_every_second": round(len(yr) / out["every_second"]["rows"], 3),
               "best_iteration": int(best), "validation_logloss": evv["logloss"], "test_logloss": ev["logloss"],
               "test_delta_vs_every_second": ev["delta_vs_ref"], "test_delta_ci95": [lo, hi], "validation_delta_vs_every_second": evv["delta_vs_ref"]}
        out["rates"].append(row)
        print(f"{r} ticks/s: rows {len(yr):,} ({row['rows_vs_every_second']:.2f} of every second); best {best}; test {ev['logloss']:.6f} "
              f"(delta {ev['delta_vs_ref']:+.5f} [{lo:+.5f}, {hi:+.5f}]); validation delta {evv['delta_vs_ref']:+.5f} ({time.time() - t0:.0f}s)", flush=True)
        W.dump_json(out, os.path.join(d, "exp_tick8.json"))
        del Xr, yr, sel, m
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
