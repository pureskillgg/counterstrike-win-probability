"""Are the 'unknown'-type FACEIT matches good enough to be in the model?   (CUDA XGBoost for the refits; run with an environment that has xgboost + polars)

  ...python.exe -I -X utf8 scripts\\model\\faceit_audit.py [--stride 3]

Part 1, data: by source (FACEIT servers vs Valve matchmaking vs other) and header match_type, 2026 only: sizes, format (MR12 / MR15), ranks present, round quality (status ok,
disconnect / bot, roster defects, relabelled), game shape (CT win rate, rounds per match, plant rate, kills per round, live seconds).
Part 2, the final model on its frozen test matches: log loss, AUC, ECE per group (export_winprob_v3_test_predictions.parquet).
Part 3, does training on them help? Same settings as the final model, stride --stride, trained on (a) all frozen-split training matches, (b) Valve matchmaking only, (c) FACEIT only;
every model is scored on the Valve test matches and on the FACEIT test matches, paired against (a).
Writes data/combined/faceit_audit.json.
"""
import os

os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
import argparse
import glob
import json
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl

import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")


def groups_table(store):
    d = os.path.join(W.ROOT, "data", store)
    M = pl.read_parquet(os.path.join(d, "matches.parquet"))
    pages = sorted(glob.glob(os.path.join(W.ROOT, "tomes", store, "tome", "csds", "header", "dataframe_*")))
    H = pl.concat([pl.read_parquet(f, columns=["key", "server_name", "platform"]) for f in pages], how="vertical_relaxed").rename({"key": "match_key"}).unique("match_key")
    M = M.join(H, on="match_key", how="left").with_columns(
        pl.when(pl.col("server_name").str.starts_with("FACEIT")).then(pl.lit("FACEIT")).when(pl.col("server_name").str.starts_with("Valve")).then(pl.lit("Valve")).otherwise(pl.lit("other")).alias("source"))
    M = M.with_columns((pl.col("source") + " " + pl.col("match_type")).alias("group"))
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="combined")
    ap.add_argument("--stride", type=int, default=3)
    a = ap.parse_args()
    d = os.path.join(W.ROOT, "data", a.store)
    t0 = time.time()
    M = groups_table(a.store)
    M26 = M.filter(pl.col("era") == "2026")
    R = pl.read_parquet(os.path.join(d, "rounds.parquet")).filter(pl.col("era") == "2026").join(M26.select("match_key", "group", "source", "match_type", "in_scope"), on="match_key", how="inner")
    out = {"groups": {}}
    big = [g for g, n in M26.group_by("group").len().iter_rows() if n >= 50]
    for g in sorted(big):
        m = M26.filter(pl.col("group") == g)
        r = R.filter(pl.col("group") == g)
        ru = r.filter(pl.col("use"))
        usable = ru["match_key"].n_unique()
        rank = m.select(((pl.col("t_starters_avg_rank") + pl.col("ct_starters_avg_rank")) / 2).alias("r")).drop_nulls()["r"]
        mr = dict(m.group_by("max_rounds_final").len().iter_rows())
        out["groups"][g] = {
            "matches": m.height, "in_scope": int(m["in_scope"].sum()), "usable_matches": int(usable), "usable_rounds": ru.height, "rounds_per_usable_match": round(ru.height / max(usable, 1), 2),
            "max_rounds_final": {str(k): int(v) for k, v in mr.items()}, "maps": {k: int(v) for k, v in m.filter(pl.col("in_scope")).group_by("map_name").len().iter_rows()},
            "rank_present_share": round(rank.len() / m.height, 3), "rank_mean": round(float(rank.mean()), 2) if rank.len() else None,
            "channels_complete_share": round(float(m["channels_complete"].mean()), 4), "round_status_ok_share": round(float((r["round_status"] == "ok").mean()), 4),
            "disc_or_bot_share": round(float(r["disc_or_bot"].mean()), 4), "roster_defect_share_of_used": round(float(ru["missing_side"].is_not_null().mean()), 4) if ru.height else None,
            "relabelled_share_of_used": round(float(ru["final_differs_from_recorded"].fill_null(False).mean()), 4) if ru.height else None,
            "ct_win_rate": round(float((ru["final"] == 3).mean()), 4) if ru.height else None, "plant_rate": round(float(ru["plant_tick"].is_not_null().mean()), 4) if ru.height else None,
            "kills_per_round": round(float(ru["n_deaths"].mean()), 3) if ru.height else None,
            "live_seconds_mean": round(float(((ru["end_eff"] - ru["freeze"]) / 64).mean()), 1) if ru.height else None,
        }
        print(g, {k: v for k, v in out["groups"][g].items() if k not in ("maps", "max_rounds_final")}, flush=True)
    # ---- part 2: the final model per group on its frozen test matches
    P = pl.read_parquet(os.path.join(d, "export_winprob_v3_test_predictions.parquet"), columns=["match_key", "p", "label_ct_win"]).join(M26.select("match_key", "group"), on="match_key", how="left")
    Wr = pl.read_parquet(os.path.join(d, "grid.parquet"), columns=["match_key", "round", "tick", "w_round"]) if False else None
    out["final_model_by_group"] = {}
    sub = pl.read_parquet(os.path.join(d, "export_winprob_v3_test_predictions.parquet"), columns=["match_key", "round", "tick", "p", "label_ct_win"]).join(M26.select("match_key", "group"), on="match_key", how="left").with_columns(pl.lit(1.0).alias("w_round"))
    for g in sorted(big):
        s = sub.filter(pl.col("group") == g)
        if s.select("match_key").n_unique() < 30:
            continue
        r = W.evaluate(s, s["label_ct_win"].to_numpy().astype(np.int64), s["p"].to_numpy().astype(np.float64))
        out["final_model_by_group"][g] = {"n_matches": r["n_matches"], "n_rows": r["n_rows"], "logloss": r["logloss"], "logloss_ci95": r["logloss_ci95"], "auc": r["auc"], "ece10": r["ece10"], "ece10_ci95": r["ece10_ci95"]}
        print("final model", g, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out["final_model_by_group"][g].items() if k != "ece10_ci95"}, flush=True)
    W.dump_json(out, os.path.join(d, "faceit_audit.json"))  # parts 1 and 2 are saved before the refits start
    # ---- part 3: refits
    feats = json.load(open(os.path.join(W.ROOT, "models", "winprob_v3.meta.json"), encoding="utf-8"))["features"]
    need = list(dict.fromkeys(["match_key", "round", "tick", "second", "label_ct_win", "w_round"] + feats))  # only what the refits read
    lf = pl.scan_parquet(os.path.join(d, "cache_snapshot_2026.parquet")).select(need)
    kdf = lf.select("match_key").unique().collect().with_columns(pl.lit(1.0).alias("w_round"), pl.lit(0).alias("label_ct_win"), pl.lit("2026").alias("era"), pl.lit("").alias("match_date"))
    sp = W.make_split("random", kdf, seed=0)
    test_keys = sorted(sp["test"])
    # training / validation rows are read at the stride they are used at; the test matches in full (the full cache is 19 M rows, 5+ GB)
    G = lf.filter((pl.col("second") % a.stride == 0) | pl.col("match_key").is_in(test_keys)).collect()
    print(f"cache rows kept: {G.height:,}", flush=True)
    src = dict(M26.select("match_key", "source").iter_rows())
    def keep(keys, s):
        return {k for k in keys if src.get(k) == s}
    sets = {"all": (set(sp["train"]), set(sp["val"])), "valve_only": (keep(sp["train"], "Valve"), keep(sp["val"], "Valve")), "faceit_only": (keep(sp["train"], "FACEIT"), keep(sp["val"], "FACEIT"))}
    tests = {"Valve test matches": keep(sp["test"], "Valve"), "FACEIT test matches": keep(sp["test"], "FACEIT")}
    print({k: (len(v[0]), len(v[1])) for k, v in sets.items()}, {k: len(v) for k, v in tests.items()}, flush=True)
    models = {}
    for name, (trk, vak) in sets.items():
        t1 = time.time()
        _, Xtr, ytr = W.arrays(G, trk, feats, stride=a.stride)
        _, Xva, yva = W.arrays(G, vak, feats, stride=a.stride)
        m, best = W.fit_xgb(Xtr, ytr, Xva, yva, W.mono_for(feats), seed=0, device="cuda")
        models[name] = (m, best)
        print(f"fit {name}: {len(trk)} train matches, {len(ytr):,} seconds, best iteration {best} ({time.time() - t1:.0f}s)", flush=True)
    out["refits"] = {"stride": a.stride, "train_matches": {k: len(v[0]) for k, v in sets.items()}, "test_matches": {k: len(v) for k, v in tests.items()}, "scores": {}}
    for tname, tk in tests.items():
        sub, X, y = W.arrays(G, tk, feats)
        ref_p = W.predict(*models["all"], X)
        out["refits"]["scores"][tname] = {}
        for name, mb in models.items():
            p = ref_p if name == "all" else W.predict(*mb, X)
            r = W.evaluate(sub, y, p, ref=None if name == "all" else ref_p)
            row = {"logloss": r["logloss"], "logloss_ci95": r["logloss_ci95"], "auc": r["auc"], "ece10": r["ece10"]}
            if name != "all":
                row["delta_vs_all"], row["delta_vs_all_ci95"] = r["delta_vs_ref"], r["delta_vs_ref_ci95"]
            out["refits"]["scores"][tname][name] = row
            print(f"{tname}: {name}: log loss {r['logloss']:.4f}" + (f", change vs all-matches model {r['delta_vs_ref']:+.4f} {[round(x, 4) for x in r['delta_vs_ref_ci95']]}" if name != "all" else ""), flush=True)
    W.dump_json(out, os.path.join(d, "faceit_audit.json"))
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
