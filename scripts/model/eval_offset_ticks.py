"""Does the model work at EVERY tick, not only at the whole seconds it was trained on?

The training grid samples ticks freeze_end + 64 k. Here the held-out matches are re-sampled at a shifted tick phase (a fixed
half-second shift, 32 ticks, and a random phase per round), the living-player table is read from the raw player_status at those
exact ticks, and the exported model is scored on them: log loss with a match-cluster interval, next to the same model on the
training-phase ticks of the same matches. Also pushes a sample through the serving class (winprob.WinProb.predict_many with
player tables), i.e. the code path a live service runs.

  1) python make_grid.py --store combined --offset 32          (and --offset -1 for a random phase per round)
  2) python eval_offset_ticks.py --write-matches --name winprob_v1 --eras 2026      -> data/combined/test_matches_winprob_v1.txt
  3) python extract_status_1hz.py --store combined --grid grid_offset32.parquet --out-dir data\\status_1hz_offset32 --match-list ...
  4) python eval_offset_ticks.py --name winprob_v1 --eras 2026 --offsets 32,random
"""
import argparse
import json
import os
import sys

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl

import features_status as FS
import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")
ROOT = W.ROOT
MAPS = ["ancient", "anubis", "cache", "dust2", "inferno", "mirage", "nuke"]


def test_matches(store, eras, split="random", seed=0):
    G = W.load_grid(store, cols=None)
    if eras:
        G = G.filter(pl.col("era").is_in(eras.split(",")))
    return sorted(W.make_split(split, G, seed=seed)["test"])


def with_features(G, feats, status_dir):
    G = G.with_columns([(pl.col("map_name") == f"de_{m}").cast(pl.Int8).alias(f"map_is_{m}") for m in MAPS])
    if any(f.startswith("st_") for f in feats):
        G = G.with_columns(FS.loadout_features(G, status_dir))
    return G


def predict(booster, best, G, feats):
    import xgboost as xgb
    X = G.select(feats).to_numpy().astype(np.float32)
    return booster.predict(xgb.DMatrix(X), iteration_range=(0, best + 1))


def summarize(sub, y, p):
    r = W.evaluate(sub, y, p)
    return {"rows": r["n_rows"], "matches": r["n_matches"], "logloss": r["logloss"], "logloss_ci95": r["logloss_ci95"],
            "logloss_round_weighted": r["logloss_round_weighted"], "auc": r["auc"], "ece10": r["ece10"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="combined")
    ap.add_argument("--name", default="winprob_v1")
    ap.add_argument("--eras", default="2026")
    ap.add_argument("--offsets", default="32,random")
    ap.add_argument("--write-matches", action="store_true")
    ap.add_argument("--serve-sample", type=int, default=20000)
    args = ap.parse_args()
    mlist = os.path.join(ROOT, "data", args.store, f"test_matches_{args.name}.txt")
    if args.write_matches:
        tm = test_matches(args.store, args.eras)
        open(mlist, "w", encoding="utf-8").write("\n".join(tm) + "\n")
        print(f"{len(tm)} test matches -> {mlist}")
        return
    import xgboost as xgb
    tm = [ln.strip() for ln in open(mlist, encoding="utf-8") if ln.strip()]
    meta = json.load(open(os.path.join(ROOT, "models", args.name + ".meta.json"), encoding="utf-8"))
    feats, best = meta["features"], meta["best_iteration"]
    b = xgb.Booster()
    b.set_param({"device": "cpu"})
    b.load_model(os.path.join(ROOT, "models", args.name + ".json"))
    d = os.path.join(ROOT, "data", args.store)
    M = pl.read_parquet(os.path.join(d, "matches.parquet")).select("match_key", "match_date")
    out = {"model": args.name, "test_matches": len(tm)}

    def load(gridfile, status_dir):
        G = pl.read_parquet(os.path.join(d, gridfile)).filter(pl.col("match_key").is_in(tm)).join(M, on="match_key", how="left")
        if any(f.startswith("st_") for f in feats):
            G = W.drop_roster_defect_rounds(G, args.store)  # same rounds the model was fitted and exported on
            G = FS.drop_inconsistent_rounds(G, status_dir)
        # only matches whose snapshots were extracted can be scored
        return with_features(G, feats, status_dir)

    base = load("grid.parquet", os.path.join(ROOT, "data", "status_1hz"))
    base = base.filter(pl.col("st_hp_ct").is_not_null() & ~pl.col("st_hp_ct").is_nan()) if "st_hp_ct" in base.columns else base
    y = base["label_ct_win"].to_numpy().astype(np.float64)
    out["training_phase_ticks"] = summarize(base, y, predict(b, best, base, feats))
    print("training-phase ticks :", out["training_phase_ticks"]["rows"], "rows", round(out["training_phase_ticks"]["logloss"], 4), out["training_phase_ticks"]["logloss_ci95"])
    for off in args.offsets.split(","):
        gname = "grid_offset_random.parquet" if off == "random" else f"grid_offset{off}.parquet"
        sdir = os.path.join(ROOT, "data", f"status_1hz_offset{off}")
        G = load(gname, sdir)
        G = G.filter(pl.col("st_hp_ct").is_not_null() & ~pl.col("st_hp_ct").is_nan()) if "st_hp_ct" in G.columns else G
        yy = G["label_ct_win"].to_numpy().astype(np.float64)
        p = predict(b, best, G, feats)
        out[f"offset_{off}"] = summarize(G, yy, p)
        frac = (G["seconds_since_round_live"].to_numpy() % 1.0)
        out[f"offset_{off}"]["fraction_of_second_mean"] = float(frac.mean())
        print(f"phase {off:>6}: {out[f'offset_{off}']['rows']} rows  LL {out[f'offset_{off}']['logloss']:.4f} {out[f'offset_{off}']['logloss_ci95']}")
        # the serving path on a sample: player tables from the raw snapshots at these ticks, WinProb.predict_many
        if args.serve_sample and any(f.startswith("st_") for f in feats):
            import glob

            from winprob import WinProb
            fb = os.path.join(ROOT, "models", args.name.replace("winprob_v1", "winprob_v1_basic") + ".json")
            wp = WinProb.load(os.path.join(ROOT, "models", args.name + ".json"), fb if os.path.exists(fb) else None)
            samp = G.sample(n=min(args.serve_sample, G.height), seed=3)
            keys = samp.select("match_key", "round", "tick")
            S = pl.scan_parquet(sorted(glob.glob(os.path.join(sdir, "*.parquet")))).join(keys.lazy(), on=["match_key", "round", "tick"], how="semi").collect()
            players = {}
            for r in S.iter_rows(named=True):
                players.setdefault((r["match_key"], r["round"], r["tick"]), []).append(
                    {"side": r["side"], "health": r["health"], "armor": r["armor"], "has_helmet": r["has_helmet"], "has_defuser": r["has_defuser"],
                     "is_defusing": r["is_defusing"], "primary": r["inv_primary"], "flashbang": r["inv_flashbang"], "smokegrenade": r["inv_smokegrenade"],
                     "hegrenade": r["inv_hegrenade"], "molotov": r["inv_molotov"], "incgrenade": r["inv_incgrenade"], "decoy": r["inv_decoy"],
                     "equipment_value": r["equipment_value_calc"]})
            states = [{"seconds_since_round_live": r["seconds_since_round_live"], "bomb_planted": r["bomb_planted"], "seconds_since_plant": r["seconds_since_plant"],
                       "map_name": r["map_name"], "mode": "competitive", "players": players.get((r["match_key"], r["round"], r["tick"]))} for r in samp.iter_rows(named=True)]
            res = wp.predict_many(states)
            ps = np.array([o["p_ct_win"] for o in res])
            ref = predict(b, best, samp, feats)
            ok = np.array([o["status"] == "ok" for o in res])
            out[f"offset_{off}"]["serving_path"] = {"rows": int(len(ps)), "status_ok": int(ok.sum()),
                                                     "max_abs_diff_vs_batch": float(np.max(np.abs(ps[ok] - ref[ok]))) if ok.any() else None,
                                                     "logloss": float(np.mean(W.ll_vec(ps, samp["label_ct_win"].to_numpy())))}
            print("   serving path:", out[f"offset_{off}"]["serving_path"])
    path = os.path.join(d, f"eval_offset_ticks_{args.name}.json")
    json.dump(out, open(path, "w", encoding="utf-8"), indent=1)
    print("->", path)


if __name__ == "__main__":
    main()
