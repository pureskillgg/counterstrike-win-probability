"""Event-derived features for the best measured set (58 inputs: 59 minus heavy, plus score lead), 2026 matches, every second.

Causal by construction: for a grid row at tick t, each feature is the running total over the rounds' events with tick <= t, within the same
round only (reset at each round start). Per side (CT = 3, T = 2, the team codes the status features use):
  ev_dmg_ct / ev_dmg_t / ev_dmg_diff      health damage dealt by each side so far in the round (player_hurt.effective_health_removed,
                                          by attacker_team_code)
  ev_shots_ct / ev_shots_t / ev_shots_diff shots fired by each side so far in the round (weapon_fire rows, by player_team_code)
The features are joined to the grid rows with an as-of join on tick (last event at or before the row), nulls as 0 (no event yet).

Two runs: the 64-input model (58 + 6 event features) and the 58-input baseline on the same seconds, seed 0, same settings and frozen split.
The features are saved to data/combined/event_feats_2026.parquet for later use.

  ...python.exe -I -X utf8 scripts\\model\\exp_event_features.py
Writes data/combined/exp_event_features.json.
"""
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import polars as pl

import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")
CT, T = 3, 2
HEAVY = ["st_heavy_ct", "st_heavy_t"]
EV = ["ev_dmg_ct", "ev_dmg_t", "ev_dmg_diff", "ev_shots_ct", "ev_shots_t", "ev_shots_diff"]
BASE = os.path.join(W.ROOT, "tomes", "combined", "tome", "csds")


def cumulative(channel, cols, team_col, value_expr, mks, rounds_key):
    files = sorted(glob.glob(os.path.join(BASE, channel + ".COMBINED", "dataframe_*")))
    df = pl.concat([pl.read_parquet(f, columns=cols).filter(pl.col("match_key").is_in(mks)) for f in files])
    df = (df.with_columns(pl.col(team_col).alias("team"), value_expr.alias("v"))
          .filter(pl.col("team").is_in([CT, T]))
          .with_columns(pl.when(pl.col("team") == CT).then(pl.col("v")).otherwise(0).alias("ct_v"),
                        pl.when(pl.col("team") == T).then(pl.col("v")).otherwise(0).alias("t_v"))
          .sort(["match_key", "round", "tick"])
          .with_columns(pl.col("ct_v").cum_sum().over(["match_key", "round"]).alias("ct_cum"),
                        pl.col("t_v").cum_sum().over(["match_key", "round"]).alias("t_cum"))
          .select(["match_key", "round", "tick", "ct_cum", "t_cum"]).sort("tick"))
    print(f"{channel}: {df.height:,} events in 2026 matches with a CT or T team", flush=True)
    return df


def main():
    t0 = time.time()
    d = os.path.join(W.ROOT, "data", "combined")
    ref_meta = json.load(open(os.path.join(W.ROOT, "models", "winprob_v4.meta.json"), encoding="utf-8"))
    feats59 = json.load(open(os.path.join(W.ROOT, "models", "winprob_v3.meta.json"), encoding="utf-8"))["features"]
    mono59 = {f: int(ref_meta["monotone_constraints"][f]) for f in feats59}
    f58 = [f for f in feats59 if f not in HEAVY] + ["score_lead_diag"]
    need = list(dict.fromkeys(["match_key", "round", "tick", "second", "label_ct_win", "w_round"] + feats59))
    G = pl.scan_parquet(os.path.join(d, "cache_snapshot_2026.parquet")).select(need).with_columns(pl.col(feats59).cast(pl.Float32)).collect()
    lead = (pl.scan_parquet(os.path.join(d, "grid.parquet")).filter(pl.col("era") == "2026").select(["match_key", "round", "score_lead_diag"])
            .unique(subset=["match_key", "round"]).collect())
    G = G.join(lead, on=["match_key", "round"], how="left").with_columns(pl.col("score_lead_diag").fill_null(0).cast(pl.Float32))
    mks = set(G["match_key"].unique().to_list())
    G = G.sort("tick")

    dmg = cumulative("basic_player_hurt", ["match_key", "round", "tick", "attacker_team_code", "effective_health_removed"],
                     "attacker_team_code", pl.col("effective_health_removed").fill_null(0), mks, "round")
    shots = cumulative("basic_weapon_fire", ["match_key", "round", "tick", "player_team_code"], "player_team_code",
                       pl.lit(1), mks, "round")
    G = G.join_asof(dmg.rename({"ct_cum": "ev_dmg_ct", "t_cum": "ev_dmg_t"}), on="tick", by=["match_key", "round"], strategy="backward")
    G = G.join_asof(shots.rename({"ct_cum": "ev_shots_ct", "t_cum": "ev_shots_t"}), on="tick", by=["match_key", "round"], strategy="backward")
    G = G.with_columns([pl.col(c).fill_null(0).cast(pl.Float32) for c in ["ev_dmg_ct", "ev_dmg_t", "ev_shots_ct", "ev_shots_t"]])
    G = G.with_columns([(pl.col("ev_dmg_ct") - pl.col("ev_dmg_t")).alias("ev_dmg_diff"),
                        (pl.col("ev_shots_ct") - pl.col("ev_shots_t")).alias("ev_shots_diff")])
    print(f"event features joined ({time.time() - t0:.0f}s); share of seconds with any damage so far {(G['ev_dmg_ct'] + G['ev_dmg_t'] > 0).mean():.3f}",
          flush=True)
    G.select(["match_key", "round", "tick"] + EV).write_parquet(os.path.join(d, "event_feats_2026.parquet"))

    kdf = G.select("match_key").unique().with_columns(pl.lit(1.0).alias("w_round"), pl.lit(0).alias("label_ct_win"),
                                                      pl.lit("2026").alias("era"), pl.lit("").alias("match_date"))
    sp = W.make_split("random", kdf, seed=0)
    tr, va, te = sorted(sp["train"]), sorted(sp["val"]), sorted(sp["test"])
    del kdf

    def fit_eval(name, feats, ctrl=None):
        mono = [mono59.get(f, 0) for f in feats]
        _, Xtr, ytr = W.arrays(G, tr, feats, stride=1)
        sub_va, Xva, yva = W.arrays(G, va, feats, stride=1)
        sub_te, Xte, yte = W.arrays(G, te, feats, stride=1)
        m, best = W.fit_xgb(Xtr, ytr, Xva, yva, mono, seed=0, device="cuda")
        del Xtr, ytr
        pte, pva = W.predict(m, best, Xte), W.predict(m, best, Xva)
        if ctrl is None:
            ev, evv = W.evaluate(sub_te, yte, pte), W.evaluate(sub_va, yva, pva)
        else:
            ev, evv = W.evaluate(sub_te, yte, pte, ref=ctrl["pte"]), W.evaluate(sub_va, yva, pva, ref=ctrl["pva"])
        res = {"name": name, "n_features": len(feats), "best_iteration": int(best),
               "validation_logloss": evv["logloss"], "test_logloss": ev["logloss"]}
        if ctrl is not None:
            res.update({"validation_delta_vs_baseline": evv["delta_vs_ref"], "test_delta_vs_baseline": ev["delta_vs_ref"],
                        "test_delta_ci95": ev["delta_vs_ref_ci95"]})
        return res, {"pte": pte, "pva": pva}

    out = {"variants": []}
    base, cp = fit_eval("baseline: 58 inputs", f58)
    out["variants"].append(base)
    print(f"baseline 58: validation {base['validation_logloss']:.6f}; test {base['test_logloss']:.6f} ({time.time() - t0:.0f}s)", flush=True)
    W.dump_json(out, os.path.join(d, "exp_event_features.json"))
    r, _ = fit_eval("58 + event features (64)", f58 + EV, cp)
    out["variants"].append(r)
    lo, hi = r["test_delta_ci95"]
    print(f"58 + event features: validation {r['validation_logloss']:.6f} (delta {r['validation_delta_vs_baseline']:+.5f}); "
          f"test {r['test_logloss']:.6f} (delta {r['test_delta_vs_baseline']:+.5f} [{lo:+.5f}, {hi:+.5f}]) ({time.time() - t0:.0f}s)", flush=True)
    W.dump_json(out, os.path.join(d, "exp_event_features.json"))
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
