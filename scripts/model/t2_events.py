"""T2 (Meridian pin 9b450cac): do event counts and timers over recent seconds add to the distance XGBoost on validation?

Features per second, from event ticks up to and including the row's own tick (64 ticks per second), per side (CT, T):
  kills_by_<side>_last3, kills_by_<side>_last10   deaths of the OPPOSING side in the window (a T death is a kill by CT)
  blinds_by_<side>_last3, blinds_by_<side>_last10 flash blinds whose attacker was on <side>, in the window
  blindsec_by_<side>_last3, blindsec_by_<side>_last10  summed blind_duration of those blinds
  secs_since_kill_by_<side>  seconds since the last kill by <side>, capped at 60 (60 if none in the round)
Windows: 3 seconds = 192 ticks, 10 seconds = 640 ticks, both including the current tick.

Choices the pin leaves open (declared here, before any run):
  - side codes: 3 = CT and 2 = T, as in the round labels (final 3 = CT win).
  - attacker side (blinds) and victim side (kills) come from the death table: a player's side in a round is the side on their
    death record in that round. A blind whose attacker has no death record in that round has no side and is left out of the
    side-specific blind counts. The share left out is reported in the output.
  - "defuse-start flags": the grid already carries the defuse counts (st_defusing_ct / st_defusing_t) in the 57 inputs, so no
    separate defuse features are added.
  - smoke and molotov detonations are excluded, as the pin says.
Arms (validation, seeds 0, 1, 2): S = the saved SN arm of sitecheck_xgb.py (57 inputs plus train-centre distances); S plus the 14
event features. The round model is W.fit_xgb (early stopping on validation), the same recipe as every other arm.
Rule (standard, pin 9b450cac): PASS only if for EACH seed mean(S plus events minus S) on validation is at most -0.0005 with a 95%
match-clustered CI upper bound below 0, and the three-seed mean is at most -0.0005. Descriptive: the gain by round-second bucket
(0-24, 25-49, 50-74, 75-end) and the test (de_mirage) result.

Writes data/diagnostics/t2_events.json. Runs under CS2_ROOT (default: the local project; on the RunPod volume CS2_ROOT=/workspace/csds).
"""
import glob
import json
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wp_common as W  # noqa: E402
from spatial_feature_eval import paired  # noqa: E402

ROOT = W.ROOT
DIAG = os.path.join(ROOT, "data", "diagnostics")
PREDS = os.path.join(DIAG, "preds")
DEATHS = os.path.join(ROOT, "data", "combined", "deaths.parquet")
BLIND_DIR = os.path.join(ROOT, "data", "blind_events")
OUT = os.path.join(DIAG, "t2_events.json")
MAPS = ["de_mirage", "de_dust2", "de_ancient", "de_anubis", "de_inferno", "de_nuke", "de_cache"]
KEY = ["match_key", "round", "second"]
ST = [f"st{i}" for i in range(4)]
NST = [f"nst{i}" for i in range(4)]
SEEDS = [0, 1, 2]
CT, T = 3, 2
TICK_STRIDE = 2 ** 32
W3, W10 = 192, 640
SECS_CAP = 60.0
EVENT_COLS = [f"{kind}_by_{side}_{w}" for side in ("CT", "T") for kind, w in
              [("kills", "last3"), ("kills", "last10"), ("blinds", "last3"), ("blinds", "last10"),
               ("blindsec", "last3"), ("blindsec", "last10")]] + ["secs_since_kill_by_CT", "secs_since_kill_by_T"]


def window_count(keys, probe, width):
    """Events with key in (probe - width, probe]; keys sorted."""
    return np.searchsorted(keys, probe, side="right") - np.searchsorted(keys, probe - width, side="right")


def window_sum(keys, cs, probe, width):
    """Sum of per-event values (cs = prefix sums with a leading 0) over (probe - width, probe]."""
    return cs[np.searchsorted(keys, probe, side="right")] - cs[np.searchsorted(keys, probe - width, side="right")]


def secs_since(keys, probe, gid_start):
    """Seconds since the last event at or before probe in the same round; SECS_CAP if none."""
    idx = np.searchsorted(keys, probe, side="right") - 1
    ok = idx >= 0
    last = keys[np.where(ok, idx, 0)]
    ok &= last >= gid_start
    return np.where(ok, np.minimum((probe - last) / 64.0, SECS_CAP), SECS_CAP).astype(np.float32)


def sorted_join(base, frame, label_col):
    J = base.join(frame.sort(KEY).rename({label_col: "label_ref"}), on=KEY, how="inner", maintain_order="left")
    assert J.height == base.height, f"{J.height:,} matched rows, expected {base.height:,}"
    assert (J["label_ref"].to_numpy().astype(np.int64) == base["label"].to_numpy().astype(np.int64)).all(), "labels do not align"
    return J


def main():
    t0 = time.time()
    feats = json.load(open(os.path.join(ROOT, "models", "winprob_v6_drop_heavy.meta.json"), encoding="utf-8"))["features"]
    assert len(feats) == 57, len(feats)
    S_cols = feats + NST
    cols = S_cols + EVENT_COLS

    # population: the same narrow-first build as t1_engagement_stack.py (sitecheck_xgb.py population)
    geo_old = pl.concat([pl.read_parquet(os.path.join(DIAG, "allmaps_geo", f"{m}.parquet")) for m in MAPS])
    geo_new = pl.concat([pl.read_parquet(os.path.join(DIAG, "allmaps_geo_trainsites", f"{m}.parquet")) for m in MAPS])
    geo_new = geo_new.rename({f"st{i}": f"nst{i}" for i in range(4)}).unique(KEY, keep="first", maintain_order=True)
    C = pl.read_parquet(os.path.join(ROOT, "data", "combined", "cache_snapshot_2026_ext.parquet"),
                        columns=sorted(set(KEY + ["match_key", "tick", "label_ct_win", "map_name"] + feats)))
    C = W.drop_roster_defect_rounds(C)
    C = C.filter(pl.col("label_ct_win").is_not_null()).with_columns(pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int64))
    man = W.load_manifest(os.path.join(ROOT, "data", "combined", "split_manifest.json"))["assign"]
    A = pl.DataFrame({"match_key": list(man.keys()), "split": list(man.values())})
    pop = (C.select(KEY + ["map_name", "label_ct_win"])
             .join(geo_old.select(KEY).unique(KEY), on=KEY, how="semi")
             .join(geo_new.select(KEY + NST), on=KEY, how="inner")
             .join(A, on="match_key", how="inner"))
    pop = pop.unique(KEY, keep="first", maintain_order=True).sort(KEY)
    G = pop.join(C.select(KEY + ["tick"] + feats), on=KEY, how="left", maintain_order="left")
    del C, geo_old, geo_new, pop
    print(f"population rows {G.height:,} ({time.time() - t0:.0f} s)", flush=True)

    # events
    deaths = pl.read_parquet(DEATHS, columns=["match_key", "round", "tick", "player_id", "side"]).with_columns(pl.col("round").cast(pl.Int64))
    blinds = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(os.path.join(BLIND_DIR, "*.parquet")))]).with_columns(pl.col("round").cast(pl.Int64))
    side_map = deaths.select(["match_key", "round", "player_id", "side"]).unique(["match_key", "round", "player_id"], keep="first")
    n_blinds = blinds.height
    blinds = blinds.join(side_map.rename({"player_id": "attacker_id", "side": "attacker_side"}),
                         on=["match_key", "round", "attacker_id"], how="left", maintain_order="left")
    known = blinds.filter(pl.col("attacker_side").is_not_null())
    coverage = {"blind_events": n_blinds, "attacker_side_known": known.height, "share_known": known.height / max(n_blinds, 1)}
    print(f"blind events {n_blinds:,}; attacker side known for {known.height:,} ({coverage['share_known']:.3f})", flush=True)

    # round index over every (match, round) that appears in the population, the deaths or the blinds
    gk = (pl.concat([G.select(["match_key", "round"]), deaths.select(["match_key", "round"]), blinds.select(["match_key", "round"])])
            .unique().sort(["match_key", "round"]).with_row_index("gid"))

    def keys_of(frame, sel=None):
        f = frame if sel is None else frame.filter(sel)
        return (f.join(gk, on=["match_key", "round"], how="inner")
                 .select((pl.col("gid").cast(pl.Int64) * TICK_STRIDE + pl.col("tick")).alias("key"), pl.col("gid"))
                 .sort("key"))

    kd_CT = keys_of(deaths, pl.col("side") == T)     # a T death: a kill by CT
    kd_T = keys_of(deaths, pl.col("side") == CT)     # a CT death: a kill by T
    kb = {}
    for side, code in (("CT", CT), ("T", T)):
        f = known.filter(pl.col("attacker_side") == code).join(gk, on=["match_key", "round"], how="inner")
        f = f.select((pl.col("gid").cast(pl.Int64) * TICK_STRIDE + pl.col("tick")).alias("key"), "blind_duration").sort("key")
        kb[side] = (f["key"].to_numpy(), np.concatenate([[0.0], np.cumsum(f["blind_duration"].to_numpy().astype(np.float64))]))
    # kills scored by CT are the T deaths (kd_CT is built from side == T); kills scored by T are the CT deaths (kd_T)
    kd_scored = {"CT": kd_CT["key"].to_numpy(), "T": kd_T["key"].to_numpy()}
    print(f"event keys ready ({time.time() - t0:.0f} s)", flush=True)

    rows = G.select(["match_key", "round", "tick"]).join(gk, on=["match_key", "round"], how="left", maintain_order="left")
    gid = rows["gid"].to_numpy().astype(np.int64)
    if (gid < 0).any():
        raise SystemExit("a population row has no round index: the round key build is wrong")
    probe = gid * TICK_STRIDE + rows["tick"].to_numpy().astype(np.int64)
    start = gid * TICK_STRIDE
    del rows, gk

    ev = np.empty((G.height, len(EVENT_COLS)), dtype=np.float32)
    j = 0
    for side in ("CT", "T"):
        keys = kd_scored[side]
        ev[:, j] = window_count(keys, probe, W3); ev[:, j + 1] = window_count(keys, probe, W10); j += 2
        bk, cs = kb[side]
        ev[:, j] = window_count(bk, probe, W3); ev[:, j + 1] = window_count(bk, probe, W10); j += 2
        ev[:, j] = window_sum(bk, cs, probe, W3); ev[:, j + 1] = window_sum(bk, cs, probe, W10); j += 2
    ev[:, j] = secs_since(kd_scored["CT"], probe, start); ev[:, j + 1] = secs_since(kd_scored["T"], probe, start)
    print(f"event features built ({time.time() - t0:.0f} s)", flush=True)

    X = np.empty((G.height, len(cols)), dtype=np.float32)
    for jj, c in enumerate(S_cols):
        X[:, jj] = G[c].to_numpy().astype(np.float32)
    X[:, len(S_cols):] = ev
    del ev

    split = G["split"]
    is_train = (split == "train").to_numpy()
    is_val = (split == "val").to_numpy()
    is_test = ((split == "test") & (G["map_name"] == "de_mirage")).to_numpy()
    tr_idx, va_idx, te_idx = np.flatnonzero(is_train), np.flatnonzero(is_val), np.flatnonzero(is_test)
    y_all = G["label_ct_win"].to_numpy().astype(np.int64)
    base_va = G.filter(pl.col("split") == "val").select(KEY + ["label_ct_win"]).rename({"label_ct_win": "label"})
    base_te = (G.filter((pl.col("split") == "test") & (pl.col("map_name") == "de_mirage"))
                 .select(KEY + ["label_ct_win"]).rename({"label_ct_win": "label"}))
    del G
    y_va = base_va["label"].to_numpy().astype(np.int64)
    y_te = base_te["label"].to_numpy().astype(np.int64)
    sec_va = base_va["second"].to_numpy()
    print(f"validation rows {va_idx.size:,}, train rows {tr_idx.size:,}, test rows {te_idx.size:,} ({time.time() - t0:.0f} s)", flush=True)

    res = {"per_seed": {}, "coverage": coverage, "event_columns": EVENT_COLS,
           "rule": "per seed mean(S plus events minus S) on validation <= -0.0005 with CI upper < 0; three-seed mean <= -0.0005"}
    per_seed_pass, diffs = [], []
    for s in SEEDS:
        # S is refit here with the same recipe and seed, because the pod's xgboost (3.2.0) is not the 3.4.2 that made the saved
        # SN predictions. S and S plus events must come from the same library. The saved SN predictions are kept as a sanity check.
        nS = len(S_cols)
        mS, bestS = W.fit_xgb(X[tr_idx, :nS], y_all[tr_idx], X[va_idx, :nS], y_all[va_idx], mono=W.mono_for(S_cols), seed=s, device="cuda")
        pS_va = W.predict(mS, bestS, X[va_idx, :nS])
        pS_te = W.predict(mS, bestS, X[te_idx, :nS])
        del mS
        m, best = W.fit_xgb(X[tr_idx], y_all[tr_idx], X[va_idx], y_all[va_idx], mono=W.mono_for(cols), seed=s, device="cuda")
        p_va = W.predict(m, best, X[va_idx])
        p_te = W.predict(m, best, X[te_idx])
        print(f"seed {s}: S refit best iteration {bestS}; S plus events best iteration {best} ({time.time() - t0:.0f} s)", flush=True)
        sn = pl.read_parquet(os.path.join(PREDS, f"allmaps_xgb_SN_seed{s}.parquet"))
        J_va = sorted_join(base_va, sn.filter(pl.col("split") == "val").select(KEY + ["label", "p"]), "label")
        p_sn_va = J_va["p"].to_numpy()
        match_va = base_va["match_key"].to_numpy()
        d_va = paired(match_va, y_va, p_va, pS_va)
        d_te = paired(base_te["match_key"].to_numpy(), y_te, p_te, pS_te)
        d_refit_vs_saved = paired(match_va, y_va, pS_va, p_sn_va)
        print(f"seed {s}: S refit minus saved SN (sanity, same data, library differs): {d_refit_vs_saved['diff']:+.6f}", flush=True)
        buckets = {}
        for name, (lo, hi) in {"0-24": (0, 24), "25-49": (25, 49), "50-74": (50, 74), "75-end": (75, 10 ** 6)}.items():
            sub = np.flatnonzero((sec_va >= lo) & (sec_va <= hi))
            buckets[name] = paired(match_va[sub], y_va[sub], p_va[sub], pS_va[sub])
        ok = bool(d_va["diff"] <= -0.0005 and d_va["ci95"][1] < 0)
        per_seed_pass.append(ok)
        diffs.append(d_va["diff"])
        pl.DataFrame({"match_key": match_va, "round": base_va["round"].to_numpy(), "second": sec_va, "split": "val",
                      "label": y_va, "p": p_va.astype(np.float64)}).write_parquet(os.path.join(PREDS, f"t2_se_seed{s}_val.parquet"))
        res["per_seed"][s] = {"S_plus_events_minus_S_validation": d_va, "S_plus_events_minus_S_test_descriptive": d_te,
                              "S_refit_minus_saved_SN_validation": d_refit_vs_saved,
                              "by_round_second_bucket_validation": buckets, "pass": ok,
                              "best_iteration_S_plus_events": int(best), "best_iteration_S_refit": int(bestS)}
        print(f"seed {s}: validation S plus events minus S {d_va['diff']:+.6f} CI [{d_va['ci95'][0]:+.6f}, {d_va['ci95'][1]:+.6f}], pass {ok}", flush=True)
        W.dump_json({"seeds_done": list(res["per_seed"]), "per_seed": res["per_seed"]}, OUT + ".partial")
        del m

    mean = float(np.mean(diffs))
    passes = bool(all(per_seed_pass) and mean <= -0.0005)
    res.update({"mean_over_seeds_validation": mean, "per_seed_pass": per_seed_pass, "passes": passes, "seconds": round(time.time() - t0)})
    W.dump_json(res, OUT)
    print(json.dumps({"mean_over_seeds_validation": mean, "passes": passes}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
