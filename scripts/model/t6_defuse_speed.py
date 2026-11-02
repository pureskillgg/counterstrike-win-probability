"""T6 (Meridian pin da91d46d, amended twice; clarification pinned 23053c6b before any run): does a per-weapon run speed improve the defuse margin over the constant 250 units per second? (post-plant validation)

Rows: every 10th second of the frozen split, de_mirage 2026 (defuse_race_eval.base_rows), all rows used for training as in
defuse_race_eval.py (the margin is missing outside post-plant seconds, so XGBoost sends them down its default branch).

Margin (both arms): fuse_left - (distance to the nearest living CT / speed + DEFUSE), fuse_left = 40 - seconds_since_plant,
DEFUSE = 10 s, exact plant position (positions_1hz CT rows and _plants, as in defuse_race_feature.py). The nearest CT is the
same player in both arms.
  CONST  speed = 250 units per second (the DX margin of defuse_race_eval2.py)
  PW     speed = that CT's table value: the primary weapon (inv_primary from status_1hz), or the secondary when the primary is 0

Arms: 57 inputs plus one margin column. Decision: post-plant seconds of the validation split. Rule (pin da91d46d): pass if for EACH
seed mean(PW minus CONST) <= -0.0005 with a 95% match-clustered CI upper bound below 0, and the three-seed mean <= -0.0005.
Descriptive: the same comparison on post-plant test seconds (de_mirage). Any inventory code outside the table stops the run.

Writes data/diagnostics/t6_defuse_speed.json. Run in the cs2gpu environment:
  C:/Users/<user>/.pixi/workspace-envs/cs2gpu-15282324412032665829/envs/default/python.exe -I -X utf8 -u scripts/model/t6_defuse_speed.py
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
import defuse_race_eval as D  # noqa: E402

ROOT = W.ROOT
DIAG = os.path.join(ROOT, "data", "diagnostics")
OUT = os.path.join(DIAG, "t6_defuse_speed.json")
KEY = ["match_key", "round", "second"]
SEEDS = [0, 1, 2]
CT = 3
FUSE = D.FUSE
DEFUSE = D.DEFUSE
SPEED_CONST = D.SPEED  # 250 units per second

# tradeit.gg "cs2-weapon-stats" maximum movement speed (units per second), keyed by inventory code (pin da91d46d)
PRIMARY = {7: 215, 60: 225, 16: 225, 9: 200, 13: 215, 40: 230, 33: 220, 34: 240, 17: 240, 10: 220, 25: 215, 23: 235, 39: 210,
           8: 220, 19: 230, 35: 225, 27: 225, 28: 150, 24: 230, 26: 240, 29: 210, 38: 215, 11: 215, 14: 195}
SECONDARY = {4: 240, 61: 240, 1: 230, 32: 240, 30: 240, 3: 240, 36: 240, 2: 240, 63: 240, 64: 180}
KNIFE = 250  # knife value in pin da91d46d; used when both inventory codes are 0 (clarification pinned before any T6 outcome)


def nearest_ct(keys):
    """One row per planted key: the nearest living CT's distance to the exact plant, that player's id and inventory codes.
    Positions, plant file and status are read per day, as in defuse_race_feature.py. Keys with no CT position get no row."""
    plants = D.plant_sites().select(["match_key", "round", "bx", "by"]).with_columns(pl.col("round").cast(pl.Int64))
    keys = keys.select(KEY).unique()
    days = sorted(os.path.basename(f)[:-len(".parquet")] for f in glob.glob(os.path.join(ROOT, "data", "positions_1hz", "2026-*.parquet")))
    out = []
    for day in days:
        pos_f = os.path.join(ROOT, "data", "positions_1hz", day + ".parquet")
        st_f = os.path.join(ROOT, "data", "status_1hz", day + ".parquet")
        if not os.path.exists(st_f):
            print(f"no status file for {day}; its keys are left without a speed", flush=True)
            continue
        pos = (pl.read_parquet(pos_f, columns=["match_key", "round", "second", "player_id", "side", "x", "y"])
                 .filter(pl.col("side") == CT)
                 .with_columns(pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int64))
                 .filter(pl.col("x").is_not_null() & pl.col("y").is_not_null()))
        j = pos.join(keys, on=KEY, how="inner").join(plants, on=["match_key", "round"], how="inner")
        if j.height == 0:
            continue
        j = j.with_columns(((pl.col("x") - pl.col("bx")) ** 2 + (pl.col("y") - pl.col("by")) ** 2).sqrt().alias("ct_dist"))
        best = j.sort("ct_dist").unique(KEY, keep="first", maintain_order=True)
        st = (pl.read_parquet(st_f, columns=["match_key", "round", "second", "player_id", "inv_primary", "inv_secondary"])
                .with_columns(pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int64)))
        best = best.join(st, on=KEY + ["player_id"], how="left")
        out.append(best.select(KEY + ["ct_dist", "player_id", "inv_primary", "inv_secondary"]))
    return pl.concat(out)


def speeds(N):
    """Per-row speed from the table; any code outside the table, or a missing inventory row, stops the run."""
    vals, bad, secondary_used, knife_used = [], [], 0, 0
    for p, s in zip(N["inv_primary"].to_list(), N["inv_secondary"].to_list()):
        if p is None or s is None:
            bad.append((p, s))
            vals.append(None)
            continue
        if p == 0 and s == 0:
            # no gun in either slot: the knife value from pin da91d46d (clarification pinned before any T6 outcome was read)
            knife_used += 1
            vals.append(float(KNIFE))
            continue
        if p == 0:
            secondary_used += 1
            table, code = SECONDARY, s
        else:
            table, code = PRIMARY, p
        v = table.get(int(code))
        if v is None:
            bad.append((p, s))
        vals.append(None if v is None else float(v))
    if bad:
        seen = sorted({str(b) for b in bad})[:20]
        raise SystemExit(f"{len(bad)} rows have an inventory code outside the table or no status row; first codes: {seen}")
    return vals, secondary_used, knife_used


def main():
    t0 = time.time()
    feats = json.load(open(os.path.join(ROOT, "models", "winprob_v6_drop_heavy.meta.json"), encoding="utf-8"))["features"]
    assert len(feats) == 57, len(feats)
    G = D.base_rows(feats)
    G = G.join(D.plant_sites().select(["match_key", "round", "site"]).with_columns(pl.col("round").cast(pl.Int64)),
               on=["match_key", "round"], how="left", maintain_order="left")
    planted_keys = G.filter((pl.col("bomb_planted") == 1) & pl.col("seconds_since_plant").is_not_null()).select(KEY)
    print(f"planted keys (stride {D.STRIDE}): {planted_keys.height:,} ({time.time() - t0:.0f} s)", flush=True)

    N = nearest_ct(planted_keys)
    sp, secondary_used, knife_used = speeds(N)
    N = N.with_columns(pl.Series("speed", sp, dtype=pl.Float64)).rename({"ct_dist": "nearest_ct_dist_t6"})
    print(f"nearest CT found for {N.height:,} keys; secondary weapon used for {secondary_used:,}; knife value for {knife_used:,} ({time.time() - t0:.0f} s)", flush=True)

    # sanity: the nearest-CT distance reproduces the stored DX input (defuse_race_de_mirage.parquet)
    ref = (pl.read_parquet(os.path.join(DIAG, "defuse_race_de_mirage.parquet"), columns=KEY + ["nearest_ct_dist"])
             .with_columns(pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int64))
             .unique(KEY, keep="first", maintain_order=True))
    chk = N.join(ref, on=KEY, how="inner")
    repro = float((chk["nearest_ct_dist_t6"] - chk["nearest_ct_dist"]).abs().max()) if chk.height else float("nan")
    print(f"nearest-CT distance vs stored DX input: max abs diff {repro:.2e} over {chk.height:,} keys", flush=True)
    if not repro <= 1e-3:
        raise SystemExit(f"nearest-CT distance does not reproduce the stored DX input (max abs diff {repro})")

    G = G.join(N.select(KEY + ["nearest_ct_dist_t6", "speed"]), on=KEY, how="left", maintain_order="left")
    none = pl.lit(None, dtype=pl.Float64)
    fuse_left = FUSE - pl.col("seconds_since_plant")
    post = ((pl.col("bomb_planted") == 1) & pl.col("seconds_since_plant").is_not_null() & pl.col("site").is_not_null()
            & pl.col("nearest_ct_dist_t6").is_not_null() & pl.col("speed").is_not_null())
    G = G.with_columns(
        post.alias("post"),
        pl.when(post).then(fuse_left - (pl.col("nearest_ct_dist_t6") / SPEED_CONST + DEFUSE)).otherwise(none).alias("m_const"),
        pl.when(post).then(fuse_left - (pl.col("nearest_ct_dist_t6") / pl.col("speed") + DEFUSE)).otherwise(none).alias("m_pw"),
    ).sort(KEY)

    tr = G.filter(pl.col("split") == "train")
    va = G.filter(pl.col("split") == "val")
    te = G.filter((pl.col("split") == "test") & (pl.col("map_name") == "de_mirage"))
    n_post_va = int(va["post"].sum())
    n_post_te = int(te["post"].sum())
    print(f"post-plant validation rows {n_post_va:,}, post-plant test rows {n_post_te:,} ({time.time() - t0:.0f} s)", flush=True)

    arms = {"CONST": feats + ["m_const"], "PW": feats + ["m_pw"]}
    per_seed = {}
    va_match = va["match_key"].to_numpy()
    te_match = te["match_key"].to_numpy()
    va_y = va["label_ct_win"].to_numpy().astype(np.int64)
    te_y = te["label_ct_win"].to_numpy().astype(np.int64)
    va_post = va["post"].to_numpy()
    te_post = te["post"].to_numpy()
    for seed in SEEDS:
        preds = {}
        for name, cols in arms.items():
            m, best = W.fit_xgb(tr.select(cols).to_numpy().astype(np.float32), tr["label_ct_win"].to_numpy().astype(np.int64),
                                va.select(cols).to_numpy().astype(np.float32), va["label_ct_win"].to_numpy().astype(np.int64),
                                mono=W.mono_for(cols), seed=seed, device="cuda")
            preds[name] = {"val": W.predict(m, best, va.select(cols).to_numpy().astype(np.float32)),
                           "test": W.predict(m, best, te.select(cols).to_numpy().astype(np.float32)), "best": int(best)}
            print(f"seed {seed}: {name} best iteration {best} ({time.time() - t0:.0f} s)", flush=True)
        d_va = paired(va_match[va_post], va_y[va_post], preds["PW"]["val"][va_post], preds["CONST"]["val"][va_post])
        d_te = paired(te_match[te_post], te_y[te_post], preds["PW"]["test"][te_post], preds["CONST"]["test"][te_post])
        ok = bool(d_va["diff"] <= -0.0005 and d_va["ci95"][1] < 0)
        per_seed[seed] = {"PW_minus_CONST_validation": d_va, "PW_minus_CONST_test_descriptive": d_te, "pass": ok,
                          "best_iteration": {k: v["best"] for k, v in preds.items()}}
        print(f"seed {seed}: validation PW minus CONST {d_va['diff']:+.6f} CI [{d_va['ci95'][0]:+.6f}, {d_va['ci95'][1]:+.6f}], pass {ok}",
              flush=True)
        W.dump_json({"seeds_done": list(per_seed), "per_seed": per_seed}, OUT + ".partial")

    diffs = [per_seed[s]["PW_minus_CONST_validation"]["diff"] for s in SEEDS]
    mean = float(np.mean(diffs))
    passes = bool(all(per_seed[s]["pass"] for s in SEEDS) and mean <= -0.0005)
    res = {"decision_set": "post-plant validation seconds, de_mirage, every 10th second",
           "rule": "for each seed mean(PW minus CONST) <= -0.0005 with CI upper < 0; three-seed mean <= -0.0005 (pin da91d46d)",
           "rows": {"post_plant_validation": n_post_va, "post_plant_test": n_post_te, "planted_keys": planted_keys.height},
           "nearest_ct_reproduction_max_abs_diff": repro, "secondary_weapon_rows": secondary_used, "knife_value_rows": knife_used,
           "per_seed": per_seed, "mean_over_seeds_validation": mean, "per_seed_pass": [per_seed[s]["pass"] for s in SEEDS],
           "passes": passes, "seconds": round(time.time() - t0)}
    W.dump_json(res, OUT)
    print(json.dumps({"mean_over_seeds_validation": mean, "passes": passes}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
