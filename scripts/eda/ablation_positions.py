"""Ablation (EDA): bomb-site positions and map control for CS2 round win-probability.

Question: do in-round positions add information about the round winner beyond the coarse BASE
state (alive counts, bomb state, time, score)? A third model adds map name as a one-hot control.

Inputs (G:, already validated; not re-extracted from E:):
  data\\state_sample\\players_1hz.parquet, data\\state_sample\\rounds.parquet
  logs\\eda\\state_1hz_validation.json (preflight only)

Steps
  1. Preflight on the state sample and the extraction validation log.
  2. Main sample: rounds with alive_gap_snapshots == 0. Drops the one flagged round (report count).
  3. Planter position per planted round, from players_1hz only. The planter is the T player holding
     inv_c4 at the latest snapshot at or before plant_tick (tier 1; tier 2 = nearest T carrier
     snapshot after the plant if none before). Their row at plant_tick, or the nearest snapshot of
     that player when the exact row is missing, gives the position.
  4. Bomb-site centres per map: k-means k = 2 (3D) fit on TRAIN planted rounds only. Site A is the
     larger cluster. Reports sizes and separation. Test plants are used only as an out-of-sample check.
  5. Snapshot features (3D game units), saved to data\\state_sample\\positions_snapshots.parquet.
  6. XGBoost (hist, depth 4, eta 0.05, min_child_weight 20, subsample 0.8, seed 0, early stopping 100).
     Validation = 15% of the train matches (random.Random(0)), used only for early stopping.
     Models: BASE, BASE+POS, BASE+POS+MAP, each with score_diff (as specified) and without it.
  7. Test log loss, Brier, AUC; gain importance; match-level bootstrap over the 30 test matches
     (1,000 draws, seed 0) for the differences; per-map differences with per-map bootstrap intervals;
     seed-noise check (5 XGBoost seeds on BASE and BASE+POS).

Run from the project root:
  pixi run python -I -X utf8 scripts\\eda\\ablation_positions.py
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, silhouette_score

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"G:\pureskill\csds-cs2")
STATE = ROOT / "data" / "state_sample"
LOG_DIR = ROOT / "logs" / "eda"
LOG_PATH = LOG_DIR / "ablation_positions.log"
JSON_PATH = LOG_DIR / "ablation_positions_results.json"
VALIDATION_JSON = LOG_DIR / "state_1hz_validation.json"
SNAP_OUT = STATE / "positions_snapshots.parquet"

SEED = 0
TICK = 64
T, CT = 2, 3
SITE_RADIUS = 1000.0
BOMB_RADIUS = 1500.0
MIN_PLANTS = 10
VAL_FRAC = 0.15
N_BOOT = 1000
NOISE_SEEDS = [0, 1, 2, 3, 4]
KEYS = ["match_key", "round", "tick"]
EXPECT = {"rows": 1231817, "snapshots": 179466, "rounds": 2673, "matches": 150,
          "train_matches": 120, "test_matches": 30, "train_rounds": 2185, "test_rounds": 488,
          "planted_rounds": 1238, "gap_rounds": 1, "gap_flag": 9}

BASE_S = ["alive_ct", "alive_t", "alive_diff", "bomb_planted", "seconds_since_live",
          "seconds_since_plant", "round_number", "score_diff"]
BASE_N = [c for c in BASE_S if c != "score_diff"]
POS = ["ct_mean_dist_A", "ct_mean_dist_B", "ct_min_dist_A", "ct_min_dist_B",
       "t_mean_dist_A", "t_mean_dist_B", "t_min_dist_A", "t_min_dist_B",
       "ct_n_near_A", "ct_n_near_B", "t_n_near_A", "t_n_near_B",
       "ct_spread", "t_spread", "t_carrier_site_dist",
       "ct_n_near_bomb", "ct_nearest_bomb_dist"]
PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist",
          "max_depth": 4, "eta": 0.05, "min_child_weight": 20, "subsample": 0.8, "seed": SEED}

LOG_FH = None


def out(*parts):
    s = " ".join(str(p) for p in parts)
    print(s, flush=True)
    LOG_FH.write(s + "\n")
    LOG_FH.flush()


def fail(msg):
    out("ERROR REPORT:", msg)
    LOG_FH.close()
    raise SystemExit(1)


# ---------------------------------------------------------------- 1. preflight
def preflight(P, R):
    out("== 1. Preflight (state sample and validation log) ==")
    fails = []

    def chk(name, ok, detail=""):
        out(f"  [{'OK' if ok else 'FAIL'}] {name} {detail}".rstrip())
        if not ok:
            fails.append(name)

    for f in ("players_1hz.parquet", "rounds.parquet"):
        chk(f"file exists {f}", (STATE / f).exists())
    chk("validation log exists", VALIDATION_JSON.exists())
    if fails:
        fail(f"missing inputs: {fails}")
    chk("player rows", P.height == EXPECT["rows"], f"({P.height})")
    chk("rounds", R.height == EXPECT["rounds"], f"({R.height})")
    chk("matches", R["match_key"].n_unique() == EXPECT["matches"], f"({R['match_key'].n_unique()})")
    sm = dict(R.group_by("split").agg(pl.col("match_key").n_unique()).rows())
    chk("split matches train/test", sm.get("train") == EXPECT["train_matches"]
        and sm.get("test") == EXPECT["test_matches"], f"({sm})")
    sr = dict(R.group_by("split").len().rows())
    chk("split rounds train/test", sr.get("train") == EXPECT["train_rounds"]
        and sr.get("test") == EXPECT["test_rounds"], f"({sr})")
    chk("health 1-100", P["health"].min() >= 1 and P["health"].max() <= 100)
    chk("armor 0-100", P["armor"].min() >= 0 and P["armor"].max() <= 100)
    chk("team_code in {2,3}", set(P["team_code"].unique().to_list()) <= {T, CT})
    per_side = P.group_by(KEYS + ["team_code"]).len()
    chk("max 5 per side per snapshot", per_side["len"].max() <= 5, f"(max {per_side['len'].max()})")
    n_snap = P.select(KEYS).unique().height
    chk("distinct snapshots", n_snap == EXPECT["snapshots"], f"({n_snap})")
    dup = P.group_by(KEYS + ["player_id"]).len().filter(pl.col("len") > 1).height
    chk("no duplicate player-snapshot keys", dup == 0, f"({dup})")
    PR = P.select(["match_key", "round", "tick"]).unique().join(
        R.select(["match_key", "round", "live_start_tick", "round_end_tick"]),
        on=["match_key", "round"], how="left")
    chk("every snapshot has a round", PR["live_start_tick"].null_count() == 0)
    chk("snapshot ticks on the 64-tick grid from live start",
        PR.filter((pl.col("tick") - pl.col("live_start_tick")) % TICK != 0).height == 0)
    chk("snapshot ticks before round end", PR.filter(pl.col("tick") >= pl.col("round_end_tick")).height == 0)
    chk("score sum = round - 1", R.filter(pl.col("ct_score") + pl.col("t_score")
                                          != pl.col("round_number") - 1).height == 0)
    pl_rows = R.filter(pl.col("plant_tick").is_not_null())
    chk("planted rounds", pl_rows.height == EXPECT["planted_rounds"], f"({pl_rows.height})")
    chk("plant_tick inside the live window", pl_rows.filter(
        (pl.col("plant_tick") < pl.col("live_start_tick")) | (pl.col("plant_tick") >= pl.col("round_end_tick"))
    ).height == 0)
    gap = R.filter(pl.col("alive_gap_snapshots") > 0).select(["match_key", "round", "alive_gap_snapshots"]).rows()
    chk("alive-gap round only (9 flagged snapshots)", len(gap) == EXPECT["gap_rounds"]
        and gap[0][2] == EXPECT["gap_flag"], f"({gap})")
    val = json.loads(VALIDATION_JSON.read_text(encoding="utf-8"))
    a = val["2_alive_check"]
    chk("validation log: agreement 179457 of 179466", a["agree_inclusive_rule"] == 179457
        and a["snapshots"] == EXPECT["snapshots"])
    if fails:
        fail(f"preflight failed: {fails}")
    out("  preflight passed")


# ---------------------------------------------------------------- 2. main sample
def main_sample(P, R):
    out("\n== 2. Main sample (alive_gap_snapshots == 0) ==")
    gap_keys = R.filter(pl.col("alive_gap_snapshots") > 0).select(["match_key", "round", "split"])
    gk = gap_keys.select(["match_key", "round"])
    dropped_snaps = P.join(gk, on=["match_key", "round"], how="inner").select(KEYS).unique().height
    out(f"  dropped rounds: {gap_keys.height} {gap_keys.rows()}; snapshots in dropped rounds: {dropped_snaps}")
    Rm = R.join(gk, on=["match_key", "round"], how="anti")
    Pm = P.join(gk, on=["match_key", "round"], how="anti")
    snaps_m = Pm.select(KEYS).unique().height
    out(f"  main rounds {Rm.height} (train {Rm.filter(pl.col('split') == 'train').height}, "
        f"test {Rm.filter(pl.col('split') == 'test').height}); main snapshots {snaps_m}; "
        f"matches {Rm['match_key'].n_unique()} (train {Rm.filter(pl.col('split') == 'train')['match_key'].n_unique()}, "
        f"test {Rm.filter(pl.col('split') == 'test')['match_key'].n_unique()})")
    if snaps_m != EXPECT["snapshots"] - dropped_snaps or Rm.height != EXPECT["rounds"] - 1:
        fail("main-sample counts do not match the expected drop")
    return Rm, Pm, {"dropped_rounds": gap_keys.height, "dropped_snapshots": dropped_snaps,
                    "dropped_rows": gap_keys.rows(),
                    "main_rounds": Rm.height, "main_snapshots": snaps_m}


# ---------------------------------------------------------------- 3. planter positions
def planter_positions(Pm, Rm):
    out("\n== 3. Planter positions at plant (players_1hz) ==")
    planted = Rm.filter(pl.col("plant_tick").is_not_null()).select(
        ["match_key", "round", "map_name", "split", "plant_tick"])
    carriers = (Pm.filter((pl.col("team_code") == T) & (pl.col("inv_c4") > 0))
                .select(["match_key", "round", "tick", "player_id"])
                .join(planted.select(["match_key", "round", "plant_tick"]), on=["match_key", "round"], how="inner")
                .with_columns((pl.col("tick") - pl.col("plant_tick")).abs().alias("dt")))
    before = (carriers.filter(pl.col("tick") <= pl.col("plant_tick"))
              .sort(["match_key", "round", "tick"])
              .unique(subset=["match_key", "round"], keep="last", maintain_order=True)
              .select(["match_key", "round", "player_id"]).with_columns(pl.lit(1).alias("tier")))
    after = (carriers.join(before.select(["match_key", "round"]), on=["match_key", "round"], how="anti")
             .sort(["match_key", "round", "dt", "tick"])
             .unique(subset=["match_key", "round"], keep="first", maintain_order=True)
             .select(["match_key", "round", "player_id"]).with_columns(pl.lit(2).alias("tier")))
    chosen = pl.concat([before, after])
    located = (Pm.select(["match_key", "round", "tick", "player_id", "x_pos", "y_pos", "z_pos"])
               .join(chosen, on=["match_key", "round", "player_id"], how="inner")
               .join(planted.select(["match_key", "round", "plant_tick"]), on=["match_key", "round"], how="inner")
               .with_columns((pl.col("tick") - pl.col("plant_tick")).alias("gap_ticks"),
                             (pl.col("tick") - pl.col("plant_tick")).abs().alias("dt"))
               .sort(["match_key", "round", "dt", "tick"])
               .unique(subset=["match_key", "round"], keep="first", maintain_order=True)
               .select(["match_key", "round", "player_id", "tier", "gap_ticks",
                        pl.col("x_pos").alias("px"), pl.col("y_pos").alias("py"), pl.col("z_pos").alias("pz")]))
    pp = planted.join(located, on=["match_key", "round"], how="left")
    tiers = pp.group_by(pl.col("tier").fill_null(0)).len().sort("tier").rows()
    exact = pp.filter(pl.col("gap_ticks") == 0).height
    lag = pp.filter(pl.col("gap_ticks").is_not_null()).with_columns(pl.col("gap_ticks").abs().alias("a"))["a"]
    out(f"  planted rounds {pp.height}; tier counts (0 = no carrier, 1 = before plant, 2 = after) {tiers}")
    out(f"  planter row exactly at plant_tick: {exact}; |snapshot - plant| ticks: median "
        f"{float(lag.median()):.0f}, max {int(lag.max())}; split {pp.group_by('split').len().sort('split').rows()}")
    if pp.filter(pl.col("px").is_null()).height:
        out(f"  planted rounds with no located planter: {pp.filter(pl.col('px').is_null()).height}")
    return pp, {"tiers": {str(k): int(v) for k, v in tiers}, "exact_row_at_plant": exact,
                "median_abs_gap_ticks": float(lag.median()), "max_abs_gap_ticks": int(lag.max())}


# ---------------------------------------------------------------- 4. bomb sites
def derive_sites(pp):
    out("\n== 4. Bomb-site centres per map (k-means k=2, 3D; fit on TRAIN planted rounds only) ==")
    located = pp.filter(pl.col("px").is_not_null())
    sites, rows = {}, []
    for m in sorted(pp["map_name"].unique().to_list()):
        tr = located.filter((pl.col("map_name") == m) & (pl.col("split") == "train"))
        te = located.filter((pl.col("map_name") == m) & (pl.col("split") == "test"))
        pts = tr.select(["px", "py", "pz"]).to_numpy().astype(float)
        n = len(pts)
        row = {"map": m, "train_plants": n, "test_plants": te.height}
        if n < MIN_PLANTS:
            sites[m] = None
            row["note"] = f"fewer than {MIN_PLANTS} train plants; sites not derived"
            rows.append(row)
            continue
        km = KMeans(n_clusters=2, n_init=10, random_state=SEED).fit(pts)
        sizes = np.bincount(km.labels_, minlength=2)
        order = np.argsort(-sizes, kind="stable")
        cA = km.cluster_centers_[order[0]].astype(float)
        cB = km.cluster_centers_[order[1]].astype(float)
        lab = np.where(km.labels_ == order[0], 0, 1)
        dA = np.linalg.norm(pts - cA, axis=1)
        dB = np.linalg.norm(pts - cB, axis=1)
        own = np.where(lab == 0, dA, dB)
        rms = [float(np.sqrt(np.mean(own[lab == k] ** 2))) for k in (0, 1)]
        D = float(np.linalg.norm(cA - cB))
        sil = float(silhouette_score(pts, lab))
        near = np.minimum(dA, dB)
        pts_te = te.select(["px", "py", "pz"]).to_numpy().astype(float)
        if len(pts_te):
            dte = np.minimum(np.linalg.norm(pts_te - cA, axis=1), np.linalg.norm(pts_te - cB, axis=1))
            te_within, te_med = float(np.mean(dte <= SITE_RADIUS)), float(np.median(dte))
        else:
            te_within, te_med = None, None
        sites[m] = {"A": cA, "B": cB}
        row.update({"size_A": int(sizes[order[0]]), "size_B": int(sizes[order[1]]),
                    "centre_A": [round(float(v), 1) for v in cA], "centre_B": [round(float(v), 1) for v in cB],
                    "centre_distance": round(D, 1), "rms_A": round(rms[0], 1), "rms_B": round(rms[1], 1),
                    "separation_ratio": round(D / float(np.mean(rms)), 2), "silhouette": round(sil, 3),
                    "train_share_within_1000": round(float(np.mean(near <= SITE_RADIUS)), 4),
                    "test_share_within_1000": None if te_within is None else round(te_within, 4),
                    "test_median_dist": None if te_med is None else round(te_med, 1)})
        rows.append(row)
    for r in rows:
        out("  " + json.dumps(r))
    return sites, rows


# ---------------------------------------------------------------- 5. snapshot features
def build_snapshots(Pm, Rm, sites, pp):
    out("\n== 5. Snapshot features (3D game units) ==")
    maps = sorted(Rm["map_name"].unique().to_list())
    srows = []
    for m in maps:
        s = sites.get(m)
        if s is None:
            fail(f"no bomb sites for map {m}")
        srows.append({"map_name": m, "axA": float(s["A"][0]), "ayA": float(s["A"][1]), "azA": float(s["A"][2]),
                      "axB": float(s["B"][0]), "ayB": float(s["B"][1]), "azB": float(s["B"][2])})
    site_tab = pl.DataFrame(srows, schema={"map_name": pl.String, "axA": pl.Float64, "ayA": pl.Float64,
                                           "azA": pl.Float64, "axB": pl.Float64, "ayB": pl.Float64,
                                           "azB": pl.Float64})
    bomb_tab = (pp.filter(pl.col("px").is_not_null())
                .select(["match_key", "round", pl.col("px").alias("bx"), pl.col("py").alias("by"),
                         pl.col("pz").alias("bz")]))
    R2 = (Rm.select(["match_key", "round", "split", "ct_win", "map_name", "round_number", "ct_score",
                     "t_score", "plant_tick"])
          .join(bomb_tab, on=["match_key", "round"], how="left")
          .join(site_tab, on="map_name", how="left"))
    PP = (Pm.select(["match_key", "round", "tick", "seconds_since_live", "player_id", "team_code",
                     "inv_c4", "x_pos", "y_pos", "z_pos"])
          .with_columns(pl.col("x_pos").cast(pl.Float64), pl.col("y_pos").cast(pl.Float64),
                        pl.col("z_pos").cast(pl.Float64), pl.col("inv_c4").cast(pl.Int64))
          .join(R2, on=["match_key", "round"], how="left"))

    x, y, z = pl.col("x_pos"), pl.col("y_pos"), pl.col("z_pos")

    def d3(ax, ay, az):
        return ((x - ax).pow(2) + (y - ay).pow(2) + (z - az).pow(2)).sqrt()

    planted = pl.col("plant_tick").is_not_null() & (pl.col("tick") >= pl.col("plant_tick"))
    PP = PP.with_columns(
        d3(pl.col("axA"), pl.col("ayA"), pl.col("azA")).alias("dA"),
        d3(pl.col("axB"), pl.col("ayB"), pl.col("azB")).alias("dB"),
        d3(pl.col("bx"), pl.col("by"), pl.col("bz")).alias("dBomb"),
        planted.alias("planted"),
        (planted & pl.col("bx").is_not_null()).alias("bomb_ok"),
        ((pl.col("team_code") == T) & (pl.col("inv_c4") > 0)).alias("carrier"),
    )
    PP = PP.with_columns(
        pl.when(pl.col("carrier")).then(pl.min_horizontal("dA", "dB")).alias("carrier_site_dist"))

    def team_aggs(team, tag):
        c = pl.col("team_code") == team
        return [
            c.cast(pl.Int64).sum().alias(f"n_{tag}"),
            pl.when(c).then(pl.col("dA")).mean().alias(f"{tag}_mean_dist_A"),
            pl.when(c).then(pl.col("dB")).mean().alias(f"{tag}_mean_dist_B"),
            pl.when(c).then(pl.col("dA")).min().alias(f"{tag}_min_dist_A"),
            pl.when(c).then(pl.col("dB")).min().alias(f"{tag}_min_dist_B"),
            (c & (pl.col("dA") <= SITE_RADIUS)).cast(pl.Int64).sum().alias(f"{tag}_n_near_A"),
            (c & (pl.col("dB") <= SITE_RADIUS)).cast(pl.Int64).sum().alias(f"{tag}_n_near_B"),
        ]

    cb = (pl.col("team_code") == CT) & pl.col("bomb_ok")
    extras = [
        pl.col("bomb_ok").cast(pl.Int64).max().alias("snap_bomb_ok"),
        (cb & (pl.col("dBomb") <= BOMB_RADIUS)).cast(pl.Int64).sum().alias("ct_n_near_bomb_raw"),
        pl.when(cb).then(pl.col("dBomb")).min().alias("ct_nearest_bomb_raw"),
        pl.col("carrier").cast(pl.Int64).sum().alias("n_carriers"),
        pl.col("carrier_site_dist").max().alias("t_carrier_site_dist"),
    ]
    agg = PP.group_by(KEYS).agg(team_aggs(CT, "ct") + team_aggs(T, "t") + extras)

    base_pp = PP.select(["match_key", "round", "tick", "team_code", "player_id", "x_pos", "y_pos", "z_pos"])
    pairs = (base_pp.join(base_pp, on=["match_key", "round", "tick", "team_code"], suffix="_b")
             .filter(pl.col("player_id") < pl.col("player_id_b"))
             .with_columns(((pl.col("x_pos") - pl.col("x_pos_b")).pow(2)
                            + (pl.col("y_pos") - pl.col("y_pos_b")).pow(2)
                            + (pl.col("z_pos") - pl.col("z_pos_b")).pow(2)).sqrt().alias("pd")))
    spread = pairs.group_by(["match_key", "round", "tick", "team_code"]).agg(pl.col("pd").mean().alias("spread"))
    sp_ct = spread.filter(pl.col("team_code") == CT).select(KEYS + [pl.col("spread").alias("ct_spread")])
    sp_t = spread.filter(pl.col("team_code") == T).select(KEYS + [pl.col("spread").alias("t_spread")])

    S = PP.select(["match_key", "round", "tick", "seconds_since_live", "split", "map_name", "ct_win",
                   "round_number", "ct_score", "t_score", "plant_tick"]).unique(subset=KEYS, keep="any")
    out(f"  snapshots in skeleton: {S.height} (main sample expects {EXPECT['snapshots']} minus dropped)")
    F = (S.join(agg, on=KEYS, how="left")
         .join(sp_ct, on=KEYS, how="left")
         .join(sp_t, on=KEYS, how="left"))
    F = F.with_columns([pl.col(c).fill_null(0) for c in
                        ["n_ct", "n_t", "ct_n_near_A", "ct_n_near_B", "t_n_near_A", "t_n_near_B",
                         "n_carriers", "snap_bomb_ok", "ct_n_near_bomb_raw"]])
    if F["n_carriers"].max() > 1:
        fail("more than one T carrier in a snapshot")
    planted_f = pl.col("plant_tick").is_not_null() & (pl.col("tick") >= pl.col("plant_tick"))
    bomb_known = pl.col("snap_bomb_ok") == 1
    F = F.with_columns(
        pl.col("n_ct").cast(pl.Float64).alias("alive_ct"),
        pl.col("n_t").cast(pl.Float64).alias("alive_t"),
        (pl.col("n_ct") - pl.col("n_t")).cast(pl.Float64).alias("alive_diff"),
        planted_f.cast(pl.Float64).alias("bomb_planted"),
        pl.col("seconds_since_live").cast(pl.Float64),
        pl.when(planted_f).then((pl.col("tick") - pl.col("plant_tick")) / TICK).otherwise(-1.0)
          .alias("seconds_since_plant"),
        pl.col("round_number").cast(pl.Float64),
        (pl.col("ct_score") - pl.col("t_score")).cast(pl.Float64).alias("score_diff"),
        pl.col("ct_n_near_A").cast(pl.Float64), pl.col("ct_n_near_B").cast(pl.Float64),
        pl.col("t_n_near_A").cast(pl.Float64), pl.col("t_n_near_B").cast(pl.Float64),
        pl.when(bomb_known).then(pl.col("ct_n_near_bomb_raw").cast(pl.Float64)).otherwise(None)
          .alias("ct_n_near_bomb"),
        pl.when(bomb_known).then(pl.col("ct_nearest_bomb_raw")).otherwise(None).alias("ct_nearest_bomb_dist"),
    )
    nan_cols = ["ct_mean_dist_A", "ct_mean_dist_B", "ct_min_dist_A", "ct_min_dist_B",
                "t_mean_dist_A", "t_mean_dist_B", "t_min_dist_A", "t_min_dist_B",
                "ct_spread", "t_spread", "t_carrier_site_dist", "ct_n_near_bomb", "ct_nearest_bomb_dist"]
    F = F.with_columns([pl.col(c).cast(pl.Float64).fill_null(float("nan")) for c in nan_cols])
    keep = ["match_key", "round", "tick", "split", "map_name", "ct_win"] + BASE_S + POS
    F = F.select(keep).sort(KEYS)
    F.write_parquet(SNAP_OUT)

    n_pl = F.filter(pl.col("bomb_planted") == 1).height
    n_pl_known = F.filter((pl.col("bomb_planted") == 1) & pl.col("ct_nearest_bomb_dist").is_not_nan()).height
    out(f"  wrote {SNAP_OUT} ({F.height} rows, {len(BASE_S) + len(POS)} base+position features)")
    out(f"  planted snapshots {n_pl}; with a located bomb {n_pl_known}; "
        f"snapshots with no alive CT {F.filter(pl.col('alive_ct') == 0).height}, no alive T "
        f"{F.filter(pl.col('alive_t') == 0).height}; snapshots with no T carrier "
        f"{F.filter(pl.col('t_carrier_site_dist').is_nan()).height}")
    te = F.filter(pl.col("split") == "test")
    nan_share = {c: round(float(te[c].is_nan().mean()), 4) for c in POS}
    out(f"  test-split NaN share by feature: {json.dumps(nan_share)}")
    return F, {"snapshots": F.height, "planted_snapshots": n_pl, "planted_with_bomb_location": n_pl_known,
               "test_nan_share": nan_share}


# ---------------------------------------------------------------- 6. models
def mets(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return {"n": int(len(y)), "logloss": float(log_loss(y, p, labels=[0, 1])),
            "brier": float(brier_score_loss(y, p)), "auc": float(roc_auc_score(y, p))}


def fit_model(M, names, idx, y, fit_m, val_m, test_m, seed):
    params = dict(PARAMS, seed=seed)
    dfit = xgb.DMatrix(M[fit_m][:, idx], label=y[fit_m], feature_names=names)
    dval = xgb.DMatrix(M[val_m][:, idx], label=y[val_m], feature_names=names)
    dte = xgb.DMatrix(M[test_m][:, idx], feature_names=names)
    bst = xgb.train(params, dfit, num_boost_round=5000, evals=[(dval, "val")],
                    early_stopping_rounds=100, verbose_eval=False)
    best = int(bst.best_iteration)
    p = bst.predict(dte, iteration_range=(0, best + 1))
    return bst, p, best, float(bst.best_score)


def ll_rows(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def bootstrap(y, preds, mk_te, map_of_match, pair_list):
    uniq, codes = np.unique(mk_te, return_inverse=True)
    nm = len(uniq)
    draws = np.random.default_rng(SEED).integers(0, nm, size=(N_BOOT, nm))
    W = np.empty((N_BOOT, nm))
    for b in range(N_BOOT):
        W[b] = np.bincount(draws[b], minlength=nm)
    N = np.bincount(codes, minlength=nm).astype(float)
    S_ll, S_br, LL, BR, AU = {}, {}, {}, {}, {}
    for name, p in preds.items():
        S_ll[name] = np.bincount(codes, weights=ll_rows(y, p), minlength=nm)
        S_br[name] = np.bincount(codes, weights=(p - y) ** 2, minlength=nm)
        LL[name] = (W @ S_ll[name]) / (W @ N)
        BR[name] = (W @ S_br[name]) / (W @ N)
        au = np.empty(N_BOOT)
        for b in range(N_BOOT):
            au[b] = roc_auc_score(y, p, sample_weight=W[b][codes])
        AU[name] = au
    res = {}
    for a, c in pair_list:
        key = f"{a} - {c}"
        r = {}
        for met, arr_a, arr_c, better in (("logloss", LL[a], LL[c], "neg"), ("brier", BR[a], BR[c], "neg"),
                                          ("auc", AU[a], AU[c], "pos")):
            d = arr_a - arr_c
            share = float(np.mean(d < 0)) if better == "neg" else float(np.mean(d > 0))
            r[met] = {"mean": float(d.mean()), "p2_5": float(np.percentile(d, 2.5)),
                      "p97_5": float(np.percentile(d, 97.5)), "share_draws_better": share}
        res[key] = r
    # point estimates on the full test set
    point = {}
    for name in preds:
        point[name] = mets(y, preds[name])
    # per-map logloss and Brier differences (first pair in pair_list), per-map bootstrap
    a, c = pair_list[0]
    per_map = {}
    match_map = np.array([map_of_match[m] for m in uniq])
    for m in sorted(set(match_map)):
        mask = (match_map == m).astype(float)
        den = W @ (N * mask)
        point_rows = (codes >= 0) & (match_map[codes] == m)
        pt_ll = float(np.sum(ll_rows(y, preds[a])[point_rows] - ll_rows(y, preds[c])[point_rows])
                      / point_rows.sum())
        pt_br = float(np.sum(((preds[a] - y) ** 2 - (preds[c] - y) ** 2)[point_rows]) / point_rows.sum())
        d_ll = (W @ ((S_ll[a] - S_ll[c]) * mask))
        d_br = (W @ ((S_br[a] - S_br[c]) * mask))
        with np.errstate(invalid="ignore", divide="ignore"):
            bl = np.where(den > 0, d_ll / den, np.nan)
            bb = np.where(den > 0, d_br / den, np.nan)
        per_map[m] = {"test_matches": int(mask.sum()),
                      "test_snapshots": int(point_rows.sum()),
                      "diff_logloss": pt_ll, "diff_brier": pt_br,
                      "boot_logloss_p2_5": float(np.nanpercentile(bl, 2.5)) if np.isfinite(bl).any() else None,
                      "boot_logloss_p97_5": float(np.nanpercentile(bl, 97.5)) if np.isfinite(bl).any() else None,
                      "boot_valid_draws": int(np.isfinite(bl).sum()),
                      "boot_brier_p2_5": float(np.nanpercentile(bb, 2.5)) if np.isfinite(bb).any() else None,
                      "boot_brier_p97_5": float(np.nanpercentile(bb, 97.5)) if np.isfinite(bb).any() else None}
    return res, point, per_map


def main():
    global LOG_FH
    t0 = time.time()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FH = open(LOG_PATH, "w", encoding="utf-8")
    out("Ablation: bomb-site positions and map control")
    P = pl.read_parquet(STATE / "players_1hz.parquet").with_columns(
        pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64), pl.col("player_id").cast(pl.Int64))
    R = pl.read_parquet(STATE / "rounds.parquet").with_columns(
        pl.col("round").cast(pl.Int64), pl.col("plant_tick").cast(pl.Int64),
        pl.col("live_start_tick").cast(pl.Int64), pl.col("round_end_tick").cast(pl.Int64),
        pl.col("round_number").cast(pl.Int64), pl.col("ct_score").cast(pl.Int64),
        pl.col("t_score").cast(pl.Int64))
    preflight(P, R)
    Rm, Pm, drop_info = main_sample(P, R)

    # train-only check on the carve: no match on both sides is asserted below
    pp, plant_info = planter_positions(Pm, Rm)
    sites, site_rows = derive_sites(pp)
    F, feat_info = build_snapshots(Pm, Rm, sites, pp)

    maps = sorted(F["map_name"].unique().to_list())
    F = F.with_columns([(pl.col("map_name") == m).cast(pl.Float64).alias(f"map_{m}") for m in maps])
    MAP = [f"map_{m}" for m in maps]

    out("\n== 6. Models (XGBoost, early stopping on a match-level validation carved from train) ==")
    train_matches = sorted(F.filter(pl.col("split") == "train")["match_key"].unique().to_list())
    ms = list(train_matches)
    random.Random(SEED).shuffle(ms)
    n_val = round(VAL_FRAC * len(ms))
    val_set = set(ms[:n_val])
    test_matches = sorted(F.filter(pl.col("split") == "test")["match_key"].unique().to_list())
    out(f"  train matches {len(train_matches)}; validation matches {len(val_set)} (15%, seed {SEED}); "
        f"fit matches {len(train_matches) - len(val_set)}; test matches {len(test_matches)}")
    out(f"  params: {PARAMS}; early stopping 100")
    if set(train_matches) & set(test_matches):
        fail("a match appears on both train and test")

    M_all = F.select(BASE_S + POS + MAP).to_numpy().astype(np.float64)
    names_all = BASE_S + POS + MAP
    col = {c: i for i, c in enumerate(names_all)}
    y_all = F["ct_win"].to_numpy().astype(np.float64)
    split = F["split"].to_numpy()
    mk = F["match_key"].to_numpy()
    is_val = (split == "train") & np.isin(mk, list(val_set))
    is_fit = (split == "train") & ~is_val
    is_test = split == "test"
    assert not (set(mk[is_fit]) & set(mk[is_val])), "match on both fit and validation"
    assert not (set(mk[is_test]) & set(mk[is_fit | is_val])), "match on both train and test"
    out(f"  fit rows {int(is_fit.sum())} ({len(set(mk[is_fit]))} matches); validation rows {int(is_val.sum())} "
        f"({len(set(mk[is_val]))} matches); test rows {int(is_test.sum())} ({len(set(mk[is_test]))} matches)")
    out(f"  CT-win share: fit {y_all[is_fit].mean():.4f}, validation {y_all[is_val].mean():.4f}, "
        f"test {y_all[is_test].mean():.4f}")

    def idx_of(cols):
        return [col[c] for c in cols], cols

    specs = {
        "BASE": BASE_S, "BASE+POS": BASE_S + POS, "BASE+POS+MAP": BASE_S + POS + MAP,
        "BASE_noscore": BASE_N, "BASE+POS_noscore": BASE_N + POS, "BASE+POS+MAP_noscore": BASE_N + POS + MAP,
    }
    results, preds, gains, bsts = {}, {}, {}, {}
    for name, cols in specs.items():
        idx, nm_ = idx_of(cols)
        bst, p, best, vll = fit_model(M_all, nm_, idx, y_all, is_fit, is_val, is_test, SEED)
        results[name] = {**mets(y_all[is_test], p), "n_features": len(cols), "best_iter": best,
                         "val_logloss": vll}
        preds[name] = p
        bsts[name] = bst
        gains[name] = bst.get_score(importance_type="gain")
        r = results[name]
        out(f"  {name:<22} features={len(cols):>2} best_iter={best:>4} val_logloss={vll:.5f} "
            f"test_logloss={r['logloss']:.5f} test_brier={r['brier']:.5f} test_auc={r['auc']:.5f}")
    const_p = np.full(int(is_test.sum()), float(y_all[is_fit].mean()))
    const = mets(y_all[is_test], const_p)
    out(f"  CONST (fit-split CT share {y_all[is_fit].mean():.4f}) test_logloss={const['logloss']:.5f} "
        f"test_brier={const['brier']:.5f} test_auc={const['auc']:.5f}")

    out("\n  gain importance (share of total gain):")
    gain_out = {}
    for name in ("BASE+POS", "BASE+POS+MAP", "BASE+POS_noscore", "BASE+POS+MAP_noscore"):
        g = gains[name]
        tot = sum(g.values())
        top = sorted(g.items(), key=lambda kv: -kv[1])[:12]
        sp = sum(v for k, v in g.items() if k in POS) / tot
        sm_ = sum(v for k, v in g.items() if k in MAP) / tot
        sb = sum(v for k, v in g.items() if k in BASE_S) / tot
        gain_out[name] = {"top": [[k, round(v / tot, 4)] for k, v in top], "share_pos": round(sp, 4),
                          "share_map": round(sm_, 4), "share_base": round(sb, 4)}
        out(f"  {name}: POS {sp:.3f}, MAP {sm_:.3f}, BASE {sb:.3f}")
        for k, v in top:
            out(f"    {k:<24} {v / tot:.4f}")

    # bootstrap over the 30 test matches
    out(f"\n== 7. Match-level bootstrap over test matches ({N_BOOT} draws, seed {SEED}) ==")
    te_frame = F.filter(pl.col("split") == "test").select(["match_key", "map_name", "round"])
    map_of_match = {r[0]: r[1] for r in te_frame.select(["match_key", "map_name"]).unique(subset=["match_key"]).rows()}
    mk_te = mk[is_test]
    y_te = y_all[is_test]
    preds_te = {k: preds[k] for k in ("BASE", "BASE+POS", "BASE+POS+MAP")}
    pairs_s = [("BASE+POS", "BASE"), ("BASE+POS+MAP", "BASE"), ("BASE+POS+MAP", "BASE+POS")]
    boot_s, point_s, per_map = bootstrap(y_te, preds_te, mk_te, map_of_match, pairs_s)
    preds_n = {k: preds[k] for k in ("BASE_noscore", "BASE+POS_noscore", "BASE+POS+MAP_noscore")}
    for pair, rr in boot_s.items():
        out(f"  [with score_diff] {pair}: logloss {rr['logloss']['mean']:+.5f} "
            f"[{rr['logloss']['p2_5']:+.5f}, {rr['logloss']['p97_5']:+.5f}] "
            f"share<0 {rr['logloss']['share_draws_better']:.3f}; brier {rr['brier']['mean']:+.5f} "
            f"[{rr['brier']['p2_5']:+.5f}, {rr['brier']['p97_5']:+.5f}]; auc {rr['auc']['mean']:+.5f} "
            f"[{rr['auc']['p2_5']:+.5f}, {rr['auc']['p97_5']:+.5f}] share>0 {rr['auc']['share_draws_better']:.3f}")
    # no-score variant: POS vs BASE and MAP vs BASE
    boot_n2, _, _ = bootstrap(y_te, preds_n, mk_te, map_of_match,
                              [("BASE+POS_noscore", "BASE_noscore"), ("BASE+POS+MAP_noscore", "BASE_noscore"),
                               ("BASE+POS+MAP_noscore", "BASE+POS_noscore")])
    for pair, rr in boot_n2.items():
        out(f"  [no score_diff]  {pair}: logloss {rr['logloss']['mean']:+.5f} "
            f"[{rr['logloss']['p2_5']:+.5f}, {rr['logloss']['p97_5']:+.5f}] "
            f"share<0 {rr['logloss']['share_draws_better']:.3f}; brier {rr['brier']['mean']:+.5f} "
            f"[{rr['brier']['p2_5']:+.5f}, {rr['brier']['p97_5']:+.5f}]; auc {rr['auc']['mean']:+.5f} "
            f"[{rr['auc']['p2_5']:+.5f}, {rr['auc']['p97_5']:+.5f}] share>0 {rr['auc']['share_draws_better']:.3f}")

    out("\n== 8. Per-map differences, BASE+POS minus BASE (with score_diff; test rows) ==")
    per_map_rows = []
    for m, v in sorted(per_map.items()):
        row = {"map": m, **{k: (round(val, 5) if isinstance(val, float) else val) for k, val in v.items()}}
        per_map_rows.append(row)
        out("  " + json.dumps(row))
    top3 = sorted(per_map_rows, key=lambda r: -abs(r["diff_logloss"]))[:3]
    out("  three largest |diff_logloss|: " + "; ".join(
        f"{r['map']} {r['diff_logloss']:+.5f} (95% boot [{r['boot_logloss_p2_5']}, {r['boot_logloss_p97_5']}], "
        f"{r['test_matches']} matches)" for r in top3))

    # seed noise: XGBoost seed only (same split), BASE and BASE+POS with score_diff
    out("\n== 9. Seed noise (XGBoost seed only; same split) ==")
    noise = {"BASE": [], "BASE+POS": []}
    for s in NOISE_SEEDS:
        for name in ("BASE", "BASE+POS"):
            cols = specs[name]
            idx, nm_ = idx_of(cols)
            _, p, _, _ = fit_model(M_all, nm_, idx, y_all, is_fit, is_val, is_test, s)
            noise[name].append(mets(y_te, p)["logloss"])
    nb, npos = np.array(noise["BASE"]), np.array(noise["BASE+POS"])
    noise_out = {"seeds": NOISE_SEEDS, "BASE_logloss": nb.tolist(), "BASE+POS_logloss": npos.tolist(),
                 "BASE_sd": float(nb.std(ddof=1)), "BASE+POS_sd": float(npos.std(ddof=1)),
                 "diff_mean": float((npos - nb).mean()), "diff_sd": float((npos - nb).std(ddof=1))}
    out(f"  BASE logloss over seeds: {np.round(nb, 5).tolist()} sd {noise_out['BASE_sd']:.5f}")
    out(f"  BASE+POS logloss over seeds: {np.round(npos, 5).tolist()} sd {noise_out['BASE+POS_sd']:.5f}")
    out(f"  POS - BASE across seeds: mean {noise_out['diff_mean']:+.5f} sd {noise_out['diff_sd']:.5f}")

    out(f"\nelapsed {time.time() - t0:.1f} s")
    summary = {"drop": drop_info, "planter": plant_info, "sites": site_rows, "features": feat_info,
               "models": results, "constant_test": const, "gain": gain_out,
               "bootstrap_with_score": boot_s, "bootstrap_no_score": boot_n2,
               "point_with_score": point_s,
               "per_map": per_map_rows, "seed_noise": noise_out,
               "splits": {"train_matches": len(train_matches), "validation_matches": len(val_set),
                          "fit_matches": len(train_matches) - len(val_set), "test_matches": len(test_matches),
                          "test_rows": int(is_test.sum()), "fit_rows": int(is_fit.sum()),
                          "val_rows": int(is_val.sum())},
               "config": {"xgb_params": PARAMS, "val_frac": VAL_FRAC, "n_boot": N_BOOT,
                          "site_radius": SITE_RADIUS, "bomb_radius": BOMB_RADIUS, "min_plants": MIN_PLANTS,
                          "maps": maps}}
    JSON_PATH.write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    out(f"wrote {JSON_PATH}")
    LOG_FH.close()


if __name__ == "__main__":
    main()
