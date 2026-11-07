"""Ablation 1 (EDA): does team health, equipment, weapon or utility state add to the coarse round state?

Reads only G: files: the 1 Hz state sample (data\\state_sample), its validation log
(logs\\eda\\state_1hz_validation.json), and the G-side basic_weapon_fire and basic_player_death
tomes, filtered to the 150 sampled matches. Nothing is read from E:. Writes
logs\\eda\\ablation_hp_equip.json and prints a log.

Design
  BASE: alive_ct, alive_t, alive_diff, bomb_planted, seconds_since_live, seconds_since_plant
        (-1 if no plant yet), round_number, score_diff (ct_score - t_score at live start).
        score_diff is a diagnostic (the score fields are mostly side-attached), so every model is
        also fit without it (BASE_NS); both sets are reported.
  Groups, per team, from the player rows alive at each snapshot:
    HP      health sum, armor sum, helmet count, count with health < 50
    EQUIP   equipment_value_calc sum, money sum (plus CT minus T equipment difference)
    WEAPON  players whose active weapon (player_vector.weapon_code) is an AWP, rifle, SMG or pistol
    UTIL    inventory counts of flashbang, smoke, HE, molotov + incendiary; defuser count
  Rounds with alive_gap_snapshots > 0 are dropped from every main result. The full round set is
  used only for the validation checks.
  Models: XGBoost hist, depth 4, eta 0.05, min_child_weight 20, subsample 0.8, seed 0,
  early stopping 100 on a match-level validation split of 15% of the train matches (seed 0).
  Primary evaluation: the split column as given (test = 30 matches, train = 120).
  Secondary: match-grouped 5-fold CV over all 150 matches (score_diff included), same model rules.

Run from the project root:
  pixi run python -I -X utf8 scripts\\eda\\ablation_hp_equip.py
"""

import bisect
import collections
import glob
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.dataset as ds
import xgboost as xgb
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"G:\pureskill\csds-cs2")
SAMPLE = ROOT / "data" / "state_sample"
TOME = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
LOG_DIR = ROOT / "logs" / "eda"
T, CT = 2, 3
TICK = 64
START = 5
SEED = 0
VAL_FRAC = 0.15
EARLY_STOP = 100
MAX_ROUNDS = 5000
N_BOOT = 2000
N_PERM = 5
MAP_SHARE_MIN = 0.95
KEYS = ["match_key", "round", "tick"]

XGB_PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist",
              "max_depth": 4, "eta": 0.05, "min_child_weight": 20, "subsample": 0.8, "seed": SEED}

BASE = ["alive_ct", "alive_t", "alive_diff", "bomb_planted", "seconds_since_live",
        "seconds_since_plant", "round_number", "score_diff"]
BASE_NS = [c for c in BASE if c != "score_diff"]
GROUPS = {
    "HP": ["ct_health", "t_health", "ct_armor", "t_armor", "ct_helmets", "t_helmets",
           "ct_below50", "t_below50"],
    "EQUIP": ["ct_equip", "t_equip", "equip_diff", "ct_money", "t_money"],
    "WEAPON": ["ct_awp", "t_awp", "ct_rifle", "t_rifle", "ct_smg", "t_smg", "ct_pistol", "t_pistol"],
    "UTIL": ["ct_flash", "t_flash", "ct_smoke", "t_smoke", "ct_he", "t_he", "ct_molinc", "t_molinc",
             "ct_defuser", "t_defuser"],
}
ALL_COLS = BASE + [c for g in GROUPS.values() for c in g]
ALL_NS_COLS = BASE_NS + [c for g in GROUPS.values() for c in g]
MODELS = [("BASE", BASE)] + [(f"BASE+{g}", BASE + GROUPS[g]) for g in GROUPS] + [("BASE+ALL", ALL_COLS)]
MODELS_NS = ([("BASE_NS", BASE_NS)] + [(f"BASE_NS+{g}", BASE_NS + GROUPS[g]) for g in GROUPS]
             + [("BASE_NS+ALL", ALL_NS_COLS)])
COL = {c: i for i, c in enumerate(ALL_COLS)}

# CS2 weapon names (weapon_fire.weapon_name) by category. SSG 08, shotguns, heavy, taser, knives,
# grenades, C4 and decoy fall in "other".
WEAPON_NAMES = {
    "awp": {"awp"},
    "rifle": {"ak47", "m4a1", "m4a1_silencer", "aug", "famas", "galilar", "g3sg1", "scar20", "sg556"},
    "smg": {"mac10", "mp9", "mp7", "mp5sd", "ump45", "p90", "bizon"},
    "pistol": {"glock", "usp_silencer", "hkp2000", "p250", "fiveseven", "tec9", "cz75a", "deagle",
               "elite", "revolver"},
}
NAME2CAT = {n: c for c, ns in WEAPON_NAMES.items() for n in ns}
# Item definition index (player_vector.weapon_code) -> category. ASSUMPTION, stated in the note:
# the code-to-name pairs are the observed ones in notes\schema_dictionary.md ("Weapon / item code
# lookup": weapon_fire names and item_* names). Codes the dictionary does not list (8, 11, 19, 23,
# 24, 26, 32, 38, 39, 63) use the standard CS2 item definition indices. Everything else is "other":
# SSG 08 (40), shotguns (25), M249 (14), zeus (31), knives (42, 59, 5xx), grenades (43-48), C4 (49).
CODE_CAT = {
    9: "awp",
    7: "rifle", 8: "rifle", 10: "rifle", 11: "rifle", 13: "rifle", 16: "rifle", 38: "rifle",
    39: "rifle", 60: "rifle",
    17: "smg", 19: "smg", 23: "smg", 24: "smg", 26: "smg", 33: "smg", 34: "smg",
    1: "pistol", 2: "pistol", 3: "pistol", 4: "pistol", 30: "pistol", 32: "pistol", 36: "pistol",
    61: "pistol", 63: "pistol", 64: "pistol",
}
AGG_FIELDS = ["alive", "health", "armor", "helmets", "below50", "equip", "money", "flash", "smoke",
              "he", "molinc", "defuser", "awp", "rifle", "smg", "pistol"]


def fail(msg):
    raise SystemExit("ERROR: " + msg)


def log(msg=""):
    print(msg, flush=True)


# ---------------------------------------------------------------- preflight

def preflight():
    log("== 0. Preflight ==")
    paths = {"players": SAMPLE / "players_1hz.parquet", "rounds": SAMPLE / "rounds.parquet",
             "validation": LOG_DIR / "state_1hz_validation.json"}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        fail("missing input(s): " + "; ".join(missing))
    val = json.loads(paths["validation"].read_text(encoding="utf-8"))
    alive, kept = val["2_alive_check"], val["kept_check"]
    checks = {
        "kept set reproduces bakeoff.log (1709/47/1521/28683/234265)":
            (kept["channel_kept"], kept["dup_matches_removed"], kept["kept_matches"],
             kept["kept_rounds"], kept["states"]) == (1709, 47, 1521, 28683, 234265),
        "alive count agreement 179457 of 179466 (extraction validation)":
            (alive["agree_inclusive_rule"], alive["snapshots"]) == (179457, 179466),
        "health and armor outside 0-100 == 0 (extraction validation)":
            (val["1_ranges"]["h_out"], val["1_ranges"]["a_out"]) == (0, 0),
        "max players per side <= 5 (extraction validation)":
            val["3_max_per_side"]["max_players_one_side_one_snapshot"] <= 5,
        "output row count 1231817 (extraction validation)":
            val["5_outputs"]["players_1hz_rows"] == 1231817,
    }
    for name, ok in checks.items():
        log(f"  {'PASS' if ok else 'FAIL'}: {name}")
    if not all(checks.values()):
        fail("extraction validation failed; see " + str(paths["validation"]))
    return val


# ---------------------------------------------------------------- snapshot grid

def build_grid(rounds):
    cols = collections.defaultdict(list)
    for r in rounds.iter_rows(named=True):
        fz, end = int(r["live_start_tick"]), int(r["round_end_tick"])
        plant = None if r["plant_tick"] is None else int(r["plant_tick"])
        score_diff = int(r["ct_score"]) - int(r["t_score"])
        for k, t in enumerate(range(fz, end, TICK)):
            planted = plant is not None and t >= plant
            cols["match_key"].append(r["match_key"])
            cols["round"].append(int(r["round"]))
            cols["tick"].append(t)
            cols["seconds_since_live"].append(k)
            cols["split"].append(r["split"])
            cols["ct_win"].append(int(r["ct_win"]))
            cols["bomb_planted"].append(int(planted))
            cols["seconds_since_plant"].append((t - plant) / TICK if planted else -1.0)
            cols["round_number"].append(int(r["round_number"]))
            cols["score_diff"].append(score_diff)
            cols["gap"].append(int(r["alive_gap_snapshots"]))
    df = pl.DataFrame(dict(cols))
    return df.with_columns(pl.col("round").cast(pl.Int32), pl.col("tick").cast(pl.Int64),
                           pl.col("seconds_since_live").cast(pl.Int64),
                           pl.col("ct_win").cast(pl.Int64), pl.col("bomb_planted").cast(pl.Int64),
                           pl.col("seconds_since_plant").cast(pl.Float64),
                           pl.col("round_number").cast(pl.Int64), pl.col("score_diff").cast(pl.Int64),
                           pl.col("gap").cast(pl.Int64))


# ---------------------------------------------------------------- weapons

def weapon_map(sample_keys, players):
    """Item definition index -> category (fixed CODE_CAT table), checked against G-side weapon_fire names.

    share = fraction of weapon_fire events carrying this code whose weapon_name falls in the same
    category. Fires are asof-merged, so a few are stale; the check requires MAP_SHARE_MIN.
    """
    folders = glob.glob(str(TOME / "basic_weapon_fire.*"))
    if len(folders) != 1:
        fail(f"expected one basic_weapon_fire folder, found {folders}")
    files = sorted(glob.glob(os.path.join(folders[0], "dataframe_*")))
    tbl = ds.dataset(files, format="parquet").to_table(
        columns=["match_key", "round", "tick", "player_id", "player_weapon_code", "weapon_name"],
        filter=ds.field("match_key").isin(sample_keys))
    fire = pl.from_arrow(tbl)
    log(f"  weapon_fire rows in sample matches: {fire.height} ({len(files)} pages, G-side tome)")
    mode_rows = (fire.filter(pl.col("player_weapon_code").is_not_null())
                 .group_by(["player_weapon_code", "weapon_name"])
                 .agg(pl.len().alias("n")).rows())
    names_of = collections.defaultdict(collections.Counter)
    for code, name, n in mode_rows:
        names_of[int(code)][str(name)] += int(n)
    used = sorted(int(c) for c in players["weapon_code"].drop_nulls().unique().to_list())
    cat_of, table = {}, []
    for c in used:
        cat = CODE_CAT.get(c, "other")
        names = names_of.get(c, collections.Counter())
        tot = sum(names.values())
        share, top = None, "(no fire events)"
        if tot:
            share = sum(n for nm, n in names.items() if NAME2CAT.get(nm, "other") == cat) / tot
            top = names.most_common(1)[0][0]
            if cat != "other" and share < MAP_SHARE_MIN:
                fail(f"item index {c} mapped to {cat} but only {share:.4f} of its weapon_fire events agree "
                     f"(< {MAP_SHARE_MIN}); top name {top}")
        cat_of[c] = cat
        table.append({"code": c, "category": cat, "fires": tot, "top_fire_name": top,
                      "share_fire_agree": None if share is None else round(share, 5),
                      "fire_names": dict(names.most_common(4))})
    wmap = pl.DataFrame({"weapon_code": list(cat_of.keys()), "wcat": list(cat_of.values())}).with_columns(
        pl.col("weapon_code").cast(pl.Int32))
    return fire, wmap, table


def fire_agreement(fire, players):
    """Active weapon code vs weapon_fire code for the same player at the same tick (exact ticks only)."""
    f = fire.select([pl.col("match_key"), pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64),
                     pl.col("player_id").cast(pl.Int64),
                     pl.col("player_weapon_code").cast(pl.Int64).alias("fire_code")]
                    ).filter(pl.col("fire_code").is_not_null())
    p = players.select([pl.col("match_key"), pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64),
                        pl.col("player_id").cast(pl.Int64), pl.col("weapon_code").cast(pl.Int64)]
                       ).filter(pl.col("weapon_code").is_not_null())
    j = p.join(f, on=["match_key", "round", "tick", "player_id"], how="inner")
    agree = j.filter(pl.col("weapon_code") == pl.col("fire_code")).height
    return {"exact_tick_pairs": j.height, "agree": agree}


# ---------------------------------------------------------------- features

def agg_exprs():
    return [
        pl.len().alias("alive"),
        pl.col("health").sum().alias("health"),
        pl.col("armor").sum().alias("armor"),
        pl.col("has_helmet").cast(pl.Int32).sum().alias("helmets"),
        (pl.col("health") < 50).cast(pl.Int32).sum().alias("below50"),
        pl.col("equipment_value_calc").sum().alias("equip"),
        pl.col("money").sum().alias("money"),
        pl.col("inv_flashbang").sum().alias("flash"),
        pl.col("inv_smokegrenade").sum().alias("smoke"),
        pl.col("inv_hegrenade").sum().alias("he"),
        (pl.col("inv_molotov") + pl.col("inv_incgrenade")).sum().alias("molinc"),
        pl.col("has_defuser").cast(pl.Int32).sum().alias("defuser"),
        (pl.col("wcat") == "awp").cast(pl.Int32).sum().alias("awp"),
        (pl.col("wcat") == "rifle").cast(pl.Int32).sum().alias("rifle"),
        (pl.col("wcat") == "smg").cast(pl.Int32).sum().alias("smg"),
        (pl.col("wcat") == "pistol").cast(pl.Int32).sum().alias("pistol"),
    ]


def build_features(grid, players_w):
    agg = players_w.group_by(KEYS + ["team_code"]).agg(agg_exprs())
    X = grid
    for team, pre in ((CT, "ct"), (T, "t")):
        sub = agg.filter(pl.col("team_code") == team).drop("team_code")
        sub = sub.rename({c: f"{pre}_{c}" for c in AGG_FIELDS})
        X = X.join(sub, on=KEYS, how="left")
    fill_cols = [f"{p}_{c}" for p in ("ct", "t") for c in AGG_FIELDS]
    X = X.with_columns([pl.col(c).fill_null(0).cast(pl.Float64) for c in fill_cols])
    X = X.with_columns([
        pl.col("ct_alive").alias("alive_ct"),
        pl.col("t_alive").alias("alive_t"),
        (pl.col("ct_alive") - pl.col("t_alive")).alias("alive_diff"),
        (pl.col("ct_equip") - pl.col("t_equip")).alias("equip_diff"),
    ])
    return X


def death_check(X, rounds, sample_keys):
    """Independent alive check against G-side basic_player_death (inclusive rule, bake-off side)."""
    folders = glob.glob(str(TOME / "basic_player_death.*"))
    if len(folders) != 1:
        fail(f"expected one basic_player_death folder, found {folders}")
    files = sorted(glob.glob(os.path.join(folders[0], "dataframe_*")))
    tbl = ds.dataset(files, format="parquet").to_table(
        columns=["match_key", "round", "tick", "player_id", "player_team_code"],
        filter=ds.field("match_key").isin(sample_keys))
    d = pl.from_arrow(tbl).with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64))
    win = rounds.select(["match_key", pl.col("round").cast(pl.Int64), "live_start_tick", "round_end_tick"])
    d = d.join(win, on=["match_key", "round"], how="inner").filter(
        (pl.col("tick") >= pl.col("live_start_tick")) & (pl.col("tick") <= pl.col("round_end_tick")))
    first = (d.sort("tick").group_by(["match_key", "round", "player_id"], maintain_order=True)
             .agg(pl.col("tick").first(), pl.col("player_team_code").first()))
    tT, tCT = collections.defaultdict(list), collections.defaultdict(list)
    unresolved = 0
    for r in first.iter_rows(named=True):
        key = (r["match_key"], int(r["round"]))
        side = r["player_team_code"]
        if side is None:
            unresolved += 1
            continue
        (tT if side == T else tCT)[key].append(int(r["tick"]))
    for dct in (tT, tCT):
        for v in dct.values():
            v.sort()
    agree, mism = 0, []
    for m, r, t, ca, ta in X.select(["match_key", "round", "tick", "ct_alive", "t_alive"]).iter_rows():
        key = (m, int(r))
        e_t = START - bisect.bisect_right(tT.get(key, []), t)
        e_ct = START - bisect.bisect_right(tCT.get(key, []), t)
        if (int(ta), int(ca)) == (e_t, e_ct):
            agree += 1
        else:
            mism.append((m, int(r), int(t), int(ca), int(ta), e_ct, e_t))
    return {"snapshots": X.height, "agree": agree, "unresolved_death_sides": unresolved,
            "mismatch_rounds": sorted({f"{m}|r{r}" for m, r, *_ in mism}),
            "first_mismatches": [dict(zip(["match_key", "round", "tick", "table_ct", "table_t",
                                           "death_ct", "death_t"], x)) for x in mism[:10]]}


# ---------------------------------------------------------------- modelling

def split_inner(matches, seed):
    ms = sorted(matches)
    random.Random(seed).shuffle(ms)
    return set(ms[:round(VAL_FRAC * len(ms))])


def mets(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return {"n": int(len(y)), "logloss": float(log_loss(y, p, labels=[0, 1])),
            "brier": float(brier_score_loss(y, p)), "auc": float(roc_auc_score(y, p))}


def fit_eval(M, Y, fit_m, val_m, test_m, models=None):
    out = {}
    for name, cols in (MODELS if models is None else models):
        idx = [COL[c] for c in cols]
        dfit = xgb.DMatrix(M[fit_m][:, idx], label=Y[fit_m], feature_names=cols)
        dval = xgb.DMatrix(M[val_m][:, idx], label=Y[val_m], feature_names=cols)
        bst = xgb.train(XGB_PARAMS, dfit, num_boost_round=MAX_ROUNDS, evals=[(dval, "val")],
                        early_stopping_rounds=EARLY_STOP, verbose_eval=False)
        p = bst.predict(xgb.DMatrix(M[test_m][:, idx], feature_names=cols),
                        iteration_range=(0, bst.best_iteration + 1))
        out[name] = {"bst": bst, "p": p, "best_iter": int(bst.best_iteration),
                     "val_logloss": float(bst.best_score), "cols": cols}
    return out


def ll_rows(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def group_index(match_arr):
    uniq = sorted(set(match_arr))
    pos = {m: i for i, m in enumerate(uniq)}
    return np.array([pos[m] for m in match_arr], dtype=np.int64), len(uniq)


def gain_boot(y, p_base, p_mod, gid, n_groups):
    """Cluster bootstrap over matches: gain = logloss(base) - logloss(model); positive = model helps."""
    lb, lm = ll_rows(y, p_base), ll_rows(y, p_mod)
    S_b = np.bincount(gid, weights=lb, minlength=n_groups)
    S_m = np.bincount(gid, weights=lm, minlength=n_groups)
    N = np.bincount(gid, minlength=n_groups).astype(np.float64)
    point = (S_b.sum() - S_m.sum()) / N.sum()
    rng = np.random.default_rng(SEED)
    W = rng.multinomial(n_groups, np.full(n_groups, 1.0 / n_groups), size=N_BOOT).astype(np.float64)
    gains = (W @ (S_b - S_m)) / (W @ N)
    lo, hi = np.percentile(gains, [2.5, 97.5])
    return {"gain": float(point), "ci95": [float(lo), float(hi)],
            "share_boot_positive": float((gains > 0).mean())}


def perm_importance(bst, Mte, yte):
    def ll_of(Mx):
        p = bst.predict(xgb.DMatrix(Mx, feature_names=ALL_COLS), iteration_range=(0, bst.best_iteration + 1))
        return float(log_loss(yte, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1]))

    base = ll_of(Mte)
    feats, groups = {}, {}
    for c in ALL_COLS:
        j = COL[c]
        deltas = []
        for r in range(N_PERM):
            perm = np.random.default_rng(10_000 + 100 * j + r).permutation(len(yte))
            Mx = Mte.copy()
            Mx[:, j] = Mte[perm, j]
            deltas.append(ll_of(Mx) - base)
        feats[c] = {"mean": float(np.mean(deltas)), "sd": float(np.std(deltas, ddof=1))}
    for g, cols in [("BASE", BASE)] + list(GROUPS.items()):
        idx = [COL[c] for c in cols]
        deltas = []
        for r in range(N_PERM):
            perm = np.random.default_rng(20_000 + 100 * len(idx) + r).permutation(len(yte))
            Mx = Mte.copy()
            Mx[:, idx] = Mte[perm][:, idx]
            deltas.append(ll_of(Mx) - base)
        groups[g] = {"mean": float(np.mean(deltas)), "sd": float(np.std(deltas, ddof=1))}
    return {"baseline_logloss": base, "features": feats, "groups_joint": groups}


def metric_table(fits, models, Y, is_fit, yte, is_test):
    const_p = np.full(len(yte), Y[is_fit].mean())
    prim = {"CONST (fit-split CT-win share)": {**mets(yte, const_p), "best_iter": None, "val_logloss": None}}
    for name, _ in models:
        f = fits[name]
        prim[name] = {**mets(yte, f["p"]), "best_iter": f["best_iter"], "val_logloss": f["val_logloss"],
                      "n_features": len(f["cols"])}
    return prim


def log_table(prim):
    for name, row in prim.items():
        log(f"  {name:<30} test logloss {row['logloss']:.5f}  brier {row['brier']:.5f}  auc {row['auc']:.4f}  "
            f"best_iter {row.get('best_iter')}  val_logloss {row.get('val_logloss')}")


def main():
    t0 = time.perf_counter()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    val = preflight()

    log("== 1. Inputs ==")
    players = pl.read_parquet(SAMPLE / "players_1hz.parquet")
    rounds = pl.read_parquet(SAMPLE / "rounds.parquet")
    log(f"  players_1hz rows {players.height}; rounds {rounds.height}")
    assert rounds.height == 2673 and players.height == 1231817
    rs = rounds.group_by("split").len().sort("split").rows()
    log(f"  rounds by split: {rs}")
    if dict(rs) != {"test": 488, "train": 2185}:
        fail(f"round split counts differ from the extraction notes: {rs}")
    ms = rounds.group_by("split").agg(pl.col("match_key").n_unique().alias("m")).sort("split").rows()
    log(f"  matches by split (rounds table): {ms}")
    if dict(ms) != {"test": 30, "train": 120}:
        fail(f"match split counts are not 30 test / 120 train: {ms}")
    sample_keys = sorted(rounds["match_key"].unique().to_list())
    if len(sample_keys) != 150:
        fail(f"expected 150 sampled matches, found {len(sample_keys)}")
    gap_rounds = (rounds.filter(pl.col("alive_gap_snapshots") > 0)
                  .select(["match_key", "round", "split", "alive_gap_snapshots"]).rows())
    if len(gap_rounds) != 1:
        fail("expected exactly one round with alive_gap_snapshots > 0")
    log(f"  rounds with alive_gap_snapshots > 0 (dropped from main results): {gap_rounds}")

    grid = build_grid(rounds)
    gs = grid.group_by("split").len().sort("split").rows()
    log(f"  snapshot grid (all clean rounds): {grid.height} snapshots by split {gs}")
    if grid.height != 179466 or dict(gs) != {"test": 32884, "train": 146582}:
        fail("snapshot grid counts differ from the extraction notes")

    log("== 2. Weapon codes (G-side weapon_fire lookup) ==")
    fire, wmap, wtable = weapon_map(sample_keys, players)
    for row in wtable:
        log(f"  item index {row['code']:>4} -> {row['category']:<7} weapon_fire events {row['fires']:>6}  "
            f"agree {row['share_fire_agree']}  top {row['top_fire_name']}")
    players_w = players.join(wmap, on="weapon_code", how="left").with_columns(
        pl.col("wcat").fill_null("other"))
    agreement = fire_agreement(fire, players)
    log(f"  active weapon vs weapon_fire code at the same tick: {agreement}")
    out["weapon_map"] = wtable
    out["weapon_fire_agreement"] = agreement

    log("== 3. Snapshot features (all clean rounds, for validation) ==")
    X_full = build_features(grid, players_w)
    log(f"  feature table rows {X_full.height} (grid {grid.height})")
    if X_full.height != grid.height:
        fail("feature join changed the row count")
    pk = players.select(KEYS + ["seconds_since_live"]).unique()
    unmatched = pk.join(grid.select(KEYS), on=KEYS, how="anti").height
    sec_bad = (pk.join(grid.select(KEYS + ["seconds_since_live"]), on=KEYS, how="inner")
               .filter(pl.col("seconds_since_live") != pl.col("seconds_since_live_right")).height)
    log(f"  player key rows not on the grid: {unmatched}; seconds_since_live mismatches: {sec_bad}")
    side_max = int(max(X_full["ct_alive"].max(), X_full["t_alive"].max()))
    zero_side = int(((X_full["ct_alive"] == 0) | (X_full["t_alive"] == 0)).sum())
    log(f"  max alive per side: {side_max}; snapshots with 0 alive on a side: {zero_side}")
    if unmatched or sec_bad or side_max > 5:
        fail("grid or count consistency check failed")
    dc = death_check(X_full, rounds, sample_keys)
    log(f"  alive count vs G-side player_death (inclusive rule): agree {dc['agree']} of {dc['snapshots']}; "
        f"unresolved death sides {dc['unresolved_death_sides']}; mismatched rounds {dc['mismatch_rounds']}")
    out["checks"] = {"grid_rows": grid.height, "player_keys_off_grid": unmatched,
                     "seconds_mismatch": sec_bad, "max_alive_per_side": side_max,
                     "snapshots_with_zero_alive_side": zero_side, "death_check": dc}

    log("== 3b. Main-results sample: alive-gap round dropped ==")
    drop = X_full.filter(pl.col("gap") > 0)
    X = X_full.filter(pl.col("gap") == 0)
    drop_rounds = drop.select(["match_key", "round", "split"]).unique().rows()
    if len(drop_rounds) != 1 or drop.height == 0:
        fail(f"expected one dropped round with snapshots, found {drop_rounds}")
    log(f"  dropped {len(drop_rounds)} round(s), {drop.height} snapshots: {drop_rounds}")
    mb = X.group_by("split").agg(pl.col("match_key").n_unique().alias("m")).sort("split").rows()
    rb = X.select(["match_key", "round", "split"]).unique().group_by("split").len().sort("split").rows()
    if dict(mb) != {"test": 30, "train": 120}:
        fail(f"main sample match split counts differ: {mb}")
    log(f"  main sample: {X.height} snapshots; matches by split {mb}; rounds by split {rb}")
    out["main_sample"] = {"dropped_rounds": drop_rounds, "dropped_snapshots": drop.height,
                          "snapshots": X.height, "matches_by_split": dict(mb), "rounds_by_split": dict(rb)}

    summ = X.select([pl.col(c).mean().alias(c) for c in ALL_COLS]).row(0, named=True)
    out["feature_means"] = {k: round(float(v), 4) for k, v in summ.items()}

    log("== 4. Splits and matrices ==")
    M = X.select(ALL_COLS).to_numpy().astype(np.float64)
    Y = X["ct_win"].to_numpy().astype(np.float64)
    split = X["split"].to_numpy()
    mk = X["match_key"].to_numpy()
    is_test = split == "test"
    is_train = split == "train"
    train_matches = sorted(set(mk[is_train]))
    val_primary = split_inner(train_matches, SEED)
    is_val = is_train & np.isin(mk, list(val_primary))
    is_fit = is_train & ~is_val
    assert not (set(mk[is_fit]) & set(mk[is_val])), "match on both fit and validation sides"
    assert not (set(mk[is_test]) & set(mk[is_train])), "match on both train and test"
    log(f"  fit: {is_fit.sum()} snapshots, {len(set(mk[is_fit]))} matches; "
        f"validation: {is_val.sum()} snapshots, {len(val_primary)} matches; "
        f"test: {is_test.sum()} snapshots, {len(set(mk[is_test]))} matches")
    log(f"  CT-win share: fit {Y[is_fit].mean():.4f}, validation {Y[is_val].mean():.4f}, "
        f"test {Y[is_test].mean():.4f}")
    out["split"] = {"fit_snapshots": int(is_fit.sum()), "val_snapshots": int(is_val.sum()),
                    "test_snapshots": int(is_test.sum()), "test_matches": len(set(mk[is_test])),
                    "val_matches": len(val_primary), "fit_matches": len(set(mk[is_fit])),
                    "ct_win_share_test": float(Y[is_test].mean()),
                    "ct_win_share_fit": float(Y[is_fit].mean())}

    log("== 5. Primary fits (split column as given; score_diff included) ==")
    fits = fit_eval(M, Y, is_fit, is_val, is_test)
    yte = Y[is_test]
    mte = mk[is_test]
    gid, n_g = group_index(list(mte))
    prim = metric_table(fits, MODELS, Y, is_fit, yte, is_test)
    out["primary_test"] = prim
    log_table(prim)

    log("  bootstrap over test matches (gain = logloss(BASE) - logloss(model), positive = helps):")
    boots = {}
    for name, _ in MODELS[1:]:
        boots[name] = gain_boot(yte, fits["BASE"]["p"], fits[name]["p"], gid, n_g)
        b = boots[name]
        log(f"    {name:<12} gain {b['gain']:+.5f}  95% CI [{b['ci95'][0]:+.5f}, {b['ci95'][1]:+.5f}]  "
            f"P(gain>0) {b['share_boot_positive']:.3f}")
    out["primary_bootstrap_gain_vs_base"] = boots

    log("== 5b. Sensitivity: score_diff removed from every model ==")
    fits_ns = fit_eval(M, Y, is_fit, is_val, is_test, models=MODELS_NS)
    prim_ns = metric_table(fits_ns, MODELS_NS, Y, is_fit, yte, is_test)
    out["primary_test_no_score_diff"] = prim_ns
    log_table(prim_ns)
    boots_ns = {}
    for name, _ in MODELS_NS[1:]:
        boots_ns[name] = gain_boot(yte, fits_ns["BASE_NS"]["p"], fits_ns[name]["p"], gid, n_g)
        b = boots_ns[name]
        log(f"    {name:<14} gain {b['gain']:+.5f}  95% CI [{b['ci95'][0]:+.5f}, {b['ci95'][1]:+.5f}]  "
            f"P(gain>0) {b['share_boot_positive']:.3f}")
    out["primary_bootstrap_gain_vs_base_no_score_diff"] = boots_ns

    bst_all = fits["BASE+ALL"]["bst"]
    imp = bst_all.get_score(importance_type="gain")
    tot = sum(imp.values())
    gain_rows = sorted(((c, float(imp.get(c, 0.0)), float(imp.get(c, 0.0) / tot)) for c in ALL_COLS),
                       key=lambda x: -x[1])
    grp_gain = {g: float(sum(imp.get(c, 0.0) for c in cols) / tot)
                for g, cols in [("BASE", BASE)] + list(GROUPS.items())}
    out["gain_importance_all_groups"] = [{"feature": c, "gain": g, "share": s} for c, g, s in gain_rows]
    out["gain_share_by_group"] = grp_gain
    log("  gain importance (BASE+ALL), top 15:")
    for c, g, s in gain_rows[:15]:
        log(f"    {c:<22} gain {g:12.1f}  share {s:.4f}")
    log(f"  gain share by group: {{{', '.join(f'{k}: {v:.3f}' for k, v in grp_gain.items())}}}")

    perm = perm_importance(bst_all, M[is_test], yte)
    out["permutation_importance_test"] = perm
    log(f"  permutation importance on test (log loss increase, {N_PERM} repeats), BASE+ALL, baseline "
        f"{perm['baseline_logloss']:.5f}:")
    for c in sorted(ALL_COLS, key=lambda c: -perm["features"][c]["mean"]):
        f = perm["features"][c]
        log(f"    {c:<22} {f['mean']:+.5f} (sd {f['sd']:.5f})")
    for g, f in perm["groups_joint"].items():
        log(f"    [group jointly] {g:<8} {f['mean']:+.5f} (sd {f['sd']:.5f})")

    log("== 6. Secondary: match-grouped 5-fold CV over all 150 matches (score_diff included) ==")
    all_m = sorted(set(mk))
    rng = random.Random(SEED + 2)
    shuffled = list(all_m)
    rng.shuffle(shuffled)
    folds = [set(shuffled[i::5]) for i in range(5)]
    oof = {name: np.full(len(Y), np.nan) for name, _ in MODELS}
    fold_ll = collections.defaultdict(list)
    const_oof = np.full(len(Y), np.nan)
    for f_i, fold in enumerate(folds):
        test_m = np.isin(mk, list(fold))
        pool = ~test_m
        val_f = split_inner(set(mk[pool]), SEED + 10 + f_i)
        val_m = pool & np.isin(mk, list(val_f))
        fit_m = pool & ~val_m
        fe = fit_eval(M, Y, fit_m, val_m, test_m)
        const_oof[test_m] = Y[fit_m].mean()
        for name, _ in MODELS:
            oof[name][test_m] = fe[name]["p"]
            fold_ll[name].append(mets(Y[test_m], fe[name]["p"])["logloss"])
        log(f"  fold {f_i}: test matches {len(fold)}, snapshots {test_m.sum()}; "
            f"BASE ll {fold_ll['BASE'][-1]:.5f}, BASE+ALL ll {fold_ll['BASE+ALL'][-1]:.5f}")
    cv = {"CONST": {**mets(Y, const_oof), "fold_logloss_sd": None}}
    for name, _ in MODELS:
        cv[name] = {**mets(Y, oof[name]), "fold_logloss_mean": float(np.mean(fold_ll[name])),
                    "fold_logloss_sd": float(np.std(fold_ll[name], ddof=1))}
    gid_all, n_all = group_index(list(mk))
    cv_boot = {}
    for name, _ in MODELS[1:]:
        cv_boot[name] = gain_boot(Y, oof["BASE"], oof[name], gid_all, n_all)
    out["cv_150_matches"] = {"metrics": cv, "bootstrap_gain_vs_base": cv_boot}
    for name in ["CONST"] + [m for m, _ in MODELS]:
        r = cv[name]
        log(f"  {name:<12} pooled OOF logloss {r['logloss']:.5f}  brier {r['brier']:.5f}  auc {r['auc']:.4f}  "
            f"fold sd {r['fold_logloss_sd'] if r['fold_logloss_sd'] is None else round(r['fold_logloss_sd'], 5)}")
    for name, b in cv_boot.items():
        log(f"    CV gain {name:<12} {b['gain']:+.5f}  95% CI [{b['ci95'][0]:+.5f}, {b['ci95'][1]:+.5f}]  "
            f"P(gain>0) {b['share_boot_positive']:.3f}")

    out["wall_s"] = round(time.perf_counter() - t0, 1)
    out["config"] = {"xgb_params": XGB_PARAMS, "early_stopping_rounds": EARLY_STOP,
                     "val_frac": VAL_FRAC, "n_boot": N_BOOT, "n_perm": N_PERM,
                     "map_share_min": MAP_SHARE_MIN, "groups": GROUPS, "base": BASE, "base_ns": BASE_NS,
                     "drop_alive_gap_rounds": True}
    path = LOG_DIR / "ablation_hp_equip.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)
    log(f"wrote {path} (wall {out['wall_s']} s)")


if __name__ == "__main__":
    main()
