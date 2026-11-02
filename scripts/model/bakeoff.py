"""Round win-probability model bake-off on the 2026-10-01 to 08 basic tomes (seed 0).

Pipeline
  1. Header and channel loads. read() and the tome constants come from
     scripts/model/feature_validation.py, loaded by path. The state-building loop
     in its main() is ported here unchanged in logic (it is not importable).
  2. Match filter. A match is dropped if a required channel is missing:
     - every required channel: the match key is absent from that channel's keyset;
     - dense channels (every complete match has rows): zero rows also counts as missing;
     - sparse event channels (player_disconnect, bot_takeover): zero rows is a normal
       outcome, so only keyset membership applies.
     Duplicate matches: two or more kept matches with identical round_end, player_death
     and player_info events (the same demo processed twice under two keys). The lowest key
     is kept and the others are dropped. This runs after the round filters and before the
     splits, so no copy can land on the other side of a split.
  3. Round filters, applied in this order, each dropped round counted once:
     player_count != 10; not exactly 5 T and 5 CT at the start (player_info);
     disconnect or bot takeover anywhere in the round; no round_start row;
     no round_freeze_end (live start); null score at the live start;
     a death in the live window whose victim side cannot be resolved.
  4. States per kept round: the live start, plus one state per in-window death
     (first death per victim). Same rules as feature_validation.py.
  5. Splits.
     PRIMARY: match-level random. 20% of matches are test. 10% of the remaining
     matches are validation (early stopping and calibration). The rest train.
     SECONDARY: chronological by header match_date. The earliest 80% form the train
     pool and the latest 20% are test. The latest 10% of the train pool are
     validation, the earliest 90% of it train.
  6. Models: A constant, B logistic on ct_alive_minus_t, logistic on all features,
     XGBoost, LightGBM, HistGradientBoosting (monotone constraints as specified).
     XGBoost and LightGBM early stop on validation. HistGB uses its internal early
     stopping (no eval_set in sklearn). Sigmoid and isotonic calibration of the last
     three, fitted on validation only. For model selection, calibrated validation
     scores are out-of-fold (2 folds, grouped by match inside validation). The
     in-sample validation scores are also reported and flagged.
  7. Metrics on states: log loss, Brier, AUC, ECE (10 equal-width bins), reliability
     table, accuracy (reference only).
  8. Outputs: notes\\model\\bakeoff_results.md and logs\\bakeoff.log. A model file is
     saved to models\\ only if the best model by primary validation log loss is
     calibrated. No data is saved.

Run from the project root:
  pixi run python -I -X utf8 scripts\\model\\bakeoff.py
"""

import collections
import datetime as dt
import glob
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = ROOT / "logs" / "bakeoff.log"
OUT_MD = ROOT / "notes" / "model" / "bakeoff_results.md"
MODEL_DIR = ROOT / "models"

_spec = importlib.util.spec_from_file_location(
    "feature_validation", ROOT / "scripts" / "model" / "feature_validation.py")
fv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fv)

T, CT, START, TICK_RATE = fv.T, fv.CT, fv.START, fv.TICK_RATE
SEED = 0
TEST_FRAC = 0.20
VAL_FRAC = 0.10
CHRONO_TRAIN_FRAC = 0.80
N_BINS = 10
DENSE = ["round_end", "round_start", "round_state", "player_death", "player_info",
         "bomb_state", "team_change"]
SPARSE = ["player_disconnect", "bot_takeover"]
FEATS = ["alive_ct", "alive_t", "ct_alive_minus_t", "bomb_planted", "seconds_since_round_live",
         "seconds_since_plant", "round_number", "ct_score_minus_t_score", "is_pistol_round",
         "is_overtime"]
MONO_MAP = {"alive_ct": 1, "alive_t": -1, "ct_alive_minus_t": 1, "ct_score_minus_t_score": 1,
            "seconds_since_plant": 0}
MONO = [MONO_MAP.get(f, 0) for f in FEATS]
PRIMARY = "primary"
CHRONO = "chronological"
MODEL_NAMES = ["A constant", "B LR ct_alive_minus_t", "LR all features", "XGBoost", "LightGBM",
               "HistGB"]
CAL_BASES = ["XGBoost", "LightGBM", "HistGB"]


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def keyset_keys(ch):
    folder = fv.TOME / f"basic_{ch}{fv.SUFFIX}"
    files = sorted(glob.glob(str(folder / "keyset_*")))
    if not files:
        raise SystemExit(f"no keyset files in {folder}")
    return set(pl.concat([pl.read_parquet(f) for f in files])["_"].to_list())


def parse_date(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def swap_expected(r, max_rounds):
    """Sides swap after completed round r: at the halftime round, and after the 3rd round of
    each overtime period (6-round periods, MR3)."""
    return r == max_rounds // 2 or (r > max_rounds and (r - max_rounds - 3) % 6 == 0)


def ece_and_table(y, p):
    idx = np.minimum((p * N_BINS).astype(int), N_BINS - 1)
    n_total = len(y)
    ece = 0.0
    rows = []
    for b in range(N_BINS):
        m = idx == b
        n = int(m.sum())
        lo, hi = b / N_BINS, (b + 1) / N_BINS
        if n:
            mp, my = float(p[m].mean()), float(y[m].mean())
            ece += n / n_total * abs(my - mp)
        else:
            mp, my = float("nan"), float("nan")
        rows.append((lo, hi, n, mp, my))
    return float(ece), rows


def metrics(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    ece, table = ece_and_table(y, p)
    return {"n": int(len(y)),
            "logloss": float(log_loss(y, p, labels=[0, 1])),
            "brier": float(brier_score_loss(y, p)),
            "auc": float(roc_auc_score(y, p)),
            "ece": ece,
            "acc": float(((p >= 0.5).astype(int) == y).mean()),
            "table": table}


def val_fold_ids(val_matches):
    ms = sorted(val_matches)
    rng = random.Random(SEED + 1)
    rng.shuffle(ms)
    return {m: i % 2 for i, m in enumerate(ms)}


def build_states(key, rr, deaths, plants, players, max_rounds, diag):
    """Ported from scripts/model/feature_validation.py main(). Returns the state dicts for one round,
    or None if a death in the live window has an unresolved victim side."""
    frz, end = rr["freeze"], rr["end"]
    win, unresolved = [], 0
    for d in deaths:
        if d["tick"] < frz:
            diag["deaths_pre_live"] += 1
            continue
        if d["tick"] > end:
            diag["deaths_post_end"] += 1
            continue
        side = d["team"]
        if side is None:
            diag["in_window_null_team"] += 1
            side = players.get(d["pid"])
            if side is not None:
                diag["in_window_null_team_imputed"] += 1
        if side not in (T, CT):
            unresolved += 1
            continue
        win.append({**d, "side": side})
    if unresolved:
        return None
    diag["deaths_in_window"] += len(win)
    in_win = [t for t in plants if frz <= t <= end]
    diag["plants_all"] += len(plants)
    diag["plants_in_window"] += len(in_win)
    diag["plants_after_end"] += sum(1 for t in plants if t > end)
    diag["plants_before_live"] += sum(1 for t in plants if t < frz)
    if len(plants) > 1:
        diag["rounds_with_multiple_plants"] += 1
    plant_tick = min(in_win) if in_win else None
    alive = {T: START, CT: START}
    states = [{"tick": frz, "alive_t": START, "alive_ct": START, "kind": "live_start"}]
    seen = set()
    for d in win:
        if d["pid"] in seen:
            diag["in_window_duplicate_victim_skipped"] += 1
            continue
        seen.add(d["pid"])
        alive[d["side"]] -= 1
        states.append({"tick": d["tick"], "alive_t": alive[T], "alive_ct": alive[CT],
                       "kind": "death"})
    out = []
    for s in states:
        planted = int(plant_tick is not None and plant_tick <= s["tick"])
        out.append({
            "match_key": key[0], "round": key[1], "tick": s["tick"], "kind": s["kind"],
            "alive_ct": s["alive_ct"], "alive_t": s["alive_t"],
            "ct_alive_minus_t": s["alive_ct"] - s["alive_t"],
            "bomb_planted": planted,
            "seconds_since_round_live": (s["tick"] - frz) / TICK_RATE,
            "seconds_since_plant": ((s["tick"] - plant_tick) / TICK_RATE) if planted else -1.0,
            "round_number": key[1],
            "ct_score_minus_t_score": rr["ct_score"] - rr["t_score"],
            "is_pistol_round": int(key[1] in (1, max_rounds // 2 + 1)),
            "is_overtime": int(key[1] > max_rounds),
            "label_ct_win": int(rr["winner"] == CT),
        })
    return out


def score_checks(RE, RS, H):
    frz = {}
    for x in RS.filter(pl.col("event_type") == "round_freeze_end").iter_rows(named=True):
        frz[(x["match_key"], int(x["round"]))] = (int(x["t_score"]), int(x["ct_score"]))
    info = {}
    for x in RE.iter_rows(named=True):
        k = (x["match_key"], int(x["round"]))
        if k in frz:
            info[k] = {"winner": x["winner_team_code"], "t": frz[k][0], "ct": frz[k][1],
                       "max": H[k[0]]["max_rounds"]}
    sum_ok = sum(1 for (m, r), v in info.items() if v["t"] + v["ct"] == r - 1)
    counts = collections.Counter()
    bad_rounds = collections.Counter()
    n_sched_swaps = 0
    n_sched_swaps_consistent = 0
    for (m, r), a in info.items():
        b = info.get((m, r + 1))
        if b is None:
            counts["no_next_round_with_freeze"] += 1
            continue
        cw, tw = int(a["winner"] == CT), int(a["winner"] == T)
        t_post, ct_post = a["t"] + tw, a["ct"] + cw
        sw = swap_expected(r, a["max"])
        pred = (ct_post, t_post) if sw else (t_post, ct_post)
        ok = pred == (b["t"], b["ct"])
        if sw:
            n_sched_swaps += 1
            n_sched_swaps_consistent += int(ok)
        if ok:
            counts["consistent"] += 1
        else:
            counts["inconsistent"] += 1
            bad_rounds[(r, a["max"], sw)] += 1
    return {"rounds_with_freeze": len(info), "sum_eq_r_minus_1": sum_ok,
            "transitions": dict(counts), "schedule_swap_transitions": n_sched_swaps,
            "schedule_swap_consistent": n_sched_swaps_consistent,
            "inconsistent_by_round": sorted(bad_rounds.items())}


def completion_checks(RE, H):
    n_by = RE.group_by("match_key").agg(pl.len().alias("n")).iter_rows()
    n_by = {m: n for m, n in n_by}
    stats = collections.Counter()
    examples = collections.defaultdict(list)
    for m, h in H.items():
        n = n_by.get(m, 0)
        final_sum = h["t_starters_score_final"] + h["ct_starters_score_final"]
        mr = h["max_rounds"]
        max_final = max(h["t_starters_score_final"], h["ct_starters_score_final"])
        completed = max_final >= mr // 2 + 1 or n > mr
        if n == final_sum:
            stats["n_rounds_equals_header_final_sum"] += 1
        elif n < final_sum:
            stats["n_rounds_below_header_final_sum"] += 1
            if len(examples["below"]) < 3:
                examples["below"].append(f"{m[:40]} n={n} header_sum={final_sum} max_rounds={mr}")
        else:
            stats["n_rounds_above_header_final_sum"] += 1
            if len(examples["above"]) < 3:
                examples["above"].append(f"{m[:40]} n={n} header_sum={final_sum} max_rounds={mr}")
        if n < mr:
            stats["matches_below_max_rounds"] += 1
            if completed:
                stats["below_max_rounds_completed_by_threshold"] += 1
            else:
                stats["below_max_rounds_not_completed"] += 1
                if len(examples["notcomp"]) < 5:
                    examples["notcomp"].append(
                        f"{m[:40]} n={n} header_scores={h['t_starters_score_final']}-"
                        f"{h['ct_starters_score_final']} max_rounds={mr}")
        if not completed:
            stats["not_completed_overall"] += 1
    return stats, examples


def make_splits(matches, H):
    keys = sorted(matches)
    shuffled = list(keys)
    random.Random(SEED).shuffle(shuffled)
    n_test = round(TEST_FRAC * len(keys))
    test, pool = shuffled[:n_test], shuffled[n_test:]
    n_val = round(VAL_FRAC * len(pool))
    val, train = pool[:n_val], pool[n_val:]
    prim = {"train": set(train), "val": set(val), "test": set(test)}
    ordered = sorted(keys, key=lambda k: (parse_date(H[k]["match_date"]), k))
    n_pool = round(CHRONO_TRAIN_FRAC * len(ordered))
    pool_c, test_c = ordered[:n_pool], ordered[n_pool:]
    n_val_c = round(VAL_FRAC * len(pool_c))
    train_c, val_c = pool_c[:len(pool_c) - n_val_c], pool_c[len(pool_c) - n_val_c:]
    chrono = {"train": set(train_c), "val": set(val_c), "test": set(test_c)}
    for name, sp in ((PRIMARY, prim), (CHRONO, chrono)):
        assert not (sp["train"] & sp["val"]) and not (sp["train"] & sp["test"]) \
            and not (sp["val"] & sp["test"]), f"{name}: a match is on two sides of a split"
        assert sp["train"] | sp["val"] | sp["test"] == set(keys), f"{name}: split does not cover all matches"
    return {PRIMARY: prim, CHRONO: chrono}


def xy(df):
    X = df.select(FEATS).to_numpy().astype(np.float64)
    y = df["label_ct_win"].to_numpy().astype(np.int64)
    return X, y


def fit_models(tr, va, te, fold_va):
    Xtr, ytr = xy(tr)
    Xva, yva = xy(va)
    Xte, yte = xy(te)
    res = {}
    objs = {}
    rate = float(ytr.mean())
    res["A constant"] = {"pva": np.full(len(yva), rate), "pte": np.full(len(yte), rate),
                         "info": f"train CT-win rate {rate:.4f}"}
    ci = FEATS.index("ct_alive_minus_t")
    mB = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr[:, [ci]], ytr)
    res["B LR ct_alive_minus_t"] = {"pva": mB.predict_proba(Xva[:, [ci]])[:, 1],
                                    "pte": mB.predict_proba(Xte[:, [ci]])[:, 1],
                                    "info": "standardised, C=1.0"}
    mL = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr)
    res["LR all features"] = {"pva": mL.predict_proba(Xva)[:, 1], "pte": mL.predict_proba(Xte)[:, 1],
                              "info": "standardised, C=1.0, 10 features"}

    mX = xgb.XGBClassifier(n_estimators=3000, learning_rate=0.05, max_depth=4, min_child_weight=20,
                           subsample=0.8, tree_method="hist", device="cpu",
                           monotone_constraints=tuple(MONO), early_stopping_rounds=100,
                           eval_metric="logloss", random_state=SEED)
    mX.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    bx = int(mX.best_iteration)
    pva_x = mX.predict_proba(Xva, iteration_range=(0, bx + 1))[:, 1]
    default_x = np.allclose(mX.predict_proba(Xva)[:, 1], pva_x)
    res["XGBoost"] = {"pva": pva_x,
                      "pte": mX.predict_proba(Xte, iteration_range=(0, bx + 1))[:, 1],
                      "info": f"hist, depth 4, eta 0.05, best_iteration {bx}; "
                              f"default predict uses same trees: {default_x}"}
    objs["XGBoost"] = mX

    mG = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.05, num_leaves=15,
                            min_child_samples=50, monotone_constraints=MONO, random_state=SEED,
                            deterministic=True, force_row_wise=True, verbose=-1)
    mG.fit(Xtr, ytr, eval_X=Xva, eval_y=yva, eval_metric="binary_logloss",
           callbacks=[lgb.early_stopping(100, verbose=False)])
    bg = int(mG.best_iteration_)
    pva_g = mG.predict_proba(Xva, num_iteration=bg)[:, 1]
    default_g = np.allclose(mG.predict_proba(Xva)[:, 1], pva_g)
    res["LightGBM"] = {"pva": pva_g,
                       "pte": mG.predict_proba(Xte, num_iteration=bg)[:, 1],
                       "info": f"leaves 15, lr 0.05, best_iteration {bg}; eval_X/eval_y; "
                               f"default predict uses same trees: {default_g}"}
    objs["LightGBM"] = mG

    mH = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=3000, max_leaf_nodes=15,
                                        min_samples_leaf=50, monotonic_cst=MONO,
                                        early_stopping=True, validation_fraction=0.1,
                                        n_iter_no_change=100, scoring="loss", random_state=SEED)
    mH.fit(Xtr, ytr)
    res["HistGB"] = {"pva": mH.predict_proba(Xva)[:, 1], "pte": mH.predict_proba(Xte)[:, 1],
                     "info": f"leaves 15, lr 0.05, internal early stopping (random 10% of "
                             f"training rows), n_iter {int(mH.n_iter_)}"}
    objs["HistGB"] = mH
    return res, objs, (Xva, yva, Xte, yte)


def calibrate_base(base, model, Xva, yva, Xte, fold_va):
    out, cals = {}, {}
    for method in ("sigmoid", "isotonic"):
        cal_full = CalibratedClassifierCV(FrozenEstimator(model), method=method).fit(Xva, yva)
        p_in = cal_full.predict_proba(Xva)[:, 1]
        p_te = cal_full.predict_proba(Xte)[:, 1]
        p_oof = np.empty(len(yva))
        for k in (0, 1):
            keep = fold_va != k
            cal_k = CalibratedClassifierCV(FrozenEstimator(model), method=method).fit(
                Xva[keep], yva[keep])
            p_oof[~keep] = cal_k.predict_proba(Xva[~keep])[:, 1]
        name = f"{base} + {method}"
        out[name] = {"pva": p_oof, "pva_in": p_in, "pte": p_te,
                     "info": f"CalibratedClassifierCV(FrozenEstimator({base})), fitted on validation; "
                             f"validation selection score is 2-fold out-of-fold"}
        cals[name] = cal_full
    return out, cals


def calibrator_params(cal):
    c = cal.calibrated_classifiers_[0].calibrators[0]
    if hasattr(c, "a_"):
        return {"method": "sigmoid", "a": float(c.a_), "b": float(c.b_)}
    return {"method": "isotonic", "x_thresholds": [float(v) for v in c.X_thresholds_],
            "y_thresholds": [float(v) for v in c.y_thresholds_]}


def run_split(label, sets, S, H):
    tr = S.filter(pl.col("match_key").is_in(sorted(sets["train"])))
    va = S.filter(pl.col("match_key").is_in(sorted(sets["val"])))
    te = S.filter(pl.col("match_key").is_in(sorted(sets["test"])))
    fold_map = val_fold_ids(sets["val"])
    fold_va = np.array([fold_map[m] for m in va["match_key"].to_list()])
    res, objs, (Xva, yva, Xte, yte) = fit_models(tr, va, te, fold_va)
    cal_objs = {}
    for base in CAL_BASES:
        out, cals = calibrate_base(base, objs[base], Xva, yva, Xte, fold_va)
        res.update(out)
        cal_objs.update(cals)
    results = {}
    for name, r in res.items():
        results[name] = {
            "val": metrics(yva, r["pva"]),
            "test": metrics(yte, r["pte"]),
            "val_in_sample": metrics(yva, r["pva_in"]) if "pva_in" in r else None,
            "info": r["info"],
            "calibrated": " + " in name,
        }
    return {"label": label, "results": results, "objs": objs, "cal_objs": cal_objs,
            "sizes": {"train": tr, "val": va, "test": te},
            "y": {"val": yva, "test": yte}}


def set_summary(df):
    n_matches = df["match_key"].n_unique()
    n_rounds = df.select(["match_key", "round"]).unique().height
    return {"matches": n_matches, "rounds": n_rounds, "states": df.height,
            "ct_share": float(df["label_ct_win"].mean())}


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return out


def f4(x):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"


def reliability_rows(table):
    rows = []
    for lo, hi, n, mp, my in table:
        if n:
            rows.append([f"{lo:.1f}-{hi:.1f}", n, f4(mp), f4(my), f4(abs(my - mp))])
    return rows


def main():
    t0 = time.time()
    print("== Round win-probability model bake-off ==")
    print(f"tome: {fv.TOME}")
    print(f"seed={SEED}; test_frac={TEST_FRAC}; validation_frac_of_train_pool={VAL_FRAC}; "
          f"chrono_train_frac={CHRONO_TRAIN_FRAC}")
    print(f"features ({len(FEATS)}): {FEATS}")
    print(f"monotone: {dict(zip(FEATS, MONO))}")

    hdr = fv.read("header", ["key", "match_id", "match_date", "max_rounds", "tick_rate",
                             "t_starters_score_final", "ct_starters_score_final"])
    H = {r["key"]: r for r in hdr.iter_rows(named=True)}
    all_matches = set(H)
    tick_rates = collections.Counter(r["tick_rate"] for r in H.values())
    max_rounds_dist = collections.Counter(r["max_rounds"] for r in H.values())
    print(f"\n== 0. Header ==\nheader matches: {len(all_matches)}; tick_rate: {dict(tick_rates)}; "
          f"max_rounds: {dict(max_rounds_dist)}")
    assert set(tick_rates) == {TICK_RATE}, "unexpected tick rate"

    RE = fv.read("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code",
                               "player_count"])
    RST = fv.read("round_start", ["match_key", "round", "tick"])
    RS = fv.read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
    DE = fv.read("player_death", ["match_key", "round", "tick", "player_id", "player_team_code",
                                  "weapon_name", "attacker_id"])
    BS = fv.read("bomb_state", ["match_key", "round", "tick", "event_type"])
    DC = fv.read("player_disconnect", ["match_key", "round", "tick"])
    BT = fv.read("bot_takeover", ["match_key", "round", "tick"])
    TC = fv.read("team_change", ["match_key", "round", "tick"])
    PI = fv.read("player_info", ["match_key", "round", "player_id", "team_code"])
    frames = {"round_end": RE, "round_start": RST, "round_state": RS, "player_death": DE,
              "player_info": PI, "bomb_state": BS, "team_change": TC,
              "player_disconnect": DC, "bot_takeover": BT}
    n_rounds_total = RE.height
    print(f"round_end rows (rounds): {n_rounds_total}; unique (match, round): "
          f"{RE.unique(['match_key', 'round']).height}")

    # ---------- 1. match-level channel check
    print("\n== 1. Match-level channel check ==")
    chan_rows = []
    missing_by_match = collections.defaultdict(list)
    for ch in DENSE + SPARSE:
        df = frames[ch]
        ks = keyset_keys(ch)
        have = set(df["match_key"].unique().to_list())
        extra = have - all_matches
        no_keyset = all_matches - ks
        zero_rows = all_matches - have
        dense = ch in DENSE
        miss = no_keyset | (zero_rows if dense else set())
        for m in miss:
            missing_by_match[m].append(ch)
        chan_rows.append([ch, "dense" if dense else "sparse", df.height, len(have), len(ks),
                          len(zero_rows), len(no_keyset), len(extra), len(miss)])
        print(f"  {ch:<20} rows={df.height:>7} matches_with_rows={len(have):>5} keyset={len(ks):>5} "
              f"zero_row_matches={len(zero_rows):>5} not_in_keyset={len(no_keyset)} "
              f"rows_outside_header={len(extra)} counted_missing={len(miss)} "
              f"({'dense' if dense else 'sparse: zero rows is normal'})")
    dropped_matches = set(missing_by_match)
    kept_matches = all_matches - dropped_matches
    n_rounds_dropped_match = sum(1 for x in RE.select("match_key").to_series().to_list()
                                 if x in dropped_matches)
    combos = collections.Counter(tuple(sorted(v)) for v in missing_by_match.values())
    print(f"matches dropped for a missing channel: {len(dropped_matches)} "
          f"(rounds in them: {n_rounds_dropped_match})")
    for combo, n in combos.most_common():
        print(f"    missing {list(combo)}: {n} matches")
    print(f"matches kept at this stage: {len(kept_matches)}")

    # event signature per match (round_end, player_death, player_info) for duplicate detection
    sig_parts = collections.defaultdict(list)
    for x in RE.iter_rows(named=True):
        sig_parts[x["match_key"]].append(repr(("R", int(x["round"]), int(x["tick"]),
                                               int(x["winner_team_code"]))))
    for x in DE.iter_rows(named=True):
        sig_parts[x["match_key"]].append(repr(("D", int(x["round"]), int(x["tick"]),
                                               str(x["player_id"]))))
    for x in PI.iter_rows(named=True):
        sig_parts[x["match_key"]].append(repr(("P", int(x["round"]), str(x["player_id"]),
                                               str(x["team_code"]))))
    sigs = {m: tuple(sorted(v)) for m, v in sig_parts.items()}

    # ---------- 2. round filters and states
    print("\n== 2. Round filters (per round, applied in this order) ==")
    rd = {}
    for x in RE.iter_rows(named=True):
        m = x["match_key"]
        if m in kept_matches:
            rd[(m, int(x["round"]))] = {"end": int(x["tick"]), "winner": x["winner_team_code"],
                                        "pc": x["player_count"]}
    rst_keys = {(x["match_key"], int(x["round"])) for x in RST.iter_rows(named=True)
                if x["match_key"] in kept_matches}
    live = {}
    live_dups = 0
    for x in RS.filter(pl.col("event_type") == "round_freeze_end").iter_rows(named=True):
        m = x["match_key"]
        if m not in kept_matches:
            continue
        k = (m, int(x["round"]))
        if k in live:
            live_dups += 1
            continue
        live[k] = {"tick": int(x["tick"]), "t": x["t_score"], "ct": x["ct_score"]}
    players = collections.defaultdict(dict)
    pi_dup_rows = 0
    for x in PI.iter_rows(named=True):
        m = x["match_key"]
        if m not in kept_matches:
            continue
        k = (m, int(x["round"]))
        if x["player_id"] in players[k]:
            pi_dup_rows += 1
        players[k][x["player_id"]] = x["team_code"]
    disc_rounds = set()
    for df in (DC, BT):
        for x in df.iter_rows(named=True):
            if x["match_key"] in kept_matches:
                disc_rounds.add((x["match_key"], int(x["round"])))
    plants = collections.defaultdict(list)
    for x in BS.filter(pl.col("event_type") == "bomb_planted").iter_rows(named=True):
        if x["match_key"] in kept_matches:
            plants[(x["match_key"], int(x["round"]))].append(int(x["tick"]))
    deaths = collections.defaultdict(list)
    for x in DE.sort("tick").iter_rows(named=True):
        m = x["match_key"]
        if m in kept_matches:
            deaths[(m, int(x["round"]))].append({"tick": int(x["tick"]), "pid": x["player_id"],
                                                 "team": x["player_team_code"],
                                                 "weapon": x["weapon_name"],
                                                 "attacker": x["attacker_id"]})
    print(f"live-start rows: {len(live)}; duplicate round_freeze_end rows skipped: {live_dups}; "
          f"player_info duplicate (match, round, player) rows: {pi_dup_rows}")

    reasons = collections.Counter()
    survivors = []
    for k in sorted(rd):
        rr = rd[k]
        pl_ = players.get(k, {})
        n_t = sum(1 for v in pl_.values() if v == T)
        n_ct = sum(1 for v in pl_.values() if v == CT)
        if rr["pc"] != 10:
            reasons["round_end.player_count != 10"] += 1
            continue
        if not (len(pl_) == 10 and n_t == START and n_ct == START):
            reasons["not exactly 5 T and 5 CT at start (player_info)"] += 1
            continue
        if k in disc_rounds:
            reasons["disconnect or bot takeover event in round"] += 1
            continue
        if k not in rst_keys:
            reasons["no round_start row"] += 1
            continue
        if k not in live:
            reasons["no round_freeze_end (live start) row"] += 1
            continue
        if live[k]["t"] is None or live[k]["ct"] is None:
            reasons["null score at live start"] += 1
            continue
        survivors.append(k)

    # ---------- duplicate matches (identical event signature under two keys)
    pre_final = {k[0] for k in survivors}
    pre_sets = make_splits(pre_final, H)
    sig_groups = collections.defaultdict(list)
    for m in sorted(kept_matches):
        sig_groups[sigs.get(m, ())].append(m)
    dedup_groups = [g for g in sig_groups.values() if len(g) > 1]
    dup_drop = set()
    for g in dedup_groups:
        dup_drop.update(g[1:])
    straddle = {PRIMARY: 0, CHRONO: 0}
    for g in dedup_groups:
        members = [m for m in g if m in pre_final]
        if len(members) < 2:
            continue
        for name in (PRIMARY, CHRONO):
            parts = {part for m in members for part in ("train", "val", "test")
                     if m in pre_sets[name][part]}
            if len(parts) > 1:
                straddle[name] += 1
    dup_rounds = sum(1 for k in survivors if k[0] in dup_drop)
    dup_matches_with_rounds = len({k[0] for k in survivors if k[0] in dup_drop})
    no_round_matches = len(kept_matches - pre_final)
    reasons["duplicate copy of a kept match (identical rounds, deaths, players)"] += dup_rounds
    survivors = [k for k in survivors if k[0] not in dup_drop]
    print(f"duplicate groups among channel-kept matches: {len(dedup_groups)} groups, "
          f"{sum(len(g) for g in dedup_groups)} matches; duplicate matches removed (with kept rounds): "
          f"{dup_matches_with_rounds}; rounds removed: {dup_rounds}")
    print(f"matches with no kept round: {no_round_matches}")
    print(f"before removal, duplicate groups with members on different sides: primary "
          f"{straddle[PRIMARY]}, chronological {straddle[CHRONO]}")

    diag = collections.Counter()
    state_rows = []
    rounds_ok = []
    for k in survivors:
        rr = {**rd[k], "freeze": live[k]["tick"], "t_score": live[k]["t"], "ct_score": live[k]["ct"]}
        local = collections.Counter()
        out = build_states(k, rr, deaths.get(k, []), plants.get(k, []), players.get(k, {}),
                           H[k[0]]["max_rounds"], local)
        if out is None:
            reasons["death with unresolved victim side in live window"] += 1
            continue
        diag.update(local)
        state_rows.extend(out)
        rounds_ok.append(k)
    S = pl.DataFrame(state_rows)
    kept_final_matches = {k[0] for k in rounds_ok}
    n_rounds_before_round_filters = len(rd)
    print(f"rounds in kept matches: {len(rd)}")
    for name, n in reasons.most_common():
        print(f"  dropped: {name}: {n}")
    print(f"rounds kept: {len(rounds_ok)} in {len(kept_final_matches)} matches")
    print(f"states: {S.height} (live start {int((S['kind'] == 'live_start').sum())}, "
          f"death {int((S['kind'] == 'death').sum())}); per round mean {S.height / len(rounds_ok):.2f}")
    print(f"diagnostics: {dict(diag)}")
    print(f"in-window disconnect/takeover rounds among kept: "
          f"{sum(1 for k in rounds_ok if k in disc_rounds)} (should be 0)")
    assert not any(k in disc_rounds for k in rounds_ok), "a round with disconnect/takeover was kept"

    # ---------- 3. score field and swap checks (all matches, not only kept)
    print("\n== 3. Score fields and side swaps (all matches) ==")
    sc = score_checks(RE, RS, H)
    print(f"freeze rows with a round: {sc['rounds_with_freeze']}; t_score + ct_score == round-1 at start: "
          f"{sc['sum_eq_r_minus_1']}")
    print(f"round-to-round transitions: {sc['transitions']}")
    print(f"transitions where the swap schedule says sides swap: {sc['schedule_swap_transitions']}; "
          f"consistent with a swap: {sc['schedule_swap_consistent']}")
    print(f"inconsistent transitions by (round, max_rounds, schedule swap): {sc['inconsistent_by_round'][:12]}")

    # ---------- 4. completion checks against header final scores (all matches)
    comp, comp_ex = completion_checks(RE, H)
    print("\n== 4. Completion checks (header final scores vs round_end counts) ==")
    for k, v in sorted(comp.items()):
        print(f"  {k}: {v}")
    for k, v in comp_ex.items():
        print(f"  e.g. {k}: {v}")

    # duplicate match check by round fingerprint (ids are redacted)
    fps = collections.defaultdict(list)
    per_match_rounds = collections.defaultdict(list)
    for x in RE.iter_rows(named=True):
        per_match_rounds[x["match_key"]].append((int(x["round"]), int(x["tick"]), int(x["winner_team_code"])))
    for m, lst in per_match_rounds.items():
        if len(lst) >= 5:
            fps[tuple(sorted(lst))].append(m)
    dup_groups = [v for v in fps.values() if len(v) > 1]
    plant_rate = BS.filter(pl.col("event_type") == "bomb_planted").height / n_rounds_total
    print(f"duplicate matches by round fingerprint (>=5 rounds): {len(dup_groups)} groups, "
          f"{sum(len(v) for v in dup_groups)} matches")

    # ---------- 5. splits
    print("\n== 5. Splits ==")
    sets = make_splits(kept_final_matches, H)
    for name, sp in sets.items():
        print(f"{name}: train {len(sp['train'])}, val {len(sp['val'])}, test {len(sp['test'])} matches")
    S_kept = S
    split_info = {}
    for name in (PRIMARY, CHRONO):
        split_info[name] = {}
        for part in ("train", "val", "test"):
            ms = sets[name][part]
            sub = S_kept.filter(pl.col("match_key").is_in(sorted(ms)))
            info = set_summary(sub)
            dates = sorted(parse_date(H[m]["match_date"]) for m in ms)
            info["date_min"] = dates[0].date().isoformat() if dates else "n/a"
            info["date_max"] = dates[-1].date().isoformat() if dates else "n/a"
            split_info[name][part] = info
            print(f"  {name:<13} {part:<5} matches={info['matches']:>5} rounds={info['rounds']:>6} "
                  f"states={info['states']:>7} CT-win share (states)={info['ct_share']:.4f} "
                  f"dates {info['date_min']}..{info['date_max']}")

    # ---------- 6. models
    runs = {}
    for name in (PRIMARY, CHRONO):
        print(f"\n== 6. Models: {name} split ==")
        t_fit = time.time()
        runs[name] = run_split(name, sets[name], S_kept, H)
        for mname, r in runs[name]["results"].items():
            print(f"  {mname:<26} val LL={r['val']['logloss']:.5f} test LL={r['test']['logloss']:.5f} "
                  f"test Brier={r['test']['brier']:.5f} test AUC={r['test']['auc']:.4f} "
                  f"test ECE={r['test']['ece']:.4f}")
            print(f"      {r['info']}")
            if r["val_in_sample"] is not None:
                print(f"      in-sample val LL (flagged, not for selection): "
                      f"{r['val_in_sample']['logloss']:.5f}")
        print(f"  fit time {time.time() - t_fit:.1f}s")

    # ---------- 7. selection by validation log loss (primary is the decision split)
    prim_res = runs[PRIMARY]["results"]
    best_prim = min(prim_res, key=lambda n: prim_res[n]["val"]["logloss"])
    best_prim_uncal = min((n for n in prim_res if " + " not in n),
                          key=lambda n: prim_res[n]["val"]["logloss"])
    best_chrono = min(runs[CHRONO]["results"], key=lambda n: runs[CHRONO]["results"][n]["val"]["logloss"])
    print("\n== 7. Selection ==")
    print(f"best by primary validation log loss (all 12 rows, calibrated scored out-of-fold): {best_prim}")
    print(f"best uncalibrated by primary validation log loss: {best_prim_uncal}")
    print(f"best by chronological validation log loss: {best_chrono}")
    lit = min(prim_res, key=lambda n: prim_res[n]["val_in_sample"]["logloss"]
              if prim_res[n]["val_in_sample"] is not None else prim_res[n]["val"]["logloss"])
    print(f"(for reference, best using in-sample calibrated validation scores: {lit})")

    saved = "no"
    model_note = ""
    if " + " in best_prim:
        base, method = best_prim.split(" + ")
        obj = runs[PRIMARY]["objs"][base]
        cal = runs[PRIMARY]["cal_objs"][best_prim]
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        if base == "XGBoost":
            path = MODEL_DIR / "round_bakeoff_xgboost_base.json"
            obj.save_model(str(path))
            saved = str(path)
        elif base == "LightGBM":
            path = MODEL_DIR / "round_bakeoff_lightgbm_base.txt"
            obj.booster_.save_model(str(path))
            saved = str(path)
        else:
            saved = "not saved: HistGB has no json or txt format"
        if saved.startswith(str(MODEL_DIR)):
            side = MODEL_DIR / f"{path.stem}_calibrator_{method}.json"
            side.write_text(json.dumps(calibrator_params(cal), indent=1), encoding="utf-8")
            model_note = f"calibrator parameters in {side}"
    else:
        model_note = "best model is uncalibrated, so no model file is saved"
    print(f"model file saved: {saved}; {model_note}")

    # ---------- 8. write notes
    best_base = best_prim.split(" + ")[0]
    related = [n for n in prim_res if n.startswith(best_base)]
    best_test = prim_res[best_prim]["test"]
    md = []
    md.append("# Round win-probability model bake-off (2026 Oct 01-08)")
    md.append("")
    md.append("Script: `scripts/model/bakeoff.py` (seed 0). Log: `logs/bakeoff.log`. Tome: "
              "`tomes/2026-10-01_to_08/tome/csds`. Per-tick channels are not in the tome. "
              "All quantities are event-derived (spec: `notes/spec/feature_spec.md`).")
    md.append("")
    md.append(f"**Headline (primary split, test set, {len(S_kept)} states total):** best by primary "
              f"validation log loss is `{best_prim}`: test log loss {best_test['logloss']:.5f}, "
              f"Brier {best_test['brier']:.5f}, AUC {best_test['auc']:.4f}, ECE {best_test['ece']:.4f}.")
    md.append("")
    md.append("## 1. Data counts and filters")
    md.append("")
    md.append(f"Header matches: {len(all_matches)}. round_end rows (rounds): {n_rounds_total}.")
    md.append("")
    md.extend(md_table(["channel", "kind", "rows", "matches with rows", "keyset matches",
                        "zero-row matches", "not in keyset", "rows outside header", "counted missing"],
                       chan_rows))
    md.append("")
    md.append("Keysets: every channel's keyset equals the header's match list, so keyset membership "
              "drops nothing. The dense-channel zero-row matches are the real filter. "
              "Sparse channels (player_disconnect, bot_takeover) are not filtered on zero rows, "
              "since zero events is a normal outcome.")
    md.append("")
    md.append("| stage | matches | rounds |")
    md.append("|---|---:|---:|")
    md.append(f"| header / round_end | {len(all_matches)} | {n_rounds_total} |")
    md.append(f"| dropped: missing a required channel | {len(dropped_matches)} | {n_rounds_dropped_match} |")
    md.append(f"| rounds in kept matches (before round filters) | {len(kept_matches)} | {len(rd)} |")
    for name, n in reasons.most_common():
        md.append(f"| dropped round: {name} | - | {n} |")
    md.append(f"| matches with no round left after the round filters | {no_round_matches} | - |")
    md.append(f"| **remaining (used)** | **{len(kept_final_matches)}** | **{len(rounds_ok)}** |")
    md.append("")
    md.append(f"Missing-channel combinations: " + "; ".join(
        f"{list(c)}: {n} matches" for c, n in combos.most_common()))
    md.append("")
    md.append(f"States: {S.height} ({int((S['kind'] == 'live_start').sum())} live starts, "
              f"{int((S['kind'] == 'death').sum())} death states). Deaths in window: "
              f"{diag['deaths_in_window']}; before live: {diag['deaths_pre_live']}; after round_end: "
              f"{diag['deaths_post_end']}; victim team null in window: {diag['in_window_null_team']} "
              f"(imputed from player_info: {diag['in_window_null_team_imputed']}); repeat "
              f"in-window victims skipped: {diag['in_window_duplicate_victim_skipped']}. Bomb plants "
              f"in window: {diag['plants_in_window']}; after round_end: {diag['plants_after_end']}; "
              f"rounds with more than one plant: {diag['rounds_with_multiple_plants']}.")
    md.append("")
    md.append("## 2. Score fields and side swaps")
    md.append("")
    md.append(f"- t_score + ct_score == round - 1 at live start: {sc['sum_eq_r_minus_1']} of "
              f"{sc['rounds_with_freeze']} rounds.")
    md.append(f"- Round-to-round transitions: {sc['transitions']}.")
    md.append(f"- Transitions where the side-swap schedule (halftime at max_rounds/2, then after the "
              f"3rd round of each overtime period) predicts a swap: {sc['schedule_swap_transitions']}; "
              f"consistent with a swap: {sc['schedule_swap_consistent']}.")
    md.append(f"- Inconsistent transitions (round, max_rounds, schedule swap): "
              f"{sc['inconsistent_by_round'][:20] if sc['inconsistent_by_round'] else 'none'}.")
    md.append("- Reading: the score fields follow the team on each side, so ct_score - t_score is the "
              "CT-minus-T score for the sides in play. This is the basis for the feature.")
    md.append("")
    md.append("## 3. Splits")
    md.append("")
    rows = []
    for name in (PRIMARY, CHRONO):
        for part in ("train", "val", "test"):
            si = split_info[name][part]
            rows.append([name, part, si["matches"], si["rounds"], si["states"], f"{si['ct_share']:.4f}",
                         f"{si['date_min']} .. {si['date_max']}"])
    md.extend(md_table(["split", "set", "matches", "rounds", "states", "CT-win share (states)",
                        "match dates"], rows))
    md.append("")
    md.append("Primary: 20% of matches are test; 10% of the remaining matches are validation. "
              "Chronological: earliest 80% (train pool) and latest 20% (test); the latest 10% of the train "
              "pool are validation. No match appears on two sides of any split (checked by assertion).")
    md.append("")
    md.append("## 4. Results")
    md.append("")
    md.append("Metrics are on states. Log loss, Brier and ECE are the primary criteria; accuracy is "
              "reference only. Calibrated validation log loss is out-of-fold (2 folds by match inside "
              "validation). The in-sample value is in the last column and is not for selection.")
    md.append("")
    for name in (PRIMARY, CHRONO):
        md.append(f"### {name} split")
        md.append("")
        rows = []
        for mname, r in runs[name]["results"].items():
            for setname in ("val", "test"):
                m = r[setname]
                ins = ""
                if setname == "val" and r["val_in_sample"] is not None:
                    ins = f4(r["val_in_sample"]["logloss"])
                rows.append([mname, setname, m["n"], f4(m["logloss"]), f4(m["brier"]), f4(m["auc"]),
                             f4(m["ece"]), f4(m["acc"]), ins])
        md.extend(md_table(["model", "set", "states", "log loss", "Brier", "AUC", "ECE (10 bins)",
                            "accuracy (ref)", "val LL in-sample (calibrated only)"], rows))
        md.append("")
    md.append("Comparison with the first run (duplicate copies not removed). Primary test "
              "log loss, from `logs/bakeoff_run1_no_dedupe.log`. The change mixes duplicate removal with "
              "a different split (the match set changed), so it is not a clean measure of leakage:")
    md.append("")
    run1 = {"A constant": 0.69214, "B LR ct_alive_minus_t": 0.45790, "LR all features": 0.45160,
            "XGBoost": 0.43353, "LightGBM": 0.43416, "HistGB": 0.43317}
    rows = []
    for mname in MODEL_NAMES:
        if mname in prim_res:
            rows.append([mname, f4(run1[mname]), f4(prim_res[mname]["test"]["logloss"]),
                         f"{prim_res[mname]['test']['logloss'] - run1[mname]:+.4f}"])
    md.extend(md_table(["model", "run 1 test LL (duplicates kept)", "this run test LL", "change"], rows))
    md.append("")
    md.append("Model settings (no tuning was done; the same settings were used on both splits): "
              "XGBoost hist, depth 4, eta 0.05, min_child_weight 20, subsample 0.8, early stopping 100. "
              "LightGBM 15 leaves, lr 0.05, min_child_samples 50, early stopping 100. HistGB 15 leaves, "
              "lr 0.05, min_samples_leaf 50, internal early stopping. Logistic regressions are "
              "standardised with C=1.0.")
    md.append("")
    md.append("Per-model details:")
    md.append("")
    for name in (PRIMARY, CHRONO):
        for mname, r in runs[name]["results"].items():
            md.append(f"- {name}, {mname}: {r['info']}")
    md.append("")
    md.append("## 5. Selection and calibration")
    md.append("")
    md.append(f"- Best by primary validation log loss: **{best_prim}** (test log loss "
              f"{best_test['logloss']:.5f}).")
    md.append(f"- Best uncalibrated by primary validation log loss: {best_prim_uncal}.")
    uncal_sorted = sorted((n for n in prim_res if " + " not in n),
                          key=lambda n: prim_res[n]["val"]["logloss"])
    ru = uncal_sorted[1]
    md.append(f"- Runner-up among uncalibrated models on primary validation: {ru} "
              f"({prim_res[ru]['val']['logloss']:.5f}), gap "
              f"{prim_res[ru]['val']['logloss'] - prim_res[best_prim_uncal]['val']['logloss']:.5f}. "
              "Gaps of this size between tree models were not tested for significance.")
    if " + " in best_prim:
        md.append(f"- Near-tie check: the best raw model's out-of-fold validation log loss is "
                  f"{prim_res[best_base]['val']['logloss']:.5f} against "
                  f"{prim_res[best_prim]['val']['logloss']:.5f} for its calibrated variant.")
    else:
        cands = [n for n in prim_res if n.startswith(best_base) and " + " in n]
        if cands:
            cal_best = min(cands, key=lambda n: prim_res[n]["val"]["logloss"])
            md.append(f"- Calibrated variants: the best is {cal_best} at "
                      f"{prim_res[cal_best]['val']['logloss']:.5f} out-of-fold validation log loss, against "
                      f"{prim_res[best_prim]['val']['logloss']:.5f} for {best_prim}. The rule as written "
                      f"picks the raw model.")
    md.append(f"- Best by chronological validation log loss: {best_chrono}.")
    md.append(f"- Model file: {saved}. {model_note}.")
    md.append("")
    md.append(f"### Calibration of `{best_base}` (primary split, test set)")
    md.append("")
    for n in related:
        r = prim_res[n]["test"]
        md.append(f"- {n}: test log loss {r['logloss']:.5f}, Brier {r['brier']:.5f}, ECE {r['ece']:.4f}")
    md.append("")
    for n in related:
        md.append(f"Reliability: {n}, primary test")
        md.append("")
        md.extend(md_table(["bin", "states", "mean predicted", "observed CT-win", "|gap|"],
                           reliability_rows(prim_res[n]["test"]["table"])))
        md.append("")
    md.append(f"Reliability: {best_base} raw, chronological test")
    md.append("")
    ch_base = runs[CHRONO]["results"][best_base]
    md.extend(md_table(["bin", "states", "mean predicted", "observed CT-win", "|gap|"],
                       reliability_rows(ch_base["test"]["table"])))
    md.append("")
    md.append("## 6. What looked wrong or needs a check")
    md.append("")
    n16 = sum(1 for h in H.values() if h["max_rounds"] == 16)
    md.append(f"- Feature validation's claim (`notes/spec/feature_spec.md`, section 7) that 1,391 "
              f"matches have fewer than 24 rounds and suggests truncated demos is contradicted by the "
              f"header: {comp.get('matches_below_max_rounds', 0)} matches have fewer rounds than "
              f"max_rounds, of which {comp.get('below_max_rounds_completed_by_threshold', 0)} end when one "
              f"side reaches the win threshold (max_rounds//2+1). Only "
              f"{comp.get('below_max_rounds_not_completed', 0)} look incomplete. All {n16} matches in the "
              f"16-round format are below 24 rounds by construction, so the 24-round filter was never "
              f"meaningful for them. A short match is a normal end, not a fault.")
    md.append(f"- Round count vs header final scores: {comp.get('n_rounds_equals_header_final_sum', 0)} "
              f"matches agree; {comp.get('n_rounds_below_header_final_sum', 0)} have fewer round_end rows than "
              f"the header total (the tome is missing the last rounds, consistent with truncation), and "
              f"{comp.get('n_rounds_above_header_final_sum', 0)} have more. Cause for the extra rounds: "
              f"UNVERIFIED. Kept rounds are still complete per round, so the filters do not depend on this.")
    md.append(f"- Matches missing a dense channel: " + "; ".join(
        f"{list(c)} {n}" for c, n in combos.most_common()) + f". Plants per round in the whole tome: "
        f"{plant_rate:.2f}. A match with 9 or more rounds and zero bomb_state rows is then unlikely "
        "to be a real no-plant match, so these look like missing data (UNVERIFIED: no per-tick source "
        "to confirm).")
    md.append(f"- Rounds with player_count != 10: "
              f"{reasons.get('round_end.player_count != 10', 0)} of the rounds in kept matches. "
              f"Disconnect or takeover rounds dropped: {reasons.get('disconnect or bot takeover event in round', 0)}. "
              f"Rounds not 5 vs 5 at start from player_info: "
              f"{reasons.get('not exactly 5 T and 5 CT at start (player_info)', 0)}.")
    md.append(f"- Duplicate matches: {len(dedup_groups)} groups ({sum(len(g) for g in dedup_groups)} matches) "
              f"have identical round_end, player_death and player_info events under different keys. "
              f"Copies differ in header match_date by seconds to over an hour. The first run did not remove "
              f"them, and copies landed on different sides of the split: primary {straddle[PRIMARY]} groups, "
              f"chronological {straddle[CHRONO]} groups. This run keeps the lowest key per group. "
              f"demo_id and sharecode are 'redacted' in every header row, so the check is on events only. "
              f"Round-fingerprint check (>=5 rounds, all matches) found {len(dup_groups)} groups.")
    n_1008 = sum(1 for m in all_matches if H[m]["match_date"].startswith("2026-10-08"))
    key_agree = sum(1 for m in all_matches if H[m]["match_date"][:10] == "-".join(m.split("/")[1:4]))
    md.append(f"- match_date: {n_1008} of {len(all_matches)} matches ({100 * n_1008 / len(all_matches):.0f}%) "
              f"are dated 2026-10-08. The match_date day equals the processing day in the key path for "
              f"{key_agree} of {len(all_matches)} matches. The spec calls matchDate the played date. Whether "
              f"the field is really the played date is UNVERIFIED. The chronological test set is the latest "
              f"20% by timestamp, and here it is all on 2026-10-08, so it checks within-day drift, not "
              f"multi-day drift.")
    md.append("- Sigmoid calibration: sklearn's `_SigmoidCalibration.fit` fits a logistic on the raw "
              "predicted probabilities (not their logits), so it cannot represent the identity map. For that "
              "reason it can make a model worse than the raw score, even in-sample. Compare XGBoost raw and "
              "XGBoost + sigmoid validation log loss in the results table. Isotonic is not limited this way. "
              "No other calibrator was tried.")
    md.append("- Event-anchored states: a state exists only at the live start and at each death. A bomb "
              "plant with no later death adds no state, and the plant's effect appears only at the next "
              "state. This follows the spec, but it changes the time distribution of states.")
    md.append("- HistGB early stopping uses a random 10% of training rows (row-level), not the match-level "
              "validation set. Its stopping point may be optimistic. Calibration of HistGB still uses the "
              "match-level validation set.")
    md.append("- LightGBM 4.7.0 deprecates eval_set in favour of eval_X/eval_y. The run uses eval_X/eval_y "
              "as the task specifies.")
    md.append("- Calibrated models: in-sample validation scores are flattering. Selection uses out-of-fold "
              "validation scores. Test scores come from a calibrator fitted on all of validation.")
    md.append("- The validation fraction is read as 10% of the 80% (primary) or the earliest 80% (chrono) "
              "train pool, not 10% of all matches. This is an interpretation.")
    md.append("- Metrics are on states, not rounds, so long rounds (more deaths) carry more weight. "
              "Round-level metrics were not computed.")
    md.append("- No hyperparameter search was run, so these results are for one fixed setting per model.")
    md.append("")
    md.append(f"Run time: {time.time() - t0:.1f}s.")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT_MD}")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_fh:
        real_stdout = sys.stdout
        sys.stdout = Tee(real_stdout, log_fh)
        try:
            main()
        finally:
            sys.stdout = real_stdout
