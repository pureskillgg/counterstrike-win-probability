"""Task A: reproduce and save the XGBoost model on the PRIMARY split, seed 0.

Reuses scripts/model/bakeoff.py for everything it exposes at module level: constants,
feature list, monotone constraints, splits (make_splits), state builder (build_states),
label and feature matrix (xy), metrics, keyset loader and set summary. The data-building
steps that bakeoff.py keeps inline in main() (channel check, round filters, duplicate
removal, state loop) cannot be imported, so they are repeated here in the same order and
with the same logic. The resulting counts are checked against bakeoff_results.md before
the metrics are compared.

The model is saved only if the data counts AND the test log loss and Brier score
reproduce the reported values. Nothing is saved in --no-dedupe diagnostic mode.

Run from the project root:
  pixi run python -I scripts\\model\\xgb_primary_seed0.py             (reproduce; save on pass)
  pixi run python -I scripts\\model\\xgb_primary_seed0.py --no-dedupe (diagnostic; never saves)
"""

import collections
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "logs"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "xgb_primary_seed0.json"
META_PATH = MODEL_DIR / "xgb_primary_seed0_meta.json"

_spec = importlib.util.spec_from_file_location("bakeoff", ROOT / "scripts" / "model" / "bakeoff.py")
bo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bo)
fv = bo.fv
T, CT, START, TICK_RATE = bo.T, bo.CT, bo.START, bo.TICK_RATE
PRIMARY = bo.PRIMARY

# Reported values: notes/model/bakeoff_results.md (deduplicated run, 18:58), section 4 primary
# split XGBoost rows and section 5 test calibration line (5 decimals). Best iteration: section 4 details.
REF = {"test_ll": 0.43660, "test_brier": 0.14637, "val_ll": 0.4351, "val_brier": 0.1455, "best_iter": 261}
TOL = 0.0002
# Reported counts: bakeoff_results.md sections 1 and 3 (primary split, deduplicated run).
REF_COUNTS = {
    "matches": 1521, "rounds": 28683, "states": 234265,
    "train": (1095, 20470, 167379), "val": (122, 2364, 19170), "test": (304, 5849, 47716),
}


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


def build_data(dedupe):
    """Ported from bakeoff.py main(), sections 0 to 5 (logic unchanged)."""
    hdr = fv.read("header", ["key", "match_id", "match_date", "max_rounds", "tick_rate",
                             "t_starters_score_final", "ct_starters_score_final"])
    H = {r["key"]: r for r in hdr.iter_rows(named=True)}
    all_matches = set(H)
    assert set(r["tick_rate"] for r in H.values()) == {TICK_RATE}, "unexpected tick rate"

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

    # ---- 1. match-level channel check
    missing_by_match = collections.defaultdict(list)
    for ch in bo.DENSE + bo.SPARSE:
        df = frames[ch]
        ks = bo.keyset_keys(ch)
        have = set(df["match_key"].unique().to_list())
        no_keyset = all_matches - ks
        zero_rows = all_matches - have
        dense = ch in bo.DENSE
        miss = no_keyset | (zero_rows if dense else set())
        for m in miss:
            missing_by_match[m].append(ch)
    dropped = set(missing_by_match)
    kept_matches = all_matches - dropped
    print(f"[1] header matches {len(all_matches)}; dropped for missing channel {len(dropped)}; "
          f"kept {len(kept_matches)}")

    # ---- event signature per match (duplicate detection)
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

    # ---- 2. round filters
    rd = {}
    for x in RE.iter_rows(named=True):
        m = x["match_key"]
        if m in kept_matches:
            rd[(m, int(x["round"]))] = {"end": int(x["tick"]), "winner": x["winner_team_code"],
                                        "pc": x["player_count"]}
    rst_keys = {(x["match_key"], int(x["round"])) for x in RST.iter_rows(named=True)
                if x["match_key"] in kept_matches}
    live = {}
    for x in RS.filter(pl.col("event_type") == "round_freeze_end").iter_rows(named=True):
        m = x["match_key"]
        if m not in kept_matches:
            continue
        k = (m, int(x["round"]))
        if k in live:
            continue
        live[k] = {"tick": int(x["tick"]), "t": x["t_score"], "ct": x["ct_score"]}
    players = collections.defaultdict(dict)
    for x in PI.iter_rows(named=True):
        m = x["match_key"]
        if m not in kept_matches:
            continue
        players[(m, int(x["round"]))][x["player_id"]] = x["team_code"]
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
    print(f"[2] rounds in kept matches {len(rd)}; after round filters {len(survivors)}")
    for name, n in reasons.most_common():
        print(f"      dropped: {name}: {n}")

    # ---- duplicate matches (identical event signature under two keys)
    sig_groups = collections.defaultdict(list)
    for m in sorted(kept_matches):
        sig_groups[sigs.get(m, ())].append(m)
    dedup_groups = [g for g in sig_groups.values() if len(g) > 1]
    dup_drop = set()
    for g in dedup_groups:
        dup_drop.update(g[1:])
    print(f"[2b] duplicate groups among channel-kept matches: {len(dedup_groups)} groups; "
          f"matches in them {sum(len(g) for g in dedup_groups)}; "
          f"matches that would be dropped: {len(dup_drop)}; dedupe applied: {dedupe}")
    if dedupe:
        n_before = len(survivors)
        survivors = [k for k in survivors if k[0] not in dup_drop]
        print(f"      rounds removed by dedupe: {n_before - len(survivors)}")

    # ---- 2b. states
    state_rows = []
    rounds_ok = []
    for k in survivors:
        rr = {**rd[k], "freeze": live[k]["tick"], "t_score": live[k]["t"], "ct_score": live[k]["ct"]}
        out = bo.build_states(k, rr, deaths.get(k, []), plants.get(k, []), players.get(k, {}),
                              H[k[0]]["max_rounds"], collections.Counter())
        if out is None:
            reasons["death with unresolved victim side in live window"] += 1
            continue
        state_rows.extend(out)
        rounds_ok.append(k)
    S = pl.DataFrame(state_rows)
    kept_final = {k[0] for k in rounds_ok}
    print(f"[3] rounds kept {len(rounds_ok)} in {len(kept_final)} matches; states {S.height}")

    # ---- 5. primary split (bakeoff.make_splits)
    sets = bo.make_splits(kept_final, H)[PRIMARY]
    return {"S": S, "sets": sets, "H": H, "kept_final": kept_final,
            "n_rounds": len(rounds_ok), "dedup_groups": len(dedup_groups),
            "dup_drop": len(dup_drop)}


def date_range(ms, H):
    ds = sorted(bo.parse_date(H[m]["match_date"]).date().isoformat() for m in ms)
    return [ds[0], ds[-1]]


def run(dedupe, diag):
    t0 = time.time()
    print("== XGBoost, primary split, seed 0 ==")
    print("mode: " + ("DIAGNOSTIC (no dedupe). Nothing is saved." if diag else
                      "reproduce. Model is saved only if counts and metrics reproduce."))
    print(f"features ({len(bo.FEATS)}): {bo.FEATS}")
    print(f"monotone: {bo.MONO}")
    print(f"tome: {fv.TOME}")

    D = build_data(dedupe)
    S, sets, H = D["S"], D["sets"], D["H"]

    # ---- counts check
    split_frames = {}
    counts_ok = True
    print("\n-- Counts vs bakeoff_results.md --")
    got = [("kept matches", len(D["kept_final"]), REF_COUNTS["matches"]),
           ("kept rounds", D["n_rounds"], REF_COUNTS["rounds"]),
           ("states total", S.height, REF_COUNTS["states"])]
    for part in ("train", "val", "test"):
        sub = S.filter(pl.col("match_key").is_in(sorted(sets[part])))
        split_frames[part] = sub
        info = bo.set_summary(sub)
        got += [(f"{part} matches", info["matches"], REF_COUNTS[part][0]),
                (f"{part} rounds", info["rounds"], REF_COUNTS[part][1]),
                (f"{part} states", info["states"], REF_COUNTS[part][2])]
    for label, g, e in got:
        ok = g == e
        counts_ok = counts_ok and ok
        print(f"  {label:<16} got {g:>7}  reported {e:>7}  {'OK' if ok else 'MISMATCH'}")
    print(f"counts match: {counts_ok}")

    tr, va, te = split_frames["train"], split_frames["val"], split_frames["test"]
    print(f"\ntrain dates {date_range(sets['train'], H)}; val dates {date_range(sets['val'], H)}; "
          f"test dates {date_range(sets['test'], H)}")

    # ---- XGBoost, same settings as bakeoff.fit_models (XGBoost block)
    Xtr, ytr = bo.xy(tr)
    Xva, yva = bo.xy(va)
    Xte, yte = bo.xy(te)
    mX = xgb.XGBClassifier(n_estimators=3000, learning_rate=0.05, max_depth=4, min_child_weight=20,
                           subsample=0.8, tree_method="hist", device="cpu",
                           monotone_constraints=tuple(bo.MONO), early_stopping_rounds=100,
                           eval_metric="logloss", random_state=bo.SEED)
    t_fit = time.time()
    mX.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    bx = int(mX.best_iteration)
    pva = mX.predict_proba(Xva, iteration_range=(0, bx + 1))[:, 1]
    pte = mX.predict_proba(Xte, iteration_range=(0, bx + 1))[:, 1]
    default_same = bool(np.allclose(mX.predict_proba(Xva)[:, 1], pva))
    mv = bo.metrics(yva, pva)
    mt = bo.metrics(yte, pte)
    print(f"\nfit time {time.time() - t_fit:.1f}s; best_iteration {bx} (reported {REF['best_iter']}); "
          f"default predict uses same trees: {default_same}")
    print(f"validation: log loss {mv['logloss']:.5f} (reported {REF['val_ll']:.4f}); "
          f"Brier {mv['brier']:.5f} (reported {REF['val_brier']:.4f})")
    print(f"test:       log loss {mt['logloss']:.5f} (reported {REF['test_ll']:.4f}); "
          f"Brier {mt['brier']:.5f} (reported {REF['test_brier']:.4f}); "
          f"AUC {mt['auc']:.4f}; ECE {mt['ece']:.4f}; acc {mt['acc']:.4f}")

    ll_ok = abs(mt["logloss"] - REF["test_ll"]) <= TOL
    br_ok = abs(mt["brier"] - REF["test_brier"]) <= TOL
    print(f"test log loss within {TOL}: {ll_ok} (diff {mt['logloss'] - REF['test_ll']:+.5f})")
    print(f"test Brier within {TOL}: {br_ok} (diff {mt['brier'] - REF['test_brier']:+.5f})")
    reproduced = counts_ok and ll_ok and br_ok
    print(f"\nREPRODUCED: {reproduced}")

    if diag:
        print("DIAGNOSTIC mode: nothing saved.")
        print(f"done in {time.time() - t0:.1f}s")
        return 0
    if not reproduced:
        print("NOT REPRODUCED: no model saved.")
        print(f"done in {time.time() - t0:.1f}s")
        return 2

    # ---- save model, verify reload, write meta
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    mX.save_model(str(MODEL_PATH))
    booster = xgb.Booster()
    booster.load_model(str(MODEL_PATH))
    p_reload = booster.predict(xgb.DMatrix(Xte), iteration_range=(0, bx + 1))
    reload_ok = bool(np.allclose(p_reload, pte, atol=1e-6))
    print(f"saved {MODEL_PATH}; reload check (test predictions match): {reload_ok}")
    if not reload_ok:
        print("RELOAD CHECK FAILED: model file written but predictions differ.")
        return 3

    def clean(m):
        return {k: m[k] for k in ("n", "logloss", "brier", "auc", "ece", "acc")}

    meta = {
        "model_file": "models/xgb_primary_seed0.json",
        "model_format": "XGBoost Booster JSON (Booster.save_model). Load with xgboost.Booster().load_model; "
                        "predict with iteration_range=(0, best_iteration + 1).",
        "xgboost_version": xgb.__version__,
        "label": "ct_win = (winner_team_code == 3); round-level label repeated on each state of the round",
        "features": bo.FEATS,
        "feature_order_note": "Pass the feature matrix columns in the order listed in features.",
        "monotone_constraints": bo.MONO,
        "best_iteration": bx,
        "predict_iteration_range": [0, bx + 1],
        "xgb_settings": {"n_estimators": 3000, "learning_rate": 0.05, "max_depth": 4,
                         "min_child_weight": 20, "subsample": 0.8, "tree_method": "hist",
                         "early_stopping_rounds": 100, "eval_metric": "logloss",
                         "random_state": bo.SEED},
        "split": {"name": "primary", "seed": bo.SEED, "test_frac": bo.TEST_FRAC,
                  "val_frac_of_train_pool": bo.VAL_FRAC,
                  "train_matches": len(sets["train"]), "val_matches": len(sets["val"]),
                  "test_matches": len(sets["test"]),
                  "train_dates": date_range(sets["train"], H),
                  "val_dates": date_range(sets["val"], H),
                  "test_dates": date_range(sets["test"], H)},
        "data": {"tome": "tomes/2026-10-01_to_08/tome/csds", "dedupe_applied": dedupe,
                 "kept_matches": len(D["kept_final"]), "kept_rounds": D["n_rounds"],
                 "states_total": S.height},
        "metrics_test": clean(mt),
        "metrics_val": clean(mv),
        "reproduction": {"reported_test_log_loss": REF["test_ll"],
                         "reported_test_brier": REF["test_brier"], "tolerance": TOL,
                         "reproduced": reproduced},
        "source_script": "scripts/model/xgb_primary_seed0.py",
        "created": time.strftime("%Y-%m-%d"),
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {META_PATH}")
    print(f"done in {time.time() - t0:.1f}s")
    return 0


def main():
    diag = "--no-dedupe" in sys.argv[1:]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_name = "xgb_primary_seed0_diag_no_dedupe.log" if diag else "xgb_primary_seed0.log"
    status = 1
    with open(LOG_DIR / log_name, "w", encoding="utf-8") as fh:
        real_stdout = sys.stdout
        sys.stdout = Tee(real_stdout, fh)
        try:
            status = run(dedupe=not diag, diag=diag)
        except Exception:
            traceback.print_exc(file=sys.stdout)
            status = 1
        finally:
            sys.stdout = real_stdout
    print(f"exit status {status}; log: {LOG_DIR / log_name}")
    return status


if __name__ == "__main__":
    sys.exit(main())
