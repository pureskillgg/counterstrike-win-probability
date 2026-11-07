"""A4 sensitivity: effect of the disconnect/takeover rule on the bake-off XGBoost (primary split, seed 0).

Variants of the disconnect/takeover round filter (rounds already passing player_count and 5v5 are the
only ones it applies to; order as in bakeoff.py):
  baseline        : any event carrying the round label (as bakeoff.py)
  round_window    : event tick in [round_start tick, round_end tick]  (ASSUMPTION: round = start to end)
  live_window     : event tick in [round_freeze_end tick, round_end tick] (ASSUMPTION: alive counts start at live start)
Everything else is the bake-off code path: channel check, six predicates, duplicate removal (lowest key),
unresolved-victim check, bk.build_states for the states, bk.make_splits for the split, and the XGBoost
settings from bakeoff.py fit_models(). The split uses the same match set in every variant (asserted).

Reports, per variant: rounds, states, XGBoost best_iteration, validation and test log loss, and the test log
loss on the baseline test rounds only (same rounds for every variant). ASSUMPTION: the baseline row must
reproduce bakeoff.py (test log loss 0.43660, best_iteration 261) or the replication is not trusted.

Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\A4_sensitivity.py
"""
import collections
import glob
import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path("G:/pureskill/csds-cs2")


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fv = load("feature_validation", "scripts/model/feature_validation.py")
bk = load("bakeoff", "scripts/model/bakeoff.py")
T, CT, START, TICK_RATE = fv.T, fv.CT, fv.START, fv.TICK_RATE
TOME, SUFFIX = fv.TOME, fv.SUFFIX
DENSE = ["round_end", "round_start", "round_state", "player_death", "player_info",
         "bomb_state", "team_change"]
SPARSE = ["player_disconnect", "bot_takeover"]


def keyset_keys(ch):
    folder = TOME / f"basic_{ch}{SUFFIX}"
    files = sorted(glob.glob(str(folder / "keyset_*")))
    return set(pl.concat([pl.read_parquet(f) for f in files])["_"].to_list())


def main():
    hdr = fv.read("header", ["key", "match_id", "match_date", "max_rounds", "tick_rate",
                             "t_starters_score_final", "ct_starters_score_final"])
    H = {r["key"]: r for r in hdr.iter_rows(named=True)}
    all_matches = set(H)
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

    missing = collections.defaultdict(list)
    for ch in DENSE + SPARSE:
        df = frames[ch]
        ks = keyset_keys(ch)
        have = set(df["match_key"].unique().to_list())
        miss = (all_matches - ks) | ((all_matches - have) if ch in DENSE else set())
        for m in miss:
            missing[m].append(ch)
    kept = all_matches - set(missing)
    print(f"kept matches after channel stage: {len(kept)}")

    rd, rst, frz = {}, {}, {}
    for x in RE.iter_rows(named=True):
        if x["match_key"] in kept:
            rd[(x["match_key"], int(x["round"]))] = {"end": int(x["tick"]), "winner": x["winner_team_code"],
                                                     "pc": x["player_count"]}
    rst_keys = set()
    for x in RST.iter_rows(named=True):
        if x["match_key"] in kept:
            k = (x["match_key"], int(x["round"]))
            rst_keys.add(k)
            rst.setdefault(k, int(x["tick"]))
    live = {}
    for x in RS.filter(pl.col("event_type") == "round_freeze_end").iter_rows(named=True):
        if x["match_key"] in kept:
            k = (x["match_key"], int(x["round"]))
            if k not in live:
                live[k] = {"tick": int(x["tick"]), "t": x["t_score"], "ct": x["ct_score"]}
    players = collections.defaultdict(dict)
    for x in PI.iter_rows(named=True):
        if x["match_key"] in kept:
            players[(x["match_key"], int(x["round"]))][x["player_id"]] = x["team_code"]
    deaths = collections.defaultdict(list)
    for x in DE.sort("tick").iter_rows(named=True):
        if x["match_key"] in kept:
            deaths[(x["match_key"], int(x["round"]))].append({
                "tick": int(x["tick"]), "pid": x["player_id"], "team": x["player_team_code"],
                "weapon": x["weapon_name"], "attacker": x["attacker_id"]})
    plants = collections.defaultdict(list)
    for x in BS.filter(pl.col("event_type") == "bomb_planted").iter_rows(named=True):
        if x["match_key"] in kept:
            plants[(x["match_key"], int(x["round"]))].append(int(x["tick"]))

    # disconnect/takeover events with their tick, per round
    ev = collections.defaultdict(list)
    for df in (DC, BT):
        for x in df.iter_rows(named=True):
            if x["match_key"] in kept:
                ev[(x["match_key"], int(x["round"]))].append(int(x["tick"]))

    def dropped_by_disc(k, rule):
        ticks = ev.get(k, [])
        if not ticks:
            return False
        if rule == "label":
            return True
        if rule == "round_window":
            return k in rst and any(rst[k] <= t <= rd[k]["end"] for t in ticks)
        if rule == "live_window":
            return k in live and any(live[k]["tick"] <= t <= rd[k]["end"] for t in ticks)
        raise ValueError(rule)

    # duplicate groups (bake-off signature, over channel-kept matches)
    sig_parts = collections.defaultdict(list)
    for x in RE.iter_rows(named=True):
        sig_parts[x["match_key"]].append(repr(("R", int(x["round"]), int(x["tick"]), int(x["winner_team_code"]))))
    for x in DE.iter_rows(named=True):
        sig_parts[x["match_key"]].append(repr(("D", int(x["round"]), int(x["tick"]), str(x["player_id"]))))
    for x in PI.iter_rows(named=True):
        sig_parts[x["match_key"]].append(repr(("P", int(x["round"]), str(x["player_id"]), str(x["team_code"]))))
    sigs = {m: tuple(sorted(v)) for m, v in sig_parts.items()}
    groups = collections.defaultdict(list)
    for m in sorted(kept):
        groups[sigs.get(m, ())].append(m)
    dup_drop = set()
    for g in groups.values():
        if len(g) > 1:
            dup_drop.update(g[1:])

    def survivors(rule):
        surv = []
        for k in sorted(rd):
            rr = rd[k]
            pl_ = players.get(k, {})
            n_t = sum(1 for v in pl_.values() if v == T)
            n_ct = sum(1 for v in pl_.values() if v == CT)
            if rr["pc"] != 10:
                continue
            if not (len(pl_) == 10 and n_t == START and n_ct == START):
                continue
            if dropped_by_disc(k, rule):
                continue
            if k not in rst_keys or k not in live:
                continue
            if live[k]["t"] is None or live[k]["ct"] is None:
                continue
            surv.append(k)
        surv = [k for k in surv if k[0] not in dup_drop]
        return surv

    def states_for(keys):
        rows = []
        unres = 0
        for k in keys:
            rr = {**rd[k], "freeze": live[k]["tick"], "t_score": live[k]["t"], "ct_score": live[k]["ct"]}
            out = bk.build_states(k, rr, deaths.get(k, []), plants.get(k, []), players.get(k, {}),
                                  H[k[0]]["max_rounds"], collections.Counter())
            if out is None:
                unres += 1
                continue
            rows.extend(out)
        return pl.DataFrame(rows), unres

    results = {}
    base_matches = None
    for rule in ("label", "round_window", "live_window"):
        keys = survivors(rule)
        S, unres = states_for(keys)
        kept_matches = {k[0] for k in keys}
        if base_matches is None:
            base_matches = set(kept_matches)
        assert kept_matches == base_matches, f"{rule}: match set differs, split would not be comparable"
        sets = bk.make_splits(kept_matches, H)
        sp = sets[bk.PRIMARY]
        test_matches = sorted(sp["test"])
        tr = S.filter(pl.col("match_key").is_in(sorted(sp["train"])))
        va = S.filter(pl.col("match_key").is_in(sorted(sp["val"])))
        te = S.filter(pl.col("match_key").is_in(test_matches))
        Xtr, ytr = bk.xy(tr)
        Xva, yva = bk.xy(va)
        Xte, yte = bk.xy(te)
        mX = xgb.XGBClassifier(n_estimators=3000, learning_rate=0.05, max_depth=4, min_child_weight=20,
                               subsample=0.8, tree_method="hist", device="cpu",
                               monotone_constraints=tuple(bk.MONO), early_stopping_rounds=100,
                               eval_metric="logloss", random_state=bk.SEED)
        mX.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        bx = int(mX.best_iteration)
        pva = mX.predict_proba(Xva, iteration_range=(0, bx + 1))[:, 1]
        pte = mX.predict_proba(Xte, iteration_range=(0, bx + 1))[:, 1]
        val_ll = bk.metrics(yva, pva)["logloss"]
        test_ll = bk.metrics(yte, pte)["logloss"]
        te_keys = set(zip(te["match_key"].to_list(), te["round"].to_list()))
        results[rule] = {"rounds": len(keys), "matches": len(kept_matches), "states": S.height,
                         "unresolved": unres, "train_states": tr.height, "val_states": va.height,
                         "test_states": te.height, "best_iteration": bx, "val_ll": val_ll,
                         "test_ll": test_ll, "test_rounds": len({(k[0], k[1]) for k in keys if k[0] in sp["test"]})}
        results[rule]["_test_frame"] = te
        results[rule]["_pte"] = pte
        results[rule]["_yte"] = yte
        results[rule]["_model"] = mX
        print(f"[{rule}] rounds {len(keys)}, matches {len(kept_matches)}, states {S.height} "
              f"(train {tr.height}, val {va.height}, test {te.height}); unresolved {unres}; "
              f"best_iteration {bx}; val LL {val_ll:.5f}; test LL {test_ll:.5f}")

    # same-rounds comparison: test states of the baseline test matches, scored by each model
    base_te = results["label"]["_test_frame"]
    base_test_rounds = set(zip(base_te["match_key"].to_list(), base_te["round"].to_list()))
    print("\nSame-rounds comparison (baseline test rounds only):")
    for rule in ("label", "round_window", "live_window"):
        te = results[rule]["_test_frame"]
        mask = np.array([(m, r) in base_test_rounds for m, r in zip(te["match_key"].to_list(),
                                                                    te["round"].to_list())])
        Xte, yte = bk.xy(te)
        pte_all = results[rule]["_model"].predict_proba(Xte, iteration_range=(0, results[rule]["best_iteration"] + 1))[:, 1]
        ll_same = bk.metrics(yte[mask], pte_all[mask])["logloss"]
        print(f"  model [{rule}] on baseline test rounds ({int(mask.sum())} states): test LL {ll_same:.5f}")
        results[rule]["same_rounds_test_ll"] = ll_same
    print("\nSummary:")
    for rule in ("label", "round_window", "live_window"):
        r = results[rule]
        print(f"  {rule:<13} rounds {r['rounds']:>6} states {r['states']:>7} best_iter {r['best_iteration']:>4} "
              f"val LL {r['val_ll']:.5f} test LL {r['test_ll']:.5f} same-rounds test LL {r['same_rounds_test_ll']:.5f}")
    print("\nbaseline replication check: bake-off log XGBoost test LL 0.43660 and best_iteration 261")
    print("done")


if __name__ == "__main__":
    main()
