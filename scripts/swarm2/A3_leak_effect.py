"""A3: size of the cross-split leakage that the dedupe removed (run1 split, 2026 tome), read-only.

Uses the bake-off's own functions by import (scripts/model/bakeoff.py has a __main__ guard, so
importing it runs no pipeline and writes no output). The run1 set is the 1,568 matches with a
surviving round before dedupe, split with bakeoff.make_splits. Checks:
  - states 241,676 and rounds 29,596 and matches 1,568 (logs/bakeoff_run1_no_dedupe.log);
  - XGBoost primary test log loss 0.43353 and validation 0.44219 (same log).
Then the experiment: remove from train and validation the copies of the 12 test matches that have
an exact twin there, refit XGBoost with the same settings, and compare test log loss on the full
test set and on the exposed matches. ASSUMPTION: this isolates cross-boundary duplicate leakage
for XGBoost only; it does not measure the rest of the bake-off.
"""
import collections
import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]

spec = importlib.util.spec_from_file_location("bakeoff", ROOT / "scripts" / "model" / "bakeoff.py")
bo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bo)
fv = bo.fv
T, CT, START = bo.T, bo.CT, bo.START
DENSE, SPARSE = bo.DENSE, bo.SPARSE


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
        have = set(frames[ch]["match_key"].unique().to_list())
        ks = bo.keyset_keys(ch)
        miss = (all_matches - ks) | ((all_matches - have) if ch in DENSE else set())
        for m in miss:
            missing[m].append(ch)
    kept = all_matches - set(missing)
    print(f"channel-kept matches: {len(kept)} (logged 1709)")

    rd = {}
    for x in RE.iter_rows(named=True):
        if x["match_key"] in kept:
            rd[(x["match_key"], int(x["round"]))] = {"end": int(x["tick"]), "winner": x["winner_team_code"],
                                                     "pc": x["player_count"]}
    rst_keys = {(x["match_key"], int(x["round"])) for x in RST.iter_rows(named=True)
                if x["match_key"] in kept}
    live = {}
    for x in RS.filter(pl.col("event_type") == "round_freeze_end").iter_rows(named=True):
        if x["match_key"] in kept:
            live.setdefault((x["match_key"], int(x["round"])), {"tick": int(x["tick"]), "t": x["t_score"],
                                                                "ct": x["ct_score"]})
    players = collections.defaultdict(dict)
    for x in PI.iter_rows(named=True):
        if x["match_key"] in kept:
            players[(x["match_key"], int(x["round"]))][x["player_id"]] = x["team_code"]
    disc = set()
    for df in (DC, BT):
        for x in df.iter_rows(named=True):
            if x["match_key"] in kept:
                disc.add((x["match_key"], int(x["round"])))
    plants = collections.defaultdict(list)
    for x in BS.filter(pl.col("event_type") == "bomb_planted").iter_rows(named=True):
        if x["match_key"] in kept:
            plants[(x["match_key"], int(x["round"]))].append(int(x["tick"]))
    deaths = collections.defaultdict(list)
    for x in DE.sort("tick").iter_rows(named=True):
        if x["match_key"] in kept:
            deaths[(x["match_key"], int(x["round"]))].append({"tick": int(x["tick"]), "pid": x["player_id"],
                                                              "team": x["player_team_code"],
                                                              "weapon": x["weapon_name"],
                                                              "attacker": x["attacker_id"]})
    survivors = []
    for k in sorted(rd):
        rr = rd[k]
        pl_ = players.get(k, {})
        n_t = sum(1 for v in pl_.values() if v == T)
        n_ct = sum(1 for v in pl_.values() if v == CT)
        if rr["pc"] != 10 or not (len(pl_) == 10 and n_t == START and n_ct == START):
            continue
        if k in disc or k not in rst_keys or k not in live:
            continue
        if live[k]["t"] is None or live[k]["ct"] is None:
            continue
        survivors.append(k)

    state_rows = []
    rounds_ok = []
    diag = collections.Counter()
    for k in survivors:
        rr = {**rd[k], "freeze": live[k]["tick"], "t_score": live[k]["t"], "ct_score": live[k]["ct"]}
        out = bo.build_states(k, rr, deaths.get(k, []), plants.get(k, []), players.get(k, {}),
                              H[k[0]]["max_rounds"], diag)
        if out is None:
            continue
        state_rows.extend(out)
        rounds_ok.append(k)
    S = pl.DataFrame(state_rows)
    pre_final = {k[0] for k in rounds_ok}
    print(f"run1 set: {len(pre_final)} matches, {len(rounds_ok)} rounds, {S.height} states "
          f"(logged 1568 matches, 29596 rounds, 241676 states)")
    assert len(pre_final) == 1568 and len(rounds_ok) == 29596 and S.height == 241676

    # exact copy groups (bakeoff rule) over all matches
    parts = collections.defaultdict(list)
    for x in RE.iter_rows(named=True):
        parts[x["match_key"]].append(repr(("R", int(x["round"]), int(x["tick"]), int(x["winner_team_code"]))))
    for x in DE.iter_rows(named=True):
        parts[x["match_key"]].append(repr(("D", int(x["round"]), int(x["tick"]), str(x["player_id"]))))
    for x in PI.iter_rows(named=True):
        parts[x["match_key"]].append(repr(("P", int(x["round"]), str(x["player_id"]), str(x["team_code"]))))
    sigs = {m: tuple(sorted(v)) for m, v in parts.items()}
    groups = collections.defaultdict(list)
    for m in kept:
        groups[sigs.get(m, ())].append(m)
    twin = collections.defaultdict(set)
    for g in groups.values():
        if len(g) > 1:
            for m in g:
                twin[m].update(x for x in g if x != m)

    sets = bo.make_splits(pre_final, H)[bo.PRIMARY]
    print(f"run1 primary split: train {len(sets['train'])}, val {len(sets['val'])}, "
          f"test {len(sets['test'])} (logged 1129/125/314)")
    assert (len(sets["train"]), len(sets["val"]), len(sets["test"])) == (1129, 125, 314)
    tr_all = sets["train"] | sets["val"]
    exposed = sorted(m for m in sets["test"] if twin.get(m, set()) & tr_all)
    print(f"exposed test matches (exact twin in train or val): {len(exposed)}")
    if len(exposed) != 12:
        print("NOTE: exposed count differs from the earlier audit (12)")

    def split_frames(train_set, val_set):
        tr = S.filter(pl.col("match_key").is_in(sorted(train_set)))
        va = S.filter(pl.col("match_key").is_in(sorted(val_set)))
        te = S.filter(pl.col("match_key").is_in(sorted(sets["test"])))
        return tr, va, te

    # with duplicates (run1 as logged)
    tr, va, te = split_frames(sets["train"], sets["val"])
    res, _, (Xva, yva, Xte, yte) = bo.fit_models(tr, va, te, None)
    x_with = res["XGBoost"]
    m_with = bo.metrics(yte, x_with["pte"])
    v_with = bo.metrics(yva, x_with["pva"])
    print(f"\nrun1 reproduction, XGBoost: val LL {v_with['logloss']:.5f} (logged 0.44219); "
          f"test LL {m_with['logloss']:.5f} (logged 0.43353); {x_with['info']}")

    exp_mask = te["match_key"].is_in(exposed).to_numpy()
    p_with = np.asarray(x_with["pte"])
    ll_exp_with = bo.metrics(yte[exp_mask], p_with[exp_mask])["logloss"]
    ll_rest_with = bo.metrics(yte[~exp_mask], p_with[~exp_mask])["logloss"]
    print(f"  test rows: all {len(yte)}, exposed {int(exp_mask.sum())} states, other {int((~exp_mask).sum())}")
    print(f"  test LL exposed matches {ll_exp_with:.5f}; other matches {ll_rest_with:.5f}")

    # without the train-side and val-side copies of the exposed test matches
    drop_train = set()
    for m in exposed:
        drop_train.update(twin[m] & tr_all)
    print(f"\nremoving {len(drop_train)} train/val copies of the exposed test matches")
    tr2, va2, te2 = split_frames(sets["train"] - drop_train, sets["val"] - drop_train)
    res2, _, (_, _, _, yte2) = bo.fit_models(tr2, va2, te2, None)
    x2 = res2["XGBoost"]
    m_wo = bo.metrics(yte2, x2["pte"])
    p_wo = np.asarray(x2["pte"])
    ll_exp_wo = bo.metrics(yte2[exp_mask], p_wo[exp_mask])["logloss"]
    ll_rest_wo = bo.metrics(yte2[~exp_mask], p_wo[~exp_mask])["logloss"]
    print(f"  XGBoost test LL all {m_wo['logloss']:.5f} (with copies {m_with['logloss']:.5f}, "
          f"difference {m_wo['logloss'] - m_with['logloss']:+.5f})")
    print(f"  exposed matches: {ll_exp_wo:.5f} without copies vs {ll_exp_with:.5f} with copies "
          f"(difference {ll_exp_wo - ll_exp_with:+.5f})")
    print(f"  other matches: {ll_rest_wo:.5f} vs {ll_rest_with:.5f} "
          f"(difference {ll_rest_wo - ll_rest_with:+.5f}; not expected to change much)")
    print(f"  {x2['info']}")


if __name__ == "__main__":
    main()
