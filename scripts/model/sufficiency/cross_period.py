"""CHECK C (cross-period): XGBoost trained on the 2026 kept set, tested on the 2024-02 tome.

Reuses, read-only:
  scripts/eda/extract_state_1hz.py  bakeoff_kept() (bake-off filters; asserts the counts in
                                    logs/bakeoff.log), loaded by path.
  scripts/model/bakeoff.py          build_states, make_splits, xy, metrics, score_checks,
                                    completion_checks, keyset_keys, FEATS, MONO, PRIMARY, SEED.
New code: the 2024 relaxed round filter and the 2024 max_rounds inference (2024 header has no
max_rounds column). Writes JSON and Markdown under notes/model/sufficiency only.

Model A: XGBoost on the full 2026 kept set minus an 8% match-level early-stop slice (the bake-off
         primary validation split).
Model B: XGBoost on the bake-off primary split (in-period reference, test = 20% of 2026 matches).
Both use the bake-off XGBoost settings, early stopping 100 on the match-level slice, seed 0.
Run: pixi run python -I -X utf8 -B scripts\\model\\sufficiency\\cross_period.py
"""

import collections
import glob
import importlib.util
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"G:\pureskill\csds-cs2")
OUT_DIR = ROOT / "notes" / "model" / "sufficiency"
OUT_JSON = OUT_DIR / "cross_period.json"
OUT_MD = OUT_DIR / "cross_period.md"
TOME_2024 = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUFFIX_2024 = ".2023-12-28,2024-02-25"
HALF_EVENT = "round_announce_last_round_half"
DEFAULT_MAX_ROUNDS_2024 = 24
DENSE_2024 = ["round_end", "round_start", "round_state", "player_death", "player_info", "bomb_state"]
SPARSE_2024 = ["player_disconnect", "bot_takeover"]
T, CT, START = 2, 3, 5
GAP_LIMIT = 0.02  # verdict rule, fixed before the run: 2024 minus 2026 in-period log loss


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ex = load_module("extract_state_1hz", ROOT / "scripts" / "eda" / "extract_state_1hz.py")
bk = ex.bk
fv = bk.fv
assert bk.T == T and bk.CT == CT and bk.START == START


def half_rounds(rs):
    ev = (rs.filter(pl.col("event_type") == HALF_EVENT)
          .group_by("match_key").agg(pl.col("round").min().alias("r")))
    return {m: int(r) for m, r in ev.iter_rows()}


def signatures(RE, DE, PI):
    parts = collections.defaultdict(list)
    for x in RE.iter_rows(named=True):
        parts[x["match_key"]].append(repr(("R", str(x["round"]), str(x["tick"]),
                                           str(x["winner_team_code"]))))
    for x in DE.iter_rows(named=True):
        parts[x["match_key"]].append(repr(("D", str(x["round"]), str(x["tick"]), str(x["player_id"]))))
    for x in PI.iter_rows(named=True):
        parts[x["match_key"]].append(repr(("P", str(x["round"]), str(x["player_id"]),
                                           str(x["team_code"]))))
    return {m: tuple(sorted(v)) for m, v in parts.items()}


def set_counts(df):
    return {"matches": int(df["match_key"].n_unique()),
            "rounds": int(df.select(["match_key", "round"]).unique().height),
            "states": int(df.height),
            "ct_win_share": round(float(df["label_ct_win"].mean()), 4)}


def read_bot_takeover():
    folder = fv.TOME / f"basic_bot_takeover{fv.SUFFIX}"
    files = sorted(glob.glob(str(folder / "dataframe_*")))
    n = sum(pl.read_parquet(f).height for f in files)
    if n:
        return fv.read("bot_takeover", ["match_key", "round", "tick"])
    print("  bot_takeover: 0 rows in this tome (page files carry no columns); treated as empty")
    return pl.DataFrame(schema={"match_key": pl.String, "round": pl.Int64, "tick": pl.Int64})


def states_for(rounds, rd, live, plants, players, deaths, H):
    rows = []
    for k in rounds:
        rr = {**rd[k], "freeze": live[k]["tick"], "t_score": live[k]["t"], "ct_score": live[k]["ct"]}
        out = bk.build_states(k, rr, deaths.get(k, []), plants.get(k, []), players.get(k, {}),
                              H[k[0]]["max_rounds"], collections.Counter())
        if out is None:
            raise RuntimeError(f"unresolved victim side for {k} in a round already kept")
        rows.extend(out)
    return pl.DataFrame(rows)


def run_2026():
    print("== 2026 kept set (bake-off filters via extract_state_1hz.bakeoff_kept) ==")
    k = ex.bakeoff_kept()
    H = k["H"]
    half26 = half_rounds(fv.read("round_state", ["match_key", "round", "event_type"]))
    with_ev = [m for m in H if m in half26]
    agree26 = sum(1 for m in with_ev if half26[m] * 2 + 2 == H[m]["max_rounds"])
    rule26 = {"header_matches": len(H), "with_halftime_event": len(with_ev),
              "event_rule_agrees_with_header": agree26,
              "without_halftime_event": len(H) - len(with_ev),
              "header_max_rounds_dist": dict(collections.Counter(h["max_rounds"] for h in H.values()))}
    print(f"  max_rounds rule on 2026: event rule (2 x round + 2) agrees with header for "
          f"{agree26} of {len(with_ev)} matches with the event; {len(H) - len(with_ev)} have no event")

    RE = fv.read("round_end", ["match_key", "round", "tick", "winner_team_code"])
    DE = fv.read("player_death", ["match_key", "round", "tick", "player_id"])
    PI = fv.read("player_info", ["match_key", "round", "player_id", "team_code"])
    sigs = signatures(RE, DE, PI)

    S = states_for(k["rounds_ok"], k["rd"], k["live"], k["plants"], k["players"], k["deaths"], H)
    kept_final = k["kept_final"]
    sets = bk.make_splits(kept_final, H)[bk.PRIMARY]
    tr_b, va_b, te_b = sets["train"], sets["val"], sets["test"]
    tr_a = tr_b | te_b
    assert not (tr_a & va_b), "model A training and early-stop slice overlap"

    def sub(ms):
        return S.filter(pl.col("match_key").is_in(sorted(ms)))

    out = {"S_trA": sub(tr_a), "S_vaA": sub(va_b),
           "S_trB": sub(tr_b), "S_vaB": sub(va_b), "S_teB": sub(te_b),
           "kept_final": kept_final, "rounds_ok": k["rounds_ok"], "states": S.height,
           "sig_hash_set": {hash(sigs[m]) for m in kept_final if m in sigs},
           "rule26": rule26,
           "split_sizes": {"A_train_plus_test": len(tr_a), "A_early_stop": len(va_b),
                           "B_train": len(tr_b), "B_val": len(va_b), "B_test": len(te_b)}}
    out["counts"] = {
        "kept_matches": len(kept_final), "kept_rounds": len(k["rounds_ok"]), "states": S.height,
        "A_train": set_counts(out["S_trA"]), "A_early_stop": set_counts(out["S_vaA"]),
        "B_train": set_counts(out["S_trB"]), "B_val": set_counts(out["S_vaB"]),
        "B_test": set_counts(out["S_teB"]),
        "bakeoff_log_counts_reproduced": True,
    }
    print(f"  2026 kept: {len(kept_final)} matches, {len(k['rounds_ok'])} rounds, {S.height} states")
    return out


def run_2024():
    old = (fv.TOME, fv.SUFFIX)
    fv.TOME, fv.SUFFIX = TOME_2024, SUFFIX_2024
    try:
        return _run_2024()
    finally:
        fv.TOME, fv.SUFFIX = old


def _run_2024():
    print("== 2024-02 relaxed kept set ==")
    hdr = fv.read("header", ["key", "match_id", "match_date", "tick_rate", "map_name",
                             "t_starters_score_final", "ct_starters_score_final"])
    H0 = {r["key"]: r for r in hdr.iter_rows(named=True)}
    all_m = set(H0)
    print(f"  header matches: {len(all_m)}; max_rounds is not a header column in this tome")
    RE = fv.read("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code",
                               "player_count"])
    RST = fv.read("round_start", ["match_key", "round", "tick"])
    RS = fv.read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
    DE = fv.read("player_death", ["match_key", "round", "tick", "player_id", "player_team_code"])
    BS = fv.read("bomb_state", ["match_key", "round", "tick", "event_type"])
    DC = fv.read("player_disconnect", ["match_key", "round", "tick"])
    BT = read_bot_takeover()
    PI = fv.read("player_info", ["match_key", "round", "player_id", "team_code"])
    frames = {"round_end": RE, "round_start": RST, "round_state": RS, "player_death": DE,
              "player_info": PI, "bomb_state": BS, "player_disconnect": DC, "bot_takeover": BT}
    print("  team_change: no folder in this tome; not required for 2024 (relaxed)")

    # channel filter: dense = zero rows or missing keyset entry; sparse = keyset membership only
    chan, dropped = [], set()
    for ch in DENSE_2024 + SPARSE_2024:
        df = frames[ch]
        ks = bk.keyset_keys(ch)
        have = set(df["match_key"].unique().to_list())
        dense = ch in DENSE_2024
        zero = all_m - have
        no_ks = all_m - ks
        miss = no_ks | (zero if dense else set())
        dropped |= miss
        row = {"channel": ch, "kind": "dense" if dense else "sparse", "rows": df.height,
               "matches_with_rows": len(have), "keyset_matches": len(ks),
               "zero_row_matches": len(zero), "not_in_keyset": len(no_ks), "counted_missing": len(miss)}
        chan.append(row)
        print(f"  {ch:<20} rows={df.height:>8} matches_with_rows={len(have):>5} keyset={len(ks):>5} "
              f"zero_row={len(zero):>5} not_in_keyset={len(no_ks):>5} counted_missing={len(miss):>5}")
    kept_matches = all_m - dropped
    print(f"  matches kept after channel filter: {len(kept_matches)} of {len(all_m)}")

    rd = {}
    for x in RE.iter_rows(named=True):
        if x["match_key"] in kept_matches:
            rd[(x["match_key"], int(x["round"]))] = {"end": int(x["tick"]), "winner": x["winner_team_code"],
                                                     "pc": x["player_count"]}
    rst_keys = {(x["match_key"], int(x["round"])) for x in RST.iter_rows(named=True)
                if x["match_key"] in kept_matches}
    live = {}
    for x in RS.filter(pl.col("event_type") == "round_freeze_end").iter_rows(named=True):
        m = x["match_key"]
        if m not in kept_matches:
            continue
        k = (m, int(x["round"]))
        if k not in live:
            live[k] = {"tick": int(x["tick"]), "t": x["t_score"], "ct": x["ct_score"]}
    players = collections.defaultdict(dict)
    for x in PI.iter_rows(named=True):
        if x["match_key"] in kept_matches:
            players[(x["match_key"], int(x["round"]))][x["player_id"]] = x["team_code"]
    disc = set()
    for df in (DC, BT):
        for x in df.iter_rows(named=True):
            if x["match_key"] in kept_matches:
                disc.add((x["match_key"], int(x["round"])))
    plants = collections.defaultdict(list)
    for x in BS.filter(pl.col("event_type") == "bomb_planted").iter_rows(named=True):
        if x["match_key"] in kept_matches:
            plants[(x["match_key"], int(x["round"]))].append(int(x["tick"]))
    deaths = collections.defaultdict(list)
    for x in DE.sort("tick").iter_rows(named=True):
        if x["match_key"] in kept_matches:
            deaths[(x["match_key"], int(x["round"]))].append(
                {"tick": int(x["tick"]), "pid": x["player_id"], "team": x["player_team_code"]})

    # relaxed round filters (2024 only): no team_change, no player_count == 10
    reasons = collections.Counter()
    pc_not10_kept_matches = sum(1 for v in rd.values() if v["pc"] != 10)
    survivors = []
    for k in sorted(rd):
        pl_ = players.get(k, {})
        n_t = sum(1 for v in pl_.values() if v == T)
        n_ct = sum(1 for v in pl_.values() if v == CT)
        if not (n_t == START and n_ct == START):
            reasons["not exactly 5 T and 5 CT at start (player_info, unique player ids)"] += 1
            continue
        if k in disc:
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
    print(f"  rounds in kept matches: {len(rd)}; after relaxed round filters: {len(survivors)}")

    # duplicate matches: identical round_end, player_death and player_info content (lowest key kept)
    sigs = signatures(RE, DE, PI)
    groups = collections.defaultdict(list)
    for m in sorted(kept_matches):
        s = sigs.get(m)
        if s:
            groups[s].append(m)
    dup_groups = [g for g in groups.values() if len(g) > 1]
    dup_drop = {m for g in dup_groups for m in g[1:]}
    n_dup_rounds = sum(1 for k in survivors if k[0] in dup_drop)
    reasons["duplicate copy of a kept match (identical events)"] += n_dup_rounds
    survivors = [k for k in survivors if k[0] not in dup_drop]
    print(f"  duplicate groups: {len(dup_groups)}; duplicate matches removed: {len(dup_drop)}; "
          f"rounds removed: {n_dup_rounds}")

    # max_rounds for 2024 (header has none): 2 x (round of halftime announcement) + 2
    half24 = half_rounds(RS)
    maxr, src_all = {}, collections.Counter()
    for m in all_m:
        if m in half24:
            maxr[m] = half24[m] * 2 + 2
            src_all["halftime announcement"] += 1
        else:
            maxr[m] = DEFAULT_MAX_ROUNDS_2024
            src_all["default 24 (no halftime announcement)"] += 1
    H24 = {m: {"max_rounds": maxr[m], "match_date": H0[m]["match_date"],
               "t_starters_score_final": H0[m]["t_starters_score_final"],
               "ct_starters_score_final": H0[m]["ct_starters_score_final"]} for m in all_m}
    print(f"  2024 max_rounds source (all header matches): {dict(src_all)}; "
          f"dist: {dict(collections.Counter(v['max_rounds'] for v in H24.values()))}")

    rows, rounds_ok = [], []
    for k in survivors:
        rr = {**rd[k], "freeze": live[k]["tick"], "t_score": live[k]["t"], "ct_score": live[k]["ct"]}
        out = bk.build_states(k, rr, deaths.get(k, []), plants.get(k, []), players.get(k, {}),
                              H24[k[0]]["max_rounds"], collections.Counter())
        if out is None:
            reasons["death with unresolved victim side in live window"] += 1
            continue
        rows.extend(out)
        rounds_ok.append(k)
    S24 = pl.DataFrame(rows)
    kept_final = {k[0] for k in rounds_ok}
    src_kept = collections.Counter("halftime announcement" if k[0] in half24
                                   else "default 24 (no halftime announcement)" for k in rounds_ok)
    print(f"  2024 kept: {len(kept_final)} matches, {len(rounds_ok)} rounds, {S24.height} states")
    for name, n in reasons.most_common():
        print(f"    dropped: {name}: {n}")

    # checks on all 2024 matches (not only kept), as in the bake-off
    RS_frz = RS.filter((pl.col("event_type") == "round_freeze_end") & pl.col("t_score").is_not_null()
                       & pl.col("ct_score").is_not_null())
    n_frz_null = RS.filter((pl.col("event_type") == "round_freeze_end")
                           & (pl.col("t_score").is_null() | pl.col("ct_score").is_null())).height
    sc = bk.score_checks(RE, RS_frz, H24)
    comp, comp_ex = bk.completion_checks(RE, H24)
    print(f"  score transitions (all 2024 matches): {sc['transitions']}; freeze rows with null score: {n_frz_null}")

    counts = {
        "header_matches": len(all_m),
        "rounds_in_round_end": RE.height,
        "channel_kept_matches": len(kept_matches),
        "dropped_missing_channel_matches": len(dropped),
        "rounds_in_channel_kept_matches": len(rd),
        "duplicate_matches_removed": len(dup_drop),
        "duplicate_groups": len(dup_groups),
        "matches_with_no_kept_round": len(kept_matches - kept_final - dup_drop),
        "kept_matches": len(kept_final),
        "kept_rounds": len(rounds_ok),
        "states": S24.height,
        "live_start_states": int((S24["kind"] == "live_start").sum()),
        "death_states": int((S24["kind"] == "death").sum()),
        "ct_win_share_states": round(float(S24["label_ct_win"].mean()), 4),
        "kept_rounds_with_total_players_not_10": sum(1 for k in rounds_ok if len(players.get(k, {})) != 10),
        "kept_rounds_with_round_end_player_count_not_10_(not filtered)": sum(
            1 for k in rounds_ok if rd[k]["pc"] != 10),
        "round_end_player_count_not_10_in_channel_kept_rounds": pc_not10_kept_matches,
        "freeze_rows_with_null_score_all_matches": n_frz_null,
        "max_rounds_source_kept_matches": dict(src_kept),
        "max_rounds_source_all_header_matches": dict(src_all),
    }
    return {"S": S24, "counts": counts, "chan": chan, "reasons": dict(reasons),
            "sc": sc, "comp": dict(comp), "kept_final": kept_final,
            "sig_hash": {m: hash(sigs[m]) for m in kept_final if m in sigs}}


def fit_xgb(Xtr, ytr, Xva, yva):
    params = dict(n_estimators=3000, learning_rate=0.05, max_depth=4, min_child_weight=20,
                  subsample=0.8, tree_method="hist", monotone_constraints=tuple(bk.MONO),
                  early_stopping_rounds=100, eval_metric="logloss", random_state=bk.SEED)
    errors = []
    for device in ("cuda", "cpu"):
        try:
            model = xgb.XGBClassifier(device=device, **params)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            fell_back = any("Device is changed from GPU to CPU" in str(w.message) for w in caught)
            used = "cpu (cuda requested; XGBoost found no usable GPU and switched to CPU)" \
                if (device == "cuda" and fell_back) else device
            bi = int(model.best_iteration)
            print(f"  XGBoost requested device={device}; ran on {used}; best_iteration={bi}")
            return model, used, bi, errors
        except Exception as e:  # noqa: BLE001
            msg = f"device={device}: {type(e).__name__}: {str(e)[:300]}"
            errors.append(msg)
            print(f"  {msg}")
    raise RuntimeError("XGBoost failed on cuda and cpu: " + " | ".join(errors))


def predict(model, X, bi):
    return model.predict_proba(X, iteration_range=(0, bi + 1))[:, 1]


def evalrow(model, set_name, df, p, label_note="VERIFIED"):
    y = df["label_ct_win"].to_numpy().astype(np.int64)
    m = bk.metrics(y, p)
    cnt = set_counts(df)
    return {"model": model, "set": set_name, **cnt,
            "logloss": m["logloss"], "brier": m["brier"], "auc": m["auc"], "ece": m["ece"],
            "acc": m["acc"], "table": m["table"], "label": label_note}


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, float) and math.isnan(o):
        return None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o


def f5(x):
    return "n/a" if x is None else f"{x:.5f}"


def f4(x):
    return "n/a" if x is None else f"{x:.4f}"


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    A = run_2026()
    B = run_2024()

    # ---- models
    Xtr_a, ytr_a = bk.xy(A["S_trA"])
    Xva_a, yva_a = bk.xy(A["S_vaA"])
    mA, devA, biA, errA = fit_xgb(Xtr_a, ytr_a, Xva_a, yva_a)
    X24, y24 = bk.xy(B["S"])
    p24_xgb = predict(mA, X24, biA)
    pva_xgb_A = predict(mA, Xva_a, biA)

    Xtr_b, ytr_b = bk.xy(A["S_trB"])
    Xva_b, yva_b = bk.xy(A["S_vaB"])
    Xte_b, yte_b = bk.xy(A["S_teB"])
    mB, devB, biB, errB = fit_xgb(Xtr_b, ytr_b, Xva_b, yva_b)
    pte_xgb_B = predict(mB, Xte_b, biB)

    # ---- baselines (same training sets)
    ci = bk.FEATS.index("ct_alive_minus_t")
    rate_a, rate_b = float(ytr_a.mean()), float(ytr_b.mean())
    p24_const = np.full(len(y24), rate_a)
    pte_const = np.full(len(yte_b), rate_b)
    lrA = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr_a[:, [ci]], ytr_a)
    p24_lr = lrA.predict_proba(X24[:, [ci]])[:, 1]
    lrB = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr_b[:, [ci]], ytr_b)
    pte_lr = lrB.predict_proba(Xte_b[:, [ci]])[:, 1]

    S24 = B["S"]
    S_teB = A["S_teB"]
    live24 = (S24["kind"] == "live_start").to_numpy()
    liveB = (S_teB["kind"] == "live_start").to_numpy()
    S24_live = S24.filter(pl.col("kind") == "live_start")
    S_teB_live = S_teB.filter(pl.col("kind") == "live_start")

    r24_xgb = evalrow("XGBoost (A)", "2024-02 relaxed kept set, all states", S24, p24_xgb)
    r24_xgb_live = evalrow("XGBoost (A)", "2024-02 relaxed kept set, live-start states", S24_live,
                           p24_xgb[live24])
    r24_const = evalrow("Constant (A train rate)", "2024-02 relaxed kept set, all states", S24, p24_const)
    r24_lr = evalrow("LR ct_alive_minus_t (A)", "2024-02 relaxed kept set, all states", S24, p24_lr)
    r26_xgb = evalrow("XGBoost (B)", "2026 primary test (in-period, 20%), all states", S_teB, pte_xgb_B)
    r26_xgb_live = evalrow("XGBoost (B)", "2026 primary test (in-period, 20%), live-start states",
                           S_teB_live, pte_xgb_B[liveB])
    r26_const = evalrow("Constant (B train rate)", "2026 primary test, all states", S_teB, pte_const)
    r26_lr = evalrow("LR ct_alive_minus_t (B)", "2026 primary test, all states", S_teB, pte_lr)
    rva_A = evalrow("XGBoost (A)", "2026 early-stop slice (8%; used for stopping, not clean)",
                    A["S_vaA"], pva_xgb_A)
    rows = [r24_xgb, r24_xgb_live, r24_const, r24_lr, r26_xgb, r26_xgb_live, r26_const, r26_lr, rva_A]

    gap = r24_xgb["logloss"] - r26_xgb["logloss"]
    beats = r24_xgb["logloss"] < r24_const["logloss"]
    if not beats:
        verdict = (f"Does not transfer: 2024 XGBoost log loss {f5(r24_xgb['logloss'])} is not better than "
                   f"the constant baseline {f5(r24_const['logloss'])}.")
    elif gap <= GAP_LIMIT:
        verdict = (f"Transfers within the rule: 2024 log loss {f5(r24_xgb['logloss'])} vs 2026 in-period "
                   f"{f5(r26_xgb['logloss'])} (gap {gap:+.5f}, limit {GAP_LIMIT}), constant baseline "
                   f"{f5(r24_const['logloss'])}. 2024 round_end winners UNVERIFIED.")
    else:
        verdict = (f"Partial transfer: 2024 log loss {f5(r24_xgb['logloss'])} vs 2026 in-period "
                   f"{f5(r26_xgb['logloss'])} (gap {gap:+.5f} exceeds {GAP_LIMIT}), better than the "
                   f"constant baseline {f5(r24_const['logloss'])}. 2024 round_end winners UNVERIFIED.")
    print("\n== verdict ==")
    print(verdict)

    cross_tome = sum(1 for m, h in B["sig_hash"].items() if h in A["sig_hash_set"])
    print(f"cross-tome duplicate matches (2024 kept vs 2026 kept, identical events): {cross_tome}")

    result = {
        "check": "C cross-period (file: cross_period)",
        "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "label_rule": "Every number is VERIFIED (computed in this run). Items marked UNVERIFIED are not checked.",
        "devices": {"model_A": devA, "model_B": devB, "model_A_errors": errA, "model_B_errors": errB},
        "best_iteration": {"model_A": biA, "model_B": biB},
        "split_sizes_2026": A["split_sizes"],
        "counts_2026": A["counts"],
        "max_rounds_rule_2026": A["rule26"],
        "counts_2024": B["counts"],
        "channels_2024": B["chan"],
        "drop_reasons_2024": B["reasons"],
        "score_checks_2024": B["sc"],
        "completion_checks_2024": B["comp"],
        "cross_tome_duplicate_matches": cross_tome,
        "metrics": rows,
        "verdict_rule": f"transfers if 2024 XGBoost log loss < constant and gap to 2026 in-period <= {GAP_LIMIT}",
        "verdict": verdict,
        "runtime_s": round(time.time() - t0, 1),
    }
    OUT_JSON.write_text(json.dumps(clean(result), indent=1, ensure_ascii=False), encoding="utf-8")

    # ---- markdown
    c26, c24 = A["counts"], B["counts"]
    md = []
    md.append("# CHECK C: cross-period transfer (2026 kept set to 2024-02 tome)")
    md.append("")
    md.append(f"Run: {result['run_at']}. Script: `scripts/model/sufficiency/cross_period.py` (seed 0, "
              f"bake-off XGBoost settings: hist, depth 4, eta 0.05, min_child_weight 20, subsample 0.8, "
              f"early stopping 100, monotone MONO). Device: A={devA}, B={devB}.")
    md.append("")
    md.append(f"**Verdict:** {verdict}")
    md.append("")
    md.append("## 1. Counts (label VERIFIED)")
    md.append("")
    md.append("| Set | Matches | Rounds | States | CT-win share (states) | Label |")
    md.append("|---|---:|---:|---:|---:|---|")
    for name, key in (("2026 kept set (bake-off filters; reproduces logs/bakeoff.log)", None),
                      ("2026 A train = primary train + test (92%)", "A_train"),
                      ("2026 A early-stop slice = primary val (8%)", "A_early_stop"),
                      ("2026 B train = primary train (72%)", "B_train"),
                      ("2026 B val = primary val (8%)", "B_val"),
                      ("2026 B test = primary test (20%, in-period)", "B_test")):
        if key is None:
            md.append(f"| {name} | {c26['kept_matches']} | {c26['kept_rounds']} | {c26['states']} | - | VERIFIED |")
        else:
            c = c26[key]
            md.append(f"| {name} | {c['matches']} | {c['rounds']} | {c['states']} | {f4(c['ct_win_share'])} | VERIFIED |")
    md.append(f"| 2024-02 kept set (relaxed filter; test for A) | {c24['kept_matches']} | {c24['kept_rounds']} | "
              f"{c24['states']} | {f4(c24['ct_win_share_states'])} | VERIFIED (winner labels UNVERIFIED) |")
    md.append("")
    md.append(f"2024 header matches: {c24['header_matches']}; 2026 header matches: {A['rule26']['header_matches']}.")
    md.append("")
    md.append("## 2. Metrics (states, bake-off convention; label VERIFIED)")
    md.append("")
    md.append("| Model | Set | Matches | Rounds | States | Log loss | Brier | AUC | ECE (10 bins) | Accuracy (ref) | Label |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        md.append(f"| {r['model']} | {r['set']} | {r['matches']} | {r['rounds']} | {r['states']} | "
                  f"{f5(r['logloss'])} | {f5(r['brier'])} | {f4(r['auc'])} | {f4(r['ece'])} | "
                  f"{f4(r['acc'])} | {r['label']} |")
    md.append("")
    md.append(f"Reference: the bake-off's XGBoost primary test row (`notes/model/bakeoff_results.md`) is "
              f"log loss 0.4366 on 47716 states; Model B on the same split gives "
              f"{f5(r26_xgb['logloss'])} on {r26_xgb['states']} states.")
    md.append("")
    md.append("## 3. 2024 filter and data checks (label VERIFIED unless marked)")
    md.append("")
    md.append("| Channel | Kind | Rows | Matches with rows | Keyset matches | Zero-row matches | Counted missing |")
    md.append("|---|---|---:|---:|---:|---:|---:|")
    for r in B["chan"]:
        md.append(f"| {r['channel']} | {r['kind']} | {r['rows']} | {r['matches_with_rows']} | "
                  f"{r['keyset_matches']} | {r['zero_row_matches']} | {r['counted_missing']} |")
    md.append("")
    md.append("team_change: no folder in the 2024 tome, so not required (relaxed). Relaxed round rules: "
              "player_count == 10 not required; player_info must show 5 T and 5 CT by unique player id; "
              "disconnect or bot takeover drops the round; then round_start, live start and score checks as in the bake-off.")
    md.append("")
    md.append("| Stage | Matches | Rounds |")
    md.append("|---|---:|---:|")
    md.append(f"| 2024 header | {c24['header_matches']} | {c24['rounds_in_round_end']} (round_end rows) |")
    md.append(f"| dropped: missing a required channel | {c24['dropped_missing_channel_matches']} | - |")
    md.append(f"| rounds in channel-kept matches | {c24['channel_kept_matches']} | {c24['rounds_in_channel_kept_matches']} |")
    for name, n in sorted(B["reasons"].items(), key=lambda x: -x[1]):
        md.append(f"| dropped round: {name} | - | {n} |")
    md.append(f"| **kept (used)** | **{c24['kept_matches']}** | **{c24['kept_rounds']}** |")
    md.append("")
    md.append(f"States: {c24['states']} ({c24['live_start_states']} live starts, {c24['death_states']} death "
              f"states). Kept rounds with total unique players not 10: {c24['kept_rounds_with_total_players_not_10']}. "
              f"Kept rounds where round_end player_count is not 10 (not filtered): "
              f"{c24['kept_rounds_with_round_end_player_count_not_10_(not filtered)']}.")
    md.append("")
    md.append(f"- **max_rounds (2024, inferred, not in header):** halftime announcement rule 2 x round + 2. "
              f"The same rule matches the 2026 header in {A['rule26']['event_rule_agrees_with_header']} of "
              f"{A['rule26']['with_halftime_event']} matches with the event (VERIFIED on 2026). "
              f"2024 source for kept matches: {c24['max_rounds_source_kept_matches']}. "
              f"max_rounds only drives is_pistol_round and is_overtime.")
    md.append(f"- **Score transitions (partial evidence only, 2024 winners UNVERIFIED):** "
              f"{B['sc']['transitions']} across all 2024 matches; inconsistent rounds by (round, max_rounds, swap): "
              f"{B['sc']['inconsistent_by_round'][:8]}.")
    md.append(f"- **Completion (round_end count vs header final score sum):** "
              f"{B['comp']}.")
    md.append(f"- **Duplicates:** {c24['duplicate_groups']} groups, {c24['duplicate_matches_removed']} copies removed "
              f"inside 2024; {cross_tome} kept 2024 matches have identical events to a kept 2026 match.")
    md.append("")
    md.append("## 4. Caveats and failures")
    md.append("")
    md.append("- 2024 round_end winners are UNVERIFIED. Results on 2024 depend on them.")
    md.append("- 2024 has no max_rounds in its header; inferred as above. Matches with no halftime announcement "
              f"({c24['max_rounds_source_all_header_matches'].get('default 24 (no halftime announcement)', 0)} "
              "matches, all short) use 24 by default.")
    md.append("- Model A and Model B differ in training size (92% vs 72% of 2026 matches); the 2026 in-period "
              "comparison is therefore not an exact control.")
    md.append("- Early-stop slices are match-level; the 2026 early-stop row is optimistic and not a clean test.")
    md.append("- No tuning, no calibration; the same bake-off settings as the 2026 primary run. Metrics are on "
              "states (long rounds weigh more); the live-start rows give one state per round.")
    md.append("- The bot_takeover page files in the 2024 tome have 0 rows and no columns; treated as empty.")
    md.append("")
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
