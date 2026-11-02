"""Team-correct CT-minus-T score lead, and the baseline XGBoost rerun with the old and corrected feature.

Question
  bakeoff.py feeds ct_score_minus_t_score = ct_score - t_score (round_state, live start) to the
  models. Notes say these fields are side-attached in most matches and team-attached in a minority,
  which would flip the sign after a side swap. This script derives the lead from round winners and
  the swap rule, compares it with the round_state difference per round, and reruns the baseline
  XGBoost on the bake-off primary split with the old and the corrected feature.

Steps
  1. Kept set: extract_state_1hz.bakeoff_kept() (bake-off filters and duplicate removal, ported; it
     asserts the counts in logs/bakeoff.log). Read only.
  2. Derived lead. For each channel-kept match, team A starts on T and team B on CT. Each round's
     winner (round_end.winner_team_code) adds one to the winning team. After round r the teams
     swap sides when bakeoff.swap_expected(r, max_rounds) is true. The lead before round r is
     score(team on CT) - score(team on T). Final team scores are checked against the header's
     t_starters_score_final and ct_starters_score_final, for matches whose round count equals the
     header total.
  3. Agreement per round: derived lead vs round_state ct_score - t_score at round_freeze_end.
  4. Features. S_old is bakeoff.build_states as written. S_new is the same state rows with
     ct_score_minus_t_score replaced by the derived lead of the round. Nothing else changes.
  5. Split: bakeoff.make_splits(kept, H)[PRIMARY], the same match sets for both features.
  6. XGBoost with the bake-off XGBoost settings (copied from bakeoff.fit_models). device cuda, falling
     back to cpu. A cpu run of both features is also reported. The old-feature cpu run should
     reproduce logs/bakeoff.log (test log loss 0.43660, Brier 0.14637) if the pipeline matches.
  7. Writes notes\\model\\sufficiency\\score_lead_fix.json (numbers only) and prints a summary.

Reuses (read only): scripts/eda/extract_state_1hz.py, scripts/model/bakeoff.py. Edits nothing.

Run from the project root:
  pixi run python -I -X utf8 scripts\\model\\sufficiency\\score_lead_fix.py
"""

import collections
import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"G:\pureskill\csds-cs2")
OUT_DIR = ROOT / "notes" / "model" / "sufficiency"
OUT_JSON = OUT_DIR / "score_lead_fix.json"
SEED = 0
N_BOOT = 2000
OLD_COL = "ct_score_minus_t_score"
# logs/bakeoff.log, primary split, XGBoost raw (CPU run)
BAKEOFF_XGB_TEST = {"logloss": 0.43660, "brier": 0.14637}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ex = load_module("extract_state_1hz", ROOT / "scripts" / "eda" / "extract_state_1hz.py")
bk = ex.bk
T, CT, FEATS, MONO = bk.T, bk.CT, bk.FEATS, bk.MONO
assert OLD_COL in FEATS


def phase(r, mr):
    if r <= mr // 2:
        return "half_1"
    if r <= mr:
        return "half_2"
    return "overtime"


def derive_leads(rd, H):
    """Team A starts on T, team B on CT. lead[(match, round)] = score(team on CT) - score(team on T)
    before the round, from round winners and the swap rule. final[match] = (A, B, rounds)."""
    by_match = collections.defaultdict(dict)
    for (m, r), v in rd.items():
        by_match[m][r] = v["winner"]
    lead, final, st = {}, {}, collections.Counter()
    for m, rounds in by_match.items():
        mr = H[m]["max_rounds"]
        rs = sorted(rounds)
        if rs != list(range(1, len(rs) + 1)):
            st["matches_with_round_gap"] += 1
        side_a = T
        a = b = 0
        for r in rs:
            lead[(m, r)] = (b - a) if side_a == T else (a - b)
            w = rounds[r]
            if w == side_a:
                a += 1
            elif w == (CT if side_a == T else T):
                b += 1
            else:
                st["winner_not_T_or_CT"] += 1
            if bk.swap_expected(r, mr):
                side_a = CT if side_a == T else T
                st["swaps_applied"] += 1
        final[m] = (a, b, len(rs))
    return lead, final, st


def header_check(final, H):
    out = collections.Counter()
    for m, (a, b, n) in final.items():
        ht, hc = H[m]["t_starters_score_final"], H[m]["ct_starters_score_final"]
        if n != ht + hc:
            out["matches_round_count_not_header_total"] += 1
            continue
        out["complete_matches"] += 1
        if (a, b) == (ht, hc):
            out["A_B_equal_header_T_CT"] += 1
        elif (a, b) == (hc, ht):
            out["A_B_equal_header_swapped"] += 1
        else:
            out["neither"] += 1
    return out


def agreement(lead, live, kept_set):
    cats_all = collections.Counter()
    cats_kept = collections.Counter()
    match_any_all, match_any_kept = set(), set()
    for (m, r), L in lead.items():
        k = (m, r)
        if k not in live:
            cats_all["no_live_start"] += 1
            continue
        t, c = live[k]["t"], live[k]["ct"]
        if t is None or c is None:
            cats_all["null_score"] += 1
            continue
        old = c - t
        if L == old:
            cat = "agree"
        elif old != 0 and L == -old:
            cat = "sign_flip"
        else:
            cat = "other"
        cats_all[cat] += 1
        if cat != "agree":
            match_any_all.add(m)
        if k in kept_set:
            cats_kept[cat] += 1
            if cat != "agree":
                match_any_kept.add(m)
    return {"all_rounds": dict(cats_all), "kept_rounds": dict(cats_kept),
            "matches_any_disagree_all": len(match_any_all),
            "matches_any_disagree_kept": len(match_any_kept)}, match_any_kept


def phase_agreement(lead, live, H):
    out = collections.defaultdict(collections.Counter)
    for (m, r), L in lead.items():
        k = (m, r)
        if k not in live or live[k]["t"] is None or live[k]["ct"] is None:
            continue
        old = live[k]["ct"] - live[k]["t"]
        mr = H[m]["max_rounds"]
        key = phase(r, mr)
        out[key]["rounds"] += 1
        out[key]["agree"] += int(L == old)
        out[key]["sign_flip"] += int(old != 0 and L == -old)
    return {k: dict(v) for k, v in out.items()}


def build_rows(kept, lead):
    H = kept["H"]
    rows_old, rows_new = [], []
    for k in kept["rounds_ok"]:
        m = k[0]
        rr = {**kept["rd"][k], "freeze": kept["live"][k]["tick"], "t_score": kept["live"][k]["t"],
              "ct_score": kept["live"][k]["ct"]}
        out = bk.build_states(k, rr, kept["deaths"].get(k, []), kept["plants"].get(k, []),
                              kept["players"].get(k, {}), H[m]["max_rounds"], collections.Counter())
        assert out is not None, "kept round with unresolved victim side"
        rows_old.extend(out)
        L = lead[k]
        rows_new.extend({**s, OLD_COL: L} for s in out)
    return pl.DataFrame(rows_old), pl.DataFrame(rows_new)


def xy(df):
    X = df.select(FEATS).to_numpy().astype(np.float64)
    y = df["label_ct_win"].to_numpy().astype(np.int64)
    return X, y


def fit_xgb(tr, va, te, device):
    """bakeoff.fit_models XGBoost block, unchanged settings. Returns (test probabilities,
    best_iteration, fell_back_to_cpu)."""
    Xtr, ytr = xy(tr)
    Xva, yva = xy(va)
    Xte, _ = xy(te)
    m = xgb.XGBClassifier(n_estimators=3000, learning_rate=0.05, max_depth=4, min_child_weight=20,
                          subsample=0.8, tree_method="hist", device=device,
                          monotone_constraints=tuple(MONO), early_stopping_rounds=100,
                          eval_metric="logloss", random_state=SEED)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    fell_back = any("Device is changed" in str(w.message) for w in caught)
    bx = int(m.best_iteration)
    p = m.predict_proba(Xte, iteration_range=(0, bx + 1))[:, 1]
    return p, bx, fell_back


def fit_device(tr, va, te, device):
    try:
        p, bx, fb = fit_xgb(tr, va, te, device)
    except Exception as exc:
        print(f"    {device} raised ({type(exc).__name__}: {str(exc)[:160]}); using cpu", flush=True)
        p, bx, _ = fit_xgb(tr, va, te, "cpu")
        return p, bx, "cpu (cuda raised)"
    if device == "cuda" and fb:
        return p, bx, "cpu (cuda requested; XGBoost used CPU)"
    return p, bx, device


def ll_vec(p, y):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def paired_delta(te, p_old, p_new, y):
    """Log loss (new) minus log loss (old), state-weighted, cluster bootstrap over test matches."""
    d = ll_vec(p_new, y) - ll_vec(p_old, y)
    uniq = sorted(te["match_key"].unique().to_list())
    pos = {m: i for i, m in enumerate(uniq)}
    g = np.array([pos[m] for m in te["match_key"].to_list()])
    M = len(uniq)
    sum_d = np.bincount(g, weights=d, minlength=M)
    cnt = np.bincount(g, minlength=M).astype(np.float64)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, M, size=(N_BOOT, M))
    boot = sum_d[idx].sum(axis=1) / cnt[idx].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"delta": float(d.mean()), "ci95": [float(lo), float(hi)],
            "share_resamples_below_0": float((boot < 0).mean())}


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("== Score lead fix: team-correct CT-minus-T lead ==")
    kept = ex.bakeoff_kept()  # prints and asserts the counts against logs/bakeoff.log
    H, rd, live = kept["H"], kept["rd"], kept["live"]
    kept_set = set(kept["rounds_ok"])

    # ---------- 2. derived lead
    lead, final, st = derive_leads(rd, H)
    print(f"\n== 2. Derived lead ==\nrounds with a derived lead (channel-kept matches): {len(lead)}; "
          f"matches: {len({m for m, _ in lead})}; diagnostics: {dict(st)}")
    hc = header_check(final, H)
    print(f"header check (final team scores vs header, complete matches only): {dict(hc)}")

    # ---------- 3. agreement with round_state
    agr, match_any_kept = agreement(lead, live, kept_set)
    print("\n== 3. Agreement: derived lead vs round_state (ct_score - t_score at live start) ==")
    print(f"all rounds in kept matches with a live start: {agr['all_rounds']}")
    print(f"kept rounds (used by the model): {agr['kept_rounds']}")
    print(f"matches with any disagreement: all rounds {agr['matches_any_disagree_all']}; "
          f"kept rounds {agr['matches_any_disagree_kept']}")
    ph = phase_agreement(lead, live, H)
    for k, v in sorted(ph.items()):
        print(f"  phase {k:<8} rounds {v.get('rounds', 0):>6} agree {v.get('agree', 0):>6} "
              f"sign_flip {v.get('sign_flip', 0):>5}")

    # ---------- 4. features
    S_old, S_new = build_rows(kept, lead)
    assert S_old.height == kept["got"]["states"], "state count differs from the kept set"
    diff_mask = (S_old[OLD_COL] != S_new[OLD_COL]).to_numpy()
    n_states_diff = int(diff_mask.sum())
    m_diff_states = sorted(set(S_old.filter(pl.Series(diff_mask))["match_key"].to_list()))
    print(f"\n== 4. Features ==\nstates: {S_old.height}; states where the feature differs: "
          f"{n_states_diff}; matches where it differs: {len(m_diff_states)}")

    # ---------- 5. split
    sp = bk.make_splits(kept["kept_final"], H)[bk.PRIMARY]
    te_m = sp["test"]
    print(f"primary split: train {len(sp['train'])}, val {len(sp['val'])}, test {len(te_m)} matches; "
          f"test matches where the feature differs: {len(te_m & set(m_diff_states))}")

    # ---------- 6. XGBoost, old and corrected feature
    results = {}
    preds = {}
    yte = None
    for label, S in (("old_round_state", S_old), ("derived_lead", S_new)):
        tr = S.filter(pl.col("match_key").is_in(sorted(sp["train"])))
        va = S.filter(pl.col("match_key").is_in(sorted(sp["val"])))
        te = S.filter(pl.col("match_key").is_in(sorted(te_m)))
        yte = te["label_ct_win"].to_numpy().astype(np.int64)
        for device in ("cuda", "cpu"):
            t1 = time.time()
            p, bx, dev = fit_device(tr, va, te, device)
            met = bk.metrics(yte, p)
            preds[(label, device)] = (te, p)
            results[(label, device)] = {"device": dev, "best_iteration": bx,
                                        "fit_s": round(time.time() - t1, 1),
                                        "test": {k: met[k] for k in ("n", "logloss", "brier", "auc", "ece")}}
            print(f"  {label:<16} requested {device:<4} ran on {dev:<36} best_iter {bx:>4} "
                  f"test LL {met['logloss']:.5f} Brier {met['brier']:.5f} AUC {met['auc']:.4f} "
                  f"({time.time() - t1:.1f}s)", flush=True)

    print("\n== 6. Old minus derived (test) ==")
    deltas = {}
    for device in ("cuda", "cpu"):
        o = results[("old_round_state", device)]["test"]
        n = results[("derived_lead", device)]["test"]
        dll, dbr = n["logloss"] - o["logloss"], n["brier"] - o["brier"]
        te, p_old = preds[("old_round_state", device)]
        _, p_new = preds[("derived_lead", device)]
        pb = paired_delta(te, p_old, p_new, yte)
        deltas[device] = {"logloss_new_minus_old": dll, "brier_new_minus_old": dbr,
                          "paired_logloss_bootstrap": pb,
                          "max_abs_prob_diff": float(np.max(np.abs(p_new - p_old)))}
        print(f"  requested {device}: test LL new - old {dll:+.6f} (95% CI {pb['ci95'][0]:+.6f} to "
              f"{pb['ci95'][1]:+.6f}); Brier new - old {dbr:+.6f}; max |p_new - p_old| "
              f"{deltas[device]['max_abs_prob_diff']:.2e}")
    cpu_old = results[("old_round_state", "cpu")]["test"]
    print(f"cross-check: old feature on cpu test LL {cpu_old['logloss']:.5f}, Brier {cpu_old['brier']:.5f} "
          f"vs logs/bakeoff.log {BAKEOFF_XGB_TEST['logloss']:.5f} / {BAKEOFF_XGB_TEST['brier']:.5f}")

    out = {
        "kept_counts": kept["got"],
        "derived": {"rounds": len(lead), "header_check": dict(hc), "diagnostics": dict(st)},
        "agreement": agr,
        "agreement_by_phase": ph,
        "features": {"states": S_old.height, "states_differ": n_states_diff,
                     "matches_differ": len(m_diff_states),
                     "test_matches_differ": len(te_m & set(m_diff_states))},
        "xgboost": {f"{k[0]}|{k[1]}": v for k, v in results.items()},
        "difference": deltas,
        "bakeoff_log_reference": BAKEOFF_XGB_TEST,
        "run_seconds": round(time.time() - t0, 1),
    }
    OUT_JSON.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_JSON}\ndone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
