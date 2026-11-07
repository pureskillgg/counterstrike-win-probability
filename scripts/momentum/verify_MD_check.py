"""Independent verifier for task MD (momentum feature test) and the MA pistol odds ratio.

Recomputes without importing MD_features.py or MD_run.py:
  1. MA: round-1 pistol winner vs match winner over header matches (one row per match). Reports the
     odds that the pistol winner wins, and the 2x2 odds ratio over team rows (its square). Match-cluster
     bootstrap, 1000 resamples, seed 0.
  2. MD: team tracking, momentum features, leakage and rule checks, the round_state lead cross-check,
     then XGBoost with bake-off settings for baseline and three augmented variants on the bake-off
     primary split. Match-cluster bootstrap (1000 resamples, seed 0) for each log loss and paired difference.

Reuses (read only): scripts/eda/extract_state_1hz.py (bakeoff_kept; bk = scripts/model/bakeoff.py).
Writes logs/momentum/verify_MD.log only.

Run from the project root:
  G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\momentum\\verify_MD_check.py
"""

import collections
import importlib.util
import sys
import time
import warnings
from pathlib import Path

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import polars as pl
import xgboost as xgb

ROOT = Path(r"G:\pureskill\csds-cs2")
LOG_PATH = ROOT / "logs" / "momentum" / "verify_MD.log"
SEED = 0
N_BOOT = 1000
STREAK_CAP = 5
REPORT_BASELINE_LL = 0.4365987668
REPORT_BASELINE_BEST_ITER = 261
REPORT_TEST_STATES = 47716
REPORT_TEST_MATCHES = 304
PREV = ["prev_round_ct_team_won", "signed_streak_ct_team"]
PIST = ["pistol_r1_winner_is_ct_team", "pistol_r13_winner_is_ct_team"]

LINES = []


def log(s=""):
    s = str(s)
    print(s, flush=True)
    LINES.append(s)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pct_ci(a):
    lo, hi = np.percentile(a, [2.5, 97.5])
    return float(lo), float(hi)


def flip(t):
    return "B" if t == "A" else "A"


def feats_at(W, ct, r):
    """Momentum features for round r. W: round -> winning team ('A' or 'B'); ct: team on CT in round r.
    Reads winners only for rounds below r."""
    prev = streak = p1 = p13 = 0
    if r >= 2:
        last = W[r - 1]
        prev = 1 if last == ct else -1
        run, k = 0, r - 1
        while k >= 1 and run < STREAK_CAP and W[k] == last:
            run += 1
            k -= 1
        streak = run if last == ct else -run
        p1 = 1 if W[1] == ct else -1
    if r >= 14:
        p13 = 1 if W[13] == ct else -1
    return {"prev_round_ct_team_won": prev, "signed_streak_ct_team": streak,
            "pistol_r1_winner_is_ct_team": p1, "pistol_r13_winner_is_ct_team": p13}


def ll_vec(p, y):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def stage_ma(fv, T, CT):
    log("== MA: round-1 pistol winner vs match winner (header matches, all channels) ==")
    hdr = fv.read("header", ["key", "match_date", "max_rounds", "t_starters_score_final",
                             "ct_starters_score_final"])
    H = {r["key"]: r for r in hdr.iter_rows(named=True)}
    rend = fv.read("round_end", ["match_key", "round", "winner_team_code"])
    code_counts = collections.Counter(int(w) for w in rend["winner_team_code"].to_list() if w is not None)
    log(f"header matches {len(H)}; round_end rows {rend.height}; winner codes {dict(code_counts)}")
    rc = {}
    for m, n, mx in rend.group_by("match_key").agg(pl.col("round").n_unique().alias("n"),
                                                   pl.col("round").max().alias("mx")).iter_rows():
        rc[m] = (int(n), int(mx))
    r1_rows = collections.defaultdict(list)
    for m, rnd, w in rend.filter(pl.col("round") == 1).select("match_key", "round",
                                                              "winner_team_code").iter_rows():
        r1_rows[m].append(int(w))
    cnt = collections.Counter()
    x = []
    for m, h in H.items():
        cnt["header_matches"] += 1
        ht, hc = int(h["t_starters_score_final"]), int(h["ct_starters_score_final"])
        n, mx = rc.get(m, (0, 0))
        if n != mx:
            cnt["round_numbers_not_contiguous"] += 1
        if m not in r1_rows:
            cnt["no_round1_row"] += 1
            continue
        if len(r1_rows[m]) != 1:
            cnt["multiple_round1_rows"] += 1
            continue
        if n != ht + hc:
            cnt["round_count_not_header_sum"] += 1
            continue
        if ht == hc:
            cnt["tied_header_totals"] += 1
            continue
        w1 = r1_rows[m][0]
        if w1 not in (T, CT):
            cnt["bad_round1_code"] += 1
            continue
        # team A starts on T in round 1; header t_starters_score_final is team A's final score
        pist = "A" if w1 == T else "B"
        win = "A" if ht > hc else "B"
        x.append(1 if pist == win else 0)
        cnt["decided_matches"] += 1
    log(f"MA counts: {dict(cnt)}")
    x = np.array(x, dtype=np.int64)
    n_dec = len(x)
    a = int(x.sum())
    b = n_dec - a
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n_dec, size=(N_BOOT, n_dec))
    ra = x[idx].sum(axis=1)
    rb = n_dec - ra
    rep_odds = ra / rb
    rep_or2 = rep_odds ** 2
    rep_rate = ra / n_dec
    out = {
        "decided_matches": n_dec,
        "table_rows_team_level": {"pistol_winner_won_match": a, "pistol_winner_lost_match": b,
                                  "pistol_loser_won_match": b, "pistol_loser_lost_match": a},
        "win_rate": a / n_dec, "win_rate_ci95": pct_ci(rep_rate),
        "odds_pistol_winner_wins": a / b, "odds_ci95": pct_ci(rep_odds),
        "or_2x2_team_rows": (a / b) ** 2, "or_2x2_ci95": pct_ci(rep_or2),
    }
    log(f"decided matches (header totals differ, rounds = header sum): {n_dec}")
    log(f"2x2 team rows: pistol winner won {a}, lost {b}; pistol loser won {b}, lost {a}")
    log(f"win rate of round-1 pistol winner: {out['win_rate']:.4f} "
        f"(95% cluster bootstrap {out['win_rate_ci95'][0]:.4f} to {out['win_rate_ci95'][1]:.4f})")
    log(f"odds that the round-1 pistol winner wins the match: {out['odds_pistol_winner_wins']:.4f} "
        f"(95% CI {out['odds_ci95'][0]:.4f} to {out['odds_ci95'][1]:.4f})")
    log(f"odds ratio, 2x2 over team rows (pistol won vs lost, outcome = match won): "
        f"{out['or_2x2_team_rows']:.4f} (95% CI {out['or_2x2_ci95'][0]:.4f} to {out['or_2x2_ci95'][1]:.4f})")
    return out


def main():
    t0 = time.time()
    log("== verify_MD: independent recomputation (MD momentum feature test) ==")
    ex = load_module("extract_state_1hz", ROOT / "scripts" / "eda" / "extract_state_1hz.py")
    bk = ex.bk
    fv = bk.fv
    T, CT = bk.T, bk.CT
    log(f"bake-off FEATS ({len(bk.FEATS)}): {bk.FEATS}")
    log(f"bake-off MONO: {bk.MONO}")
    log(f"swap rule check: swap_expected(12, 24)={bk.swap_expected(12, 24)}, "
        f"swap_expected(27, 24)={bk.swap_expected(27, 24)}, swap_expected(30, 24)={bk.swap_expected(30, 24)}")

    ma = stage_ma(fv, T, CT)

    log("")
    log("== MD: kept set (bake-off port, asserted against bakeoff.log counts) ==")
    kept = ex.bakeoff_kept()
    H, rd, live = kept["H"], kept["rd"], kept["live"]
    deaths, plants, players = kept["deaths"], kept["plants"], kept["players"]

    # ---- team tracking, written here (team A starts on T; team B on CT; swap after bk.swap_expected rounds)
    by_match = collections.defaultdict(dict)
    for (m, r), v in rd.items():
        by_match[m][r] = v["winner"]
    track = {}
    tc = collections.Counter()
    for m, rounds in by_match.items():
        mr = H[m]["max_rounds"]
        tc[f"max_rounds_{mr}_matches"] += 1
        rs = sorted(rounds)
        tc["matches"] += 1
        if rs != list(range(1, len(rs) + 1)):
            tc["matches_with_round_gap"] += 1
        side_A = T
        for r in rs:
            w = rounds[r]
            if w not in (T, CT):
                tc["winner_code_not_T_or_CT"] += 1
            ct_team = "A" if side_A == CT else "B"
            win_team = "A" if w == side_A else "B"
            track[(m, r)] = (ct_team, win_team)
            tc["rounds"] += 1
            if bk.swap_expected(r, mr):
                side_A = CT if side_A == T else T
                tc["swaps_applied"] += 1
    log(f"team tracking counts: {dict(tc)}")

    # ---- lead cross-check: derived (CT team minus T team) vs round_state ct_score - t_score at freeze end
    lead_n = lead_bad = lead_no_live = 0
    fin_ok = fin_bad = fin_inc = 0
    for m, rounds in by_match.items():
        a = b = 0
        for r in sorted(rounds):
            ct_team, win_team = track[(m, r)]
            k = (m, r)
            if k in live and live[k]["t"] is not None and live[k]["ct"] is not None:
                derived = (a - b) if ct_team == "A" else (b - a)
                observed = live[k]["ct"] - live[k]["t"]
                lead_n += 1
                lead_bad += int(derived != observed)
            else:
                lead_no_live += 1
            if win_team == "A":
                a += 1
            else:
                b += 1
        ht, hc = H[m]["t_starters_score_final"], H[m]["ct_starters_score_final"]
        n = len(rounds)
        if n != ht + hc:
            fin_inc += 1
        elif (a, b) == (ht, hc):
            fin_ok += 1
        else:
            fin_bad += 1
    log(f"lead cross-check: rounds with live start {lead_n}, mismatches {lead_bad}, rounds without live start "
        f"{lead_no_live}")
    log(f"final-score cross-check vs header (team A = T starter): complete matches agree {fin_ok}, "
        f"disagree {fin_bad}, incomplete (round count != header sum) {fin_inc}")

    # ---- features, leakage and rule checks
    feat = {}
    leak_fail = power_n = power_fail = 0
    rule = collections.Counter()
    for m, rounds in by_match.items():
        rs = sorted(rounds)
        W = {r: track[(m, r)][1] for r in rs}
        for r in rs:
            ct_team = track[(m, r)][0]
            f = feats_at(W, ct_team, r)
            feat[(m, r)] = f
            W2 = {k: (flip(v) if k >= r else v) for k, v in W.items()}
            if feats_at(W2, ct_team, r) != f:
                leak_fail += 1
            if r >= 2:
                power_n += 1
                W3 = dict(W)
                W3[r - 1] = flip(W[r - 1])
                if feats_at(W3, ct_team, r)["prev_round_ct_team_won"] == f["prev_round_ct_team_won"]:
                    power_fail += 1
            if r == 1 and any(f.values()):
                rule["round1_nonzero"] += 1
            if r <= 13 and f["pistol_r13_winner_is_ct_team"] != 0:
                rule["pistol_r13_nonzero_before_14"] += 1
            if r >= 14 and f["pistol_r13_winner_is_ct_team"] not in (1, -1):
                rule["pistol_r13_not_pm1_from_14"] += 1
            if r >= 2 and f["pistol_r1_winner_is_ct_team"] not in (1, -1):
                rule["pistol_r1_not_pm1_from_2"] += 1
            if abs(f["signed_streak_ct_team"]) > STREAK_CAP:
                rule["streak_above_cap"] += 1
            rule["rows"] += 1
    log(f"leakage test (flip round r and later; features at r must not move): rounds {len(feat)}, "
        f"changed {leak_fail}")
    log(f"power test (flip round r-1 must move prev_round at r): checked {power_n}, not moved {power_fail}")
    log(f"rule checks: {dict(rule)}")
    assert leak_fail == 0 and power_fail == 0

    # ---- state table with momentum columns (bake-off build_states, unchanged)
    state_rows = []
    for k in kept["rounds_ok"]:
        m = k[0]
        rr = {**rd[k], "freeze": live[k]["tick"], "t_score": live[k]["t"], "ct_score": live[k]["ct"]}
        out = bk.build_states(k, rr, deaths.get(k, []), plants.get(k, []), players.get(k, {}),
                              H[m]["max_rounds"], collections.Counter())
        if out is None:
            raise RuntimeError("kept round returned None")
        f = feat[k]
        for s in out:
            s.update(f)
        state_rows.extend(out)
    S = pl.DataFrame(state_rows)
    assert S.height == kept["got"]["states"] == 234265, S.height
    log(f"state table: {S.height} rows, kept rounds {len(kept['rounds_ok'])}")

    sp = bk.make_splits(kept["kept_final"], H)[bk.PRIMARY]
    tr = S.filter(pl.col("match_key").is_in(sorted(sp["train"])))
    va = S.filter(pl.col("match_key").is_in(sorted(sp["val"])))
    te = S.filter(pl.col("match_key").is_in(sorted(sp["test"])))
    assert tr.height + va.height + te.height == S.height
    M = len(sp["test"])
    uniq = sorted(te["match_key"].unique().to_list())
    assert len(uniq) == M
    pos = {m: i for i, m in enumerate(uniq)}
    g = np.array([pos[m] for m in te["match_key"].to_list()], dtype=np.int64)
    cnt_m = np.bincount(g, minlength=M).astype(np.float64)
    yte = te["label_ct_win"].to_numpy().astype(np.int64)
    log(f"primary split: train {len(sp['train'])} matches ({tr.height} states), val {len(sp['val'])} "
        f"({va.height}), test {M} matches ({te.height} states)")
    log(f"reported test matches {REPORT_TEST_MATCHES}, reported test states {REPORT_TEST_STATES}: "
        f"{M == REPORT_TEST_MATCHES}, {te.height == REPORT_TEST_STATES}")

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, M, size=(N_BOOT, M))

    def boot_mean(vec):
        s = np.bincount(g, weights=vec, minlength=M)
        return s[idx].sum(axis=1) / cnt_m[idx].sum(axis=1)

    def fit(feats):
        mono = list(bk.MONO) + [0] * (len(feats) - len(bk.FEATS))
        Xtr = tr.select(feats).to_numpy().astype(np.float64)
        ytr = tr["label_ct_win"].to_numpy().astype(np.int64)
        Xva = va.select(feats).to_numpy().astype(np.float64)
        yva = va["label_ct_win"].to_numpy().astype(np.int64)
        Xte = te.select(feats).to_numpy().astype(np.float64)
        mdl = xgb.XGBClassifier(n_estimators=3000, learning_rate=0.05, max_depth=4, min_child_weight=20,
                                subsample=0.8, tree_method="hist", device="cpu",
                                monotone_constraints=tuple(mono), early_stopping_rounds=100,
                                eval_metric="logloss", random_state=SEED)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mdl.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        bx = int(mdl.best_iteration)
        p = mdl.predict_proba(Xte, iteration_range=(0, bx + 1))[:, 1]
        return p, bx

    variants = [("baseline", bk.FEATS),
                ("plus_prev_streak", bk.FEATS + PREV),
                ("plus_pistol", bk.FEATS + PIST),
                ("plus_all", bk.FEATS + PREV + PIST)]
    vec, res = {}, {}
    log("")
    log("== XGBoost fits (bake-off settings, cpu, seed 0, early stopping on validation) ==")
    for name, feats in variants:
        t1 = time.time()
        p, bx = fit(feats)
        v = ll_vec(p, yte.astype(np.float64))
        vec[name] = v
        ll = float(v.mean())
        brier = float(np.mean((p - yte) ** 2))
        bk_ll = bk.metrics(yte, p)["logloss"]
        reps = boot_mean(v)
        res[name] = {"n_features": len(feats), "best_iteration": bx, "logloss": ll,
                     "logloss_bakeoff_metric": bk_ll, "brier": brier, "logloss_ci95": pct_ci(reps)}
        log(f"  {name:<18} features {len(feats):>2} best_iter {bx:>4} test LL {ll:.10f} "
            f"(bk.metrics {bk_ll:.10f}) Brier {brier:.6f} LL 95% CI "
            f"[{res[name]['logloss_ci95'][0]:.6f}, {res[name]['logloss_ci95'][1]:.6f}] "
            f"({time.time() - t1:.1f}s)")

    b = res["baseline"]
    log("")
    log(f"baseline vs report: LL {b['logloss']:.10f} vs {REPORT_BASELINE_LL:.10f} "
        f"(abs diff {abs(b['logloss'] - REPORT_BASELINE_LL):.2e}); best_iteration {b['best_iteration']} vs "
        f"{REPORT_BASELINE_BEST_ITER}")

    pairs = [("baseline", "plus_prev_streak"), ("baseline", "plus_pistol"), ("baseline", "plus_all"),
             ("plus_pistol", "plus_all"), ("plus_prev_streak", "plus_all")]
    paired = {}
    log("")
    log("== paired differences, augmented minus reference (test log loss; negative favours augmented) ==")
    for ref, aug in pairs:
        d = vec[aug] - vec[ref]
        reps = boot_mean(d)
        paired[f"{aug} minus {ref}"] = {"delta": float(d.mean()), "ci95": pct_ci(reps),
                                        "share_resamples_below_0": float((reps < 0).mean())}
        lo, hi = pct_ci(reps)
        log(f"  {aug} minus {ref}: {float(d.mean()):+.7f} 95% CI [{lo:+.7f}, {hi:+.7f}] "
            f"share of resamples below 0: {float((reps < 0).mean()):.3f}")

    log("")
    log(f"done in {time.time() - t0:.1f}s")
    return ma, res, paired


if __name__ == "__main__":
    try:
        main()
    finally:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.write_text("\n".join(LINES) + "\n", encoding="utf-8")
        print(f"wrote {LOG_PATH}")
