"""Check B: batch holdout for the round win-probability XGBoost model (seed 0).

Batch = the <DD> folder of the match key (csds/2026/10/<DD>/<id>/csds). The header match_date is
reported as a cross-check only.

Forward (primary)
  Train pool = kept matches in folder days 01-07.
  In-batch test = random 20% of that pool (match level), held out of training.
  Validation = random 10% of the rest (match level, early stopping). Train = the remainder.
  Out-of-batch test = every kept match in day 08.
Reverse (second pass)
  Same construction with the batches swapped: pool = day 08, out-of-batch test = days 01-07.

Model: XGBoost hist, depth 4, eta 0.05, min_child_weight 20, subsample 0.8, early stopping 100 on
the validation slice, monotone constraints bakeoff.MONO, features bakeoff.FEATS, seed 0.
device cuda is tried first; cpu is the fallback. Label ct_win = winner_team_code == 3.
Metrics: bakeoff.metrics (log loss, Brier, AUC, ECE with 10 bins).
Uncertainty: match-level bootstrap (1000 resamples, seed 0) of each test set and of the gap.
Verdict rule (fixed before the run): LARGE if the 95% bootstrap CI of the log loss gap excludes 0
and |gap| >= 0.01 nats.

Reuses scripts/eda/extract_state_1hz.py bakeoff_kept(), which imports scripts/model/bakeoff.py
read-only and asserts the bake-off kept counts. Writes only under notes/model/sufficiency.

Run from the project root:
  pixi run python -I -X utf8 scripts/model/sufficiency/batch_holdout.py
"""

import collections
import importlib.util
import json
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"G:\pureskill\csds-cs2")
OUT_DIR = ROOT / "notes" / "model" / "sufficiency"
OUT_JSON = OUT_DIR / "batch_holdout.json"
OUT_MD = OUT_DIR / "batch_holdout.md"

SEED = 0
IN_BATCH_FRAC = 0.20
VAL_FRAC = 0.10
N_BOOT = 1000
FWD_DAYS = {"01", "02", "03", "04", "05", "06", "07"}
NEW_DAY = "08"
LARGE_GAP = 0.01
EPS = 1e-15
EXPECT = {"channel_kept": 1709, "dup_matches_removed": 47, "rounds_in_kept_matches": 35502,
          "kept_rounds": 28683, "kept_matches": 1521, "states": 234265}
SET_LABEL = {"train": "train (pool minus in-batch minus validation)",
             "val": "validation (match-level 10%, early stopping)",
             "in_batch": "in-batch test (20%, held out)",
             "other": "out-of-batch test"}

spec = importlib.util.spec_from_file_location(
    "extract_state_1hz", ROOT / "scripts" / "eda" / "extract_state_1hz.py")
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)
bk = ex.bk
FEATS, MONO = bk.FEATS, bk.MONO
DEVICE = {"name": None, "errors": [], "note": ""}


def xy(df):
    X = df.select(FEATS).to_numpy().astype(np.float64)
    y = df["label_ct_win"].to_numpy().astype(np.int64)
    return X, y


def holdout(ms, frac):
    """Match-level split: sorted list shuffled with seed 0; the first frac goes to the held-out side."""
    ms = sorted(ms)
    random.Random(SEED).shuffle(ms)
    n = round(frac * len(ms))
    return set(ms[:n]), set(ms[n:])


def fit_xgb(Xtr, ytr, Xva, yva):
    params = dict(n_estimators=3000, learning_rate=0.05, max_depth=4, min_child_weight=20,
                  subsample=0.8, tree_method="hist", monotone_constraints=tuple(MONO),
                  early_stopping_rounds=100, eval_metric="logloss", random_state=SEED)
    order = [DEVICE["name"]] if DEVICE["name"] else ["cuda", "cpu"]
    for dev in order:
        try:
            m = xgb.XGBClassifier(device=dev, **params)
            with warnings.catch_warnings(record=True) as wlog:
                warnings.simplefilter("always")
                m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            fell = [str(w.message) for w in wlog if "GPU" in str(w.message) and "CPU" in str(w.message)]
            if dev == "cuda" and fell:
                # xgboost moved the fit to CPU (no usable GPU); the model is a CPU model
                DEVICE["name"] = "cpu"
                DEVICE["note"] = ("cuda requested; xgboost reported no usable GPU and ran on cpu: "
                                  + fell[0][:200])
            else:
                DEVICE["name"] = dev
            return m, int(m.best_iteration)
        except Exception as e:
            msg = f"{dev}: {type(e).__name__}: {str(e)[:200]}"
            DEVICE["errors"].append(msg)
            print("device failed -> " + msg)
    raise SystemExit("xgboost failed on cuda and cpu")


def summarize(df):
    return {"matches": df["match_key"].n_unique(),
            "rounds": df.select(["match_key", "round"]).unique().height,
            "states": df.height,
            "ct_win_share_states": float(df["label_ct_win"].mean())}


def day_mix(ms, day_of):
    return {k: v for k, v in sorted(collections.Counter(day_of[m] for m in ms).items())}


def match_index(df):
    mk = df["match_key"].to_list()
    uniq = sorted(set(mk))
    pos = {m: i for i, m in enumerate(uniq)}
    return np.array([pos[m] for m in mk], dtype=np.int64), len(uniq)


def ll_rows(y, p):
    pc = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(pc) + (1 - y) * np.log(1 - pc))


def boot_reps(y, p, mid, n_match, idx):
    ll = ll_rows(y, p)
    br = (p - y) ** 2
    reps = {"logloss": np.empty(N_BOOT), "brier": np.empty(N_BOOT), "auc": np.empty(N_BOOT)}
    for b in range(N_BOOT):
        w = np.bincount(idx[b], minlength=n_match)[mid].astype(np.float64)
        sw = w.sum()
        reps["logloss"][b] = (w * ll).sum() / sw
        reps["brier"][b] = (w * br).sum() / sw
        reps["auc"][b] = roc_auc_score(y, p, sample_weight=w)
    return reps


def pct(a):
    lo, hi = np.percentile(a, [2.5, 97.5])
    return [float(lo), float(hi)]


def strip(m):
    return {k: v for k, v in m.items() if k != "table"}


def run_direction(name, sets, day_of, S, rng, test_label):
    tr, va, te_in, te_out = (S.filter(pl.col("match_key").is_in(sorted(sets[k])))
                             for k in ("train", "val", "in_batch", "other"))
    Xtr, ytr = xy(tr)
    Xva, yva = xy(va)
    Xin, yin = xy(te_in)
    Xout, yout = xy(te_out)
    t_fit = time.time()
    model, best = fit_xgb(Xtr, ytr, Xva, yva)
    it = (0, best + 1)
    p_va = model.predict_proba(Xva, iteration_range=it)[:, 1]
    p_in = model.predict_proba(Xin, iteration_range=it)[:, 1]
    p_out = model.predict_proba(Xout, iteration_range=it)[:, 1]
    fit_s = time.time() - t_fit
    rate = float(ytr.mean())

    m_in = bk.metrics(yin, p_in)
    m_out = bk.metrics(yout, p_out)
    m_va = bk.metrics(yva, p_va)
    c_in = bk.metrics(yin, np.full(len(yin), rate))
    c_out = bk.metrics(yout, np.full(len(yout), rate))

    mid_in, n_in = match_index(te_in)
    mid_out, n_out = match_index(te_out)
    reps_in = boot_reps(yin, p_in, mid_in, n_in, rng.integers(0, n_in, size=(N_BOOT, n_in)))
    reps_out = boot_reps(yout, p_out, mid_out, n_out, rng.integers(0, n_out, size=(N_BOOT, n_out)))
    # bootstrap with unit weights must reproduce the point estimate
    assert abs(float(ll_rows(yout, p_out).mean()) - m_out["logloss"]) < 1e-6
    assert abs(float(ll_rows(yin, p_in).mean()) - m_in["logloss"]) < 1e-6

    gap = {k: m_out[k] - m_in[k] for k in ("logloss", "brier", "auc", "ece")}
    gap_ci = {k: pct(reps_out[k] - reps_in[k]) for k in ("logloss", "brier", "auc")}
    lo, hi = gap_ci["logloss"]
    large = (lo > 0 or hi < 0) and abs(gap["logloss"]) >= LARGE_GAP
    verdict = (f"{'LARGE' if large else 'NOT LARGE'} batch dependence ({name}): log loss gap "
               f"{gap['logloss']:+.5f}, 95% CI [{lo:+.5f}, {hi:+.5f}]; rule: LARGE if the CI excludes 0 "
               f"and |gap| >= {LARGE_GAP}.")

    sets_out = {}
    for k in ("train", "val", "in_batch", "other"):
        ms = sets[k]
        df = {"train": tr, "val": va, "in_batch": te_in, "other": te_out}[k]
        sets_out[k] = {**summarize(df), "days": day_mix(ms, day_of), "label": SET_LABEL[k]}

    res = {"name": name, "test_label": test_label, "device": DEVICE["name"],
           "device_note": DEVICE["note"], "best_iteration": best, "fit_s": round(fit_s, 1), "sets": sets_out,
           "validation_logloss_early_stop_set": m_va["logloss"],
           "metrics": {"in_batch": strip(m_in), "other": strip(m_out)},
           "constant_reference": {"train_rate": rate, "in_batch": strip(c_in), "other": strip(c_out)},
           "gap_other_minus_in_batch": gap,
           "bootstrap_95ci": {"in_batch": {k: pct(reps_in[k]) for k in reps_in},
                              "other": {k: pct(reps_out[k]) for k in reps_out},
                              "gap": gap_ci},
           "verdict": verdict}
    print(f"[{name}] device {DEVICE['name']}, best_iteration {best}, fit {fit_s:.1f}s")
    print(f"[{name}] in-batch LL {m_in['logloss']:.5f} Brier {m_in['brier']:.5f} AUC {m_in['auc']:.4f}")
    print(f"[{name}] other   LL {m_out['logloss']:.5f} Brier {m_out['brier']:.5f} AUC {m_out['auc']:.4f}")
    print(f"[{name}] gap LL {gap['logloss']:+.5f} CI {lo:+.5f}..{hi:+.5f}")
    print(f"[{name}] {verdict}")
    return res


def fmt_days(d):
    return ", ".join(f"{k}: {v}" for k, v in d.items())


def write_md(out, fwd, rev):
    g = out["bakeoff_kept_check"]["got"]
    L = ["# Check B: batch holdout", "",
         "Script: `scripts/model/sufficiency/batch_holdout.py`, seed 0. Batch = the `<DD>` folder of the "
         "match key. Header match_date is a cross-check only. Every number below was computed in this "
         "run (status VERIFIED).", "",
         "## Counts", "",
         "| Item | Value | Status |", "|---|---|---|",
         f"| Kept set reproduces bakeoff.log (channel kept, dup removed, kept rounds, kept matches, states) "
         f"| {g['channel_kept']}, {g['dup_matches_removed']}, {g['kept_rounds']}, {g['kept_matches']}, "
         f"{g['states']} | VERIFIED |",
         f"| Header matches by folder day | {fmt_days(out['header_matches_by_day'])} | VERIFIED |",
         f"| Kept matches by folder day | {fmt_days(out['kept_matches_by_day'])} | VERIFIED |",
         f"| Header match_date equals folder day | {out['header_date_agree']} of {out['header_rows']} "
         f"(all header rows); kept matches disagreeing: {out['kept_date_disagree']} | VERIFIED |",
         "", "## Splits", "",
         "| Run | Set | Batch days | Matches | Rounds | States | CT-win share (states) | Status |",
         "|---|---|---|---:|---:|---:|---:|---|"]
    for R in (fwd, rev):
        for k in ("train", "val", "in_batch", "other"):
            s = R["sets"][k]
            L.append(f"| {R['name']} | {s['label']} | {fmt_days(s['days'])} | {s['matches']} | {s['rounds']} "
                     f"| {s['states']} | {s['ct_win_share_states']:.4f} | VERIFIED |")
    L += ["", "## Test metrics", "",
          "| Run | Test set | States | Log loss | Brier | AUC | ECE (10 bins) | Status |",
          "|---|---|---:|---:|---:|---:|---:|---|"]
    for R in (fwd, rev):
        for k, lab in (("in_batch", "in-batch"), ("other", "out-of-batch")):
            m = R["metrics"][k]
            L.append(f"| {R['name']} | {lab}: {R['test_label'][k]} | {m['n']} | {m['logloss']:.5f} "
                     f"| {m['brier']:.5f} | {m['auc']:.4f} | {m['ece']:.4f} | VERIFIED |")
        for k, lab in (("in_batch", "constant, in-batch"), ("other", "constant, out-of-batch")):
            m = R["constant_reference"][k]
            L.append(f"| {R['name']} | {lab} (train rate {R['constant_reference']['train_rate']:.4f}) "
                     f"| {m['n']} | {m['logloss']:.5f} | {m['brier']:.5f} | {m['auc']:.4f} | {m['ece']:.4f} "
                     f"| VERIFIED |")
        gp = R["gap_other_minus_in_batch"]
        L.append(f"| {R['name']} | gap (out-of-batch minus in-batch) | - | {gp['logloss']:+.5f} "
                 f"| {gp['brier']:+.5f} | {gp['auc']:+.4f} | {gp['ece']:+.4f} | VERIFIED |")
    L += ["", "## Gap uncertainty (match bootstrap, 1000 resamples, seed 0)", "",
          "| Run | Log loss gap 95% CI | Brier gap 95% CI | AUC gap 95% CI | Status |",
          "|---|---|---|---|---|"]
    for R in (fwd, rev):
        c = R["bootstrap_95ci"]["gap"]
        L.append(f"| {R['name']} | [{c['logloss'][0]:+.5f}, {c['logloss'][1]:+.5f}] "
                 f"| [{c['brier'][0]:+.5f}, {c['brier'][1]:+.5f}] "
                 f"| [{c['auc'][0]:+.4f}, {c['auc'][1]:+.4f}] | VERIFIED |")
    L += ["", "## Verdict", ""]
    for R in (fwd, rev):
        L.append(f"- {R['verdict']}")
        L.append(f"- {R['name']}: device {R['device']}, best_iteration {R['best_iteration']}, "
                 f"early-stop validation LL {R['validation_logloss_early_stop_set']:.5f} (VERIFIED).")
    if DEVICE["note"]:
        L.append("- Device: " + DEVICE["note"])
    if DEVICE["errors"]:
        L.append("- Device errors: " + " | ".join(DEVICE["errors"]))
    L.append("")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    kept = ex.bakeoff_kept()
    got = kept["got"]
    assert got == EXPECT, f"kept set differs from bake-off: {got}"
    H, rd, live, plants, players, deaths = (kept[x] for x in
                                             ("H", "rd", "live", "plants", "players", "deaths"))
    rounds_ok = kept["rounds_ok"]
    state_rows = []
    for k in rounds_ok:
        rr = {**rd[k], "freeze": live[k]["tick"], "t_score": live[k]["t"], "ct_score": live[k]["ct"]}
        out = bk.build_states(k, rr, deaths.get(k, []), plants.get(k, []), players.get(k, {}),
                              H[k[0]]["max_rounds"], collections.Counter())
        assert out is not None
        state_rows.extend(out)
    S = pl.DataFrame(state_rows)
    assert S.height == EXPECT["states"], S.height

    kept_matches = sorted({k[0] for k in rounds_ok})
    day_of = {m: ex.match_day(m) for m in kept_matches}
    assert set(day_of.values()) <= (FWD_DAYS | {NEW_DAY}), "unexpected folder day"
    header_day = {m: ex.match_day(m) for m in H}
    header_matches_by_day = {k: v for k, v in sorted(collections.Counter(header_day.values()).items())}
    kept_matches_by_day = {k: v for k, v in sorted(collections.Counter(day_of.values()).items())}
    header_date_agree = sum(1 for m in H if str(H[m]["match_date"])[:10] == "2026-10-" + header_day[m])
    kept_date_disagree = sum(1 for m in kept_matches
                             if str(H[m]["match_date"])[:10] != "2026-10-" + day_of[m])
    print(f"header by day {header_matches_by_day}; kept by day {kept_matches_by_day}")
    print(f"header match_date == folder day: {header_date_agree} of {len(H)}; "
          f"kept disagreeing: {kept_date_disagree}")

    pool_fwd = {m for m in kept_matches if day_of[m] in FWD_DAYS}
    other_fwd = {m for m in kept_matches if day_of[m] == NEW_DAY}
    in_f, rest_f = holdout(pool_fwd, IN_BATCH_FRAC)
    val_f, train_f = holdout(rest_f, VAL_FRAC)
    fwd_sets = {"train": train_f, "val": val_f, "in_batch": in_f, "other": other_fwd}
    in_r, rest_r = holdout(other_fwd, IN_BATCH_FRAC)
    val_r, train_r = holdout(rest_r, VAL_FRAC)
    rev_sets = {"train": train_r, "val": val_r, "in_batch": in_r, "other": pool_fwd}
    for sets in (fwd_sets, rev_sets):
        parts = [sets[k] for k in ("train", "val", "in_batch", "other")]
        assert sum(len(p) for p in parts) == len(kept_matches) == len(set().union(*parts))

    rng = np.random.default_rng(SEED)
    fwd = run_direction("forward", fwd_sets, day_of, S, rng,
                        {"in_batch": "random 20% of days 01-07", "other": "all of day 08"})
    rev = run_direction("reverse", rev_sets, day_of, S, rng,
                        {"in_batch": "random 20% of day 08", "other": "all of days 01-07"})

    out = {"check": "B batch holdout", "status": "VERIFIED", "seed": SEED,
           "batch_definition": "folder day DD of key path csds/2026/10/DD/<id>/csds",
           "bakeoff_kept_check": {"got": got, "expected": EXPECT, "status": "VERIFIED"},
           "header_rows": len(H), "header_matches_by_day": header_matches_by_day,
           "kept_matches_by_day": kept_matches_by_day, "header_date_agree": header_date_agree,
           "kept_date_disagree": kept_date_disagree, "large_gap_threshold": LARGE_GAP,
           "forward": fwd, "reverse": rev, "wall_s": round(time.time() - t0, 1)}
    OUT_JSON.write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                        encoding="utf-8")
    write_md(out, fwd, rev)
    print(f"wrote {OUT_JSON} and {OUT_MD}; wall {out['wall_s']}s")


if __name__ == "__main__":
    main()
