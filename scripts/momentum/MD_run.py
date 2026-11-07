"""Task MD: momentum feature test on the bake-off event-state table (primary match split).

Variants (XGBoost, bake-off hyperparameters, device cpu, seed 0, early stopping on the validation split):
  baseline           bake-off FEATS (10 features)
  plus_prev_streak   + prev_round_ct_team_won, signed_streak_ct_team
  plus_pistol        + pistol_r1_winner_is_ct_team, pistol_r13_winner_is_ct_team
  plus_all           + all four

Uncertainty: match-cluster bootstrap, 1000 resamples of the 304 primary test matches, seed 0. The
same resample indices are used for every variant, so paired differences are paired.

Reuses (read only): scripts/eda/extract_state_1hz.py (bakeoff_kept, bk), scripts/model/sufficiency/
score_lead_fix.py (derive_leads), scripts/momentum/MD_features.py. Writes data/viz/MD.json and
data/viz/MD.svg only.

Run from the project root:
  G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\momentum\\MD_run.py
"""

import collections
import importlib.util
import json
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
MD_DIR = ROOT / "scripts" / "momentum"
OUT_JSON = ROOT / "data" / "viz" / "MD.json"
OUT_SVG = ROOT / "data" / "viz" / "MD.svg"
SEED = 0
N_BOOT = 1000
# score_lead_fix.json, old feature, cpu run: exact reproduction target for the baseline
REF_BASELINE_LL = 0.4365987668076211
REF_BASELINE_BRIER = 0.1463668231836436
REF_BASELINE_BEST_ITER = 261

PREV = ["prev_round_ct_team_won", "signed_streak_ct_team"]
PIST = ["pistol_r1_winner_is_ct_team", "pistol_r13_winner_is_ct_team"]
VARIANTS = [
    ("baseline", "Baseline (10 bake-off features)", []),
    ("plus_prev_streak", "+ prev round and streak", PREV),
    ("plus_pistol", "+ pistol r1 and r13", PIST),
    ("plus_all", "+ all four momentum features", PREV + PIST),
]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ex = load_module("extract_state_1hz", ROOT / "scripts" / "eda" / "extract_state_1hz.py")
bk = ex.bk
sl = load_module("score_lead_fix", ROOT / "scripts" / "model" / "sufficiency" / "score_lead_fix.py")
mdf = load_module("MD_features", MD_DIR / "MD_features.py")


def ll_vec(p, y):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot_mean(vals, g, M, idx):
    """Match-cluster bootstrap of the state-weighted mean of vals. Returns (bootstrap reps, point)."""
    s = np.bincount(g, weights=vals, minlength=M)
    c = np.bincount(g, minlength=M).astype(np.float64)
    return s[idx].sum(axis=1) / c[idx].sum(axis=1), float(vals.mean())


def ci95(a):
    lo, hi = np.percentile(a, [2.5, 97.5])
    return [float(lo), float(hi)]


def fit_xgb(tr, va, te, feats):
    mono = list(bk.MONO) + [0] * (len(feats) - len(bk.FEATS))
    Xtr = tr.select(feats).to_numpy().astype(np.float64)
    ytr = tr["label_ct_win"].to_numpy().astype(np.int64)
    Xva = va.select(feats).to_numpy().astype(np.float64)
    yva = va["label_ct_win"].to_numpy().astype(np.int64)
    Xte = te.select(feats).to_numpy().astype(np.float64)
    m = xgb.XGBClassifier(n_estimators=3000, learning_rate=0.05, max_depth=4, min_child_weight=20,
                          subsample=0.8, tree_method="hist", device="cpu",
                          monotone_constraints=tuple(mono), early_stopping_rounds=100,
                          eval_metric="logloss", random_state=SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    bx = int(m.best_iteration)
    p = m.predict_proba(Xte, iteration_range=(0, bx + 1))[:, 1]
    return p, bx


def svg_chart(rows, paired, n_test_m, n_test_s, ymin, ymax):
    """rows: [(label, ll, lo, hi, color)]; paired: [(label, delta, lo, hi)]."""
    W, Hh = 800, 450
    L, R, TOP, BOT = 92, 772, 96, 344
    FONT = "Helvetica, Arial, sans-serif"

    def y_of(v):
        return BOT - (v - ymin) / (ymax - ymin) * (BOT - TOP)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {Hh}" width="{W}" height="{Hh}" '
           f'font-family="{FONT}">',
           f'<rect x="0" y="0" width="{W}" height="{Hh}" fill="#ffffff"/>',
           f'<text x="32" y="36" font-size="18" font-weight="600" fill="#0f172a">'
           f'Momentum features: test log loss (lower is better)</text>',
           f'<text x="32" y="58" font-size="12" fill="#475569">XGBoost, bake-off primary split, '
           f'{n_test_m} test matches, {n_test_s:,} test states. Whiskers: 95% match-cluster bootstrap '
           f'(1000 resamples, seed 0).</text>',
           f'<text x="32" y="76" font-size="11" fill="#475569">Y-axis starts at {ymin:.3f}, not zero. '
           f'Differences between bars are small; see paired differences below.</text>']
    step = 0.005
    t = np.ceil(ymin / step) * step
    while t <= ymax + 1e-12:
        y = y_of(t)
        out.append(f'<line x1="{L}" y1="{y:.1f}" x2="{R}" y2="{y:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        out.append(f'<text x="{L - 8}" y="{y + 4:.1f}" font-size="11" text-anchor="end" '
                   f'fill="#475569">{t:.3f}</text>')
        t += step
    out.append(f'<line x1="{L}" y1="{TOP}" x2="{L}" y2="{BOT}" stroke="#94a3b8" stroke-width="1"/>')
    out.append(f'<line x1="{L}" y1="{BOT}" x2="{R}" y2="{BOT}" stroke="#94a3b8" stroke-width="1"/>')
    out.append(f'<text x="22" y="{(TOP + BOT) / 2:.0f}" font-size="12" fill="#475569" '
               f'transform="rotate(-90 22 {(TOP + BOT) / 2:.0f})" text-anchor="middle">Test log loss (nats)</text>')
    n = len(rows)
    slot = (R - L) / n
    bw = 118
    for i, (lab, ll, lo, hi, col) in enumerate(rows):
        cx = L + slot * (i + 0.5)
        yt, yb = y_of(ll), BOT
        out.append(f'<rect x="{cx - bw / 2:.1f}" y="{yt:.1f}" width="{bw}" height="{yb - yt:.1f}" fill="{col}"/>')
        out.append(f'<line x1="{cx:.1f}" y1="{y_of(lo):.1f}" x2="{cx:.1f}" y2="{y_of(hi):.1f}" '
                   f'stroke="#0f172a" stroke-width="1.5"/>')
        for v in (lo, hi):
            out.append(f'<line x1="{cx - 10:.1f}" y1="{y_of(v):.1f}" x2="{cx + 10:.1f}" y2="{y_of(v):.1f}" '
                       f'stroke="#0f172a" stroke-width="1.5"/>')
        out.append(f'<text x="{cx:.1f}" y="{y_of(hi) - 8:.1f}" font-size="13" font-weight="600" '
                   f'text-anchor="middle" fill="#0f172a">{ll:.5f}</text>')
        words = lab.split(" ")
        half = max(1, len(words) // 2)
        l1, l2 = " ".join(words[:half]), " ".join(words[half:])
        out.append(f'<text x="{cx:.1f}" y="{BOT + 20}" font-size="12" text-anchor="middle" '
                   f'fill="#0f172a">{l1}</text>')
        if l2:
            out.append(f'<text x="{cx:.1f}" y="{BOT + 36}" font-size="12" text-anchor="middle" '
                       f'fill="#0f172a">{l2}</text>')
    y0 = 396
    out.append(f'<text x="32" y="{y0}" font-size="12" font-weight="600" fill="#0f172a">'
               f'Paired difference in log loss, augmented minus baseline (match-cluster 95% interval)</text>')
    for j, (lab, d, lo, hi) in enumerate(paired):
        out.append(f'<text x="32" y="{y0 + 17 + 16 * j}" font-size="12" fill="#0f172a">{lab}: '
                   f'{d:+.5f} [{lo:+.5f}, {hi:+.5f}]</text>')
    return "\n".join(out) + "\n</svg>\n"


def main():
    t0 = time.time()
    print("== MD momentum feature test ==")
    kept = ex.bakeoff_kept()  # prints and asserts the bake-off kept counts
    H, rd, live, deaths, plants, players = (kept[x] for x in ("H", "rd", "live", "deaths", "plants", "players"))

    lead, final, st = sl.derive_leads(rd, H)
    tracks, track_stats = mdf.team_tracks(rd, H, bk)
    xc = mdf.lead_crosscheck(tracks, lead)
    print(f"team tracking: {dict(track_stats)}; lead cross-check vs derive_leads: {xc}")
    assert xc["mismatches_vs_derive_leads"] == 0, "tracked teams disagree with the derived lead"

    feats = mdf.momentum_all(tracks)
    leak = mdf.leak_checks(tracks, feats)
    rules = mdf.rule_checks(tracks, feats)
    print(f"leak checks: {leak}")
    print(f"rule checks: {rules}")
    assert leak.get("changed_when_round_r_and_later_flipped", 0) == 0, "features at round r depend on round r or later"
    assert rules.get("pistol_r13_nonzero_before_14", 0) == 0
    assert rules.get("round1_nonzero", 0) == 0

    # ---- state table: bake-off rows in the same order, with the momentum columns appended
    state_rows = []
    for k in kept["rounds_ok"]:
        m = k[0]
        rr = {**rd[k], "freeze": live[k]["tick"], "t_score": live[k]["t"], "ct_score": live[k]["ct"]}
        out = bk.build_states(k, rr, deaths.get(k, []), plants.get(k, []), players.get(k, {}),
                              H[m]["max_rounds"], collections.Counter())
        assert out is not None, "kept round with unresolved victim side"
        f = feats[k]
        for s in out:
            s.update(f)
        state_rows.extend(out)
    S = pl.DataFrame(state_rows)
    assert S.height == kept["got"]["states"] == 234265, S.height
    assert sum(S[c].null_count() for c in mdf.FEATURE_NAMES) == 0
    print(f"state table: {S.height} rows; kept rounds {len(kept['rounds_ok'])}")

    feature_counts = {}
    for c in mdf.FEATURE_NAMES:
        cnt = collections.Counter(int(v) for v in S[c].to_list())
        feature_counts[c] = {str(k): cnt[k] for k in sorted(cnt)}

    # ---- primary split (bake-off)
    sp = bk.make_splits(kept["kept_final"], H)[bk.PRIMARY]
    tr = S.filter(pl.col("match_key").is_in(sorted(sp["train"])))
    va = S.filter(pl.col("match_key").is_in(sorted(sp["val"])))
    te = S.filter(pl.col("match_key").is_in(sorted(sp["test"])))
    yte = te["label_ct_win"].to_numpy().astype(np.int64)
    uniq = sorted(te["match_key"].unique().to_list())
    pos = {m: i for i, m in enumerate(uniq)}
    g = np.array([pos[m] for m in te["match_key"].to_list()], dtype=np.int64)
    M = len(uniq)
    assert M == len(sp["test"]), (M, len(sp["test"]))
    print(f"primary split: train {len(sp['train'])}, val {len(sp['val'])}, test {M} matches, "
          f"{te.height} test states")

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, M, size=(N_BOOT, M))

    preds, res, lls = {}, {}, {}
    for name, label, extra in VARIANTS:
        feats_v = bk.FEATS + extra
        t1 = time.time()
        p, bx = fit_xgb(tr, va, te, feats_v)
        preds[name] = p
        met = bk.metrics(yte, p)
        ll_rows = ll_vec(p, yte)
        lls[name] = ll_rows
        reps, point = boot_mean(ll_rows, g, M, idx)
        assert abs(point - met["logloss"]) < 1e-8, (point, met["logloss"])
        res[name] = {"label": label, "features": feats_v, "n_features": len(feats_v),
                     "best_iteration": bx, "fit_s": round(time.time() - t1, 1),
                     "test": {"states": met["n"], "matches": M, "logloss": met["logloss"],
                              "brier": met["brier"], "auc": met["auc"], "ece": met["ece"],
                              "logloss_ci95": ci95(reps)}}
        print(f"  {name:<18} features {len(feats_v):>2} best_iter {bx:>4} test LL {met['logloss']:.7f} "
              f"Brier {met['brier']:.6f} AUC {met['auc']:.5f} ECE {met['ece']:.5f} "
              f"LL CI {ci95(reps)[0]:.5f}..{ci95(reps)[1]:.5f} ({time.time() - t1:.1f}s)", flush=True)

    base_ll = res["baseline"]["test"]["logloss"]
    reproduced = abs(base_ll - REF_BASELINE_LL) < 1e-9 and res["baseline"]["best_iteration"] == REF_BASELINE_BEST_ITER
    print(f"baseline reproduces score_lead_fix.json (LL {REF_BASELINE_LL}, best_iter {REF_BASELINE_BEST_ITER}): "
          f"{reproduced}")

    def paired(a, b):
        d = lls[b] - lls[a]
        reps, point = boot_mean(d, g, M, idx)
        return {"delta_logloss": point, "ci95": ci95(reps),
                "share_resamples_below_0": float((reps < 0).mean())}

    pair = {}
    for name, _, _ in VARIANTS[1:]:
        pair[name] = paired("baseline", name)
    pair_extra = {"plus_all_minus_plus_pistol": paired("plus_pistol", "plus_all"),
                  "plus_all_minus_plus_prev_streak": paired("plus_prev_streak", "plus_all")}
    for k, v in {**pair, **pair_extra}.items():
        print(f"  paired {k:<34} {v['delta_logloss']:+.6f} CI [{v['ci95'][0]:+.6f}, {v['ci95'][1]:+.6f}] "
              f"share resamples < 0: {v['share_resamples_below_0']:.3f}")

    # ---- winner counts (header totals), reported for the record; not used by the feature test
    wc = collections.Counter()
    for m, (a, b, n) in final.items():
        ht, hc = H[m]["t_starters_score_final"], H[m]["ct_starters_score_final"]
        wc["channel_kept_matches"] += 1
        if n != ht + hc:
            wc["round_count_not_equal_header_sum"] += 1
            continue
        wc["round_count_equals_header_sum"] += 1
        if ht == hc:
            wc["tied_header_totals"] += 1
            continue
        wc["winner_defined_totals_differ_and_sum_matches"] += 1
    wc["header_matches_total"] = len(H)
    max_rounds_counts = {k: v for k, v in track_stats.items() if k.startswith("max_rounds_")}

    rows = []
    colors = {"baseline": "#475569"}
    for name, label, _ in VARIANTS:
        colors.setdefault(name, "#2563eb")
        r_ = res[name]["test"]
        rows.append((label, r_["logloss"], r_["logloss_ci95"][0], r_["logloss_ci95"][1], colors[name]))
    lo_all = min(r[2] for r in rows)
    hi_all = max(r[3] for r in rows)
    ymin = float(np.floor((lo_all - 0.002) * 1000) / 1000)
    ymax = float(np.ceil((hi_all + 0.002) * 1000) / 1000)
    paired_lines = [
        ("vs baseline: + prev round and streak", pair["plus_prev_streak"]["delta_logloss"],
         pair["plus_prev_streak"]["ci95"][0], pair["plus_prev_streak"]["ci95"][1]),
        ("vs baseline: + pistol r1 and r13 only", pair["plus_pistol"]["delta_logloss"],
         pair["plus_pistol"]["ci95"][0], pair["plus_pistol"]["ci95"][1]),
        ("vs baseline: + all four", pair["plus_all"]["delta_logloss"],
         pair["plus_all"]["ci95"][0], pair["plus_all"]["ci95"][1]),
    ]
    svg = svg_chart(rows, paired_lines, M, te.height, ymin, ymax)
    OUT_SVG.write_text(svg, encoding="utf-8")

    out = {
        "id": "MD",
        "task": "Momentum features added to the bake-off XGBoost round model (primary split)",
        "settings": {"seed": SEED, "n_boot": N_BOOT, "device": "cpu",
                     "xgboost": "bake-off settings (3000 max trees, eta 0.05, depth 4, min_child_weight 20, "
                                "subsample 0.8, early stop 100 on validation, monotone bakeoff.MONO; new "
                                "features unconstrained)",
                     "bootstrap": "match-cluster, resample the 304 test matches with replacement; "
                                  "state-weighted mean"},
        "kept_counts": kept["got"],
        "primary_split": {"train_matches": len(sp["train"]), "val_matches": len(sp["val"]),
                          "test_matches": M, "test_states": te.height},
        "team_tracking": {"track_stats": dict(track_stats), "lead_crosscheck": xc},
        "leak_checks": leak,
        "rule_checks": rules,
        "max_rounds_counts": max_rounds_counts,
        "feature_value_counts_kept_states": feature_counts,
        "variants": res,
        "paired_vs_baseline": pair,
        "paired_other": pair_extra,
        "reference_reproduction": {"reference_logloss": REF_BASELINE_LL,
                                   "reference_best_iteration": REF_BASELINE_BEST_ITER,
                                   "reproduced": reproduced},
        "winner_counts_header": dict(wc),
        "run_seconds": round(time.time() - t0, 1),
    }
    OUT_JSON.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(f"wrote {OUT_JSON} and {OUT_SVG}; {out['run_seconds']}s")


if __name__ == "__main__":
    main()
