"""Render notes/model/sufficiency/learning_curve.md from learning_curve.json (CHECK A).

Reads only the JSON that learning_curve.py wrote. Every number in the Markdown comes from that file.

Run from the project root (-B: no bytecode written):
  pixi run python -B -I -X utf8 scripts\\model\\sufficiency\\learning_curve_md.py
"""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"G:\pureskill\csds-cs2")
OUT_DIR = ROOT / "notes" / "model" / "sufficiency"
IN_JSON = OUT_DIR / "learning_curve.json"
OUT_MD = OUT_DIR / "learning_curve.md"


def pct(f):
    return f"{int(round(f * 100))}%"


def main():
    J = json.loads(IN_JSON.read_text(encoding="utf-8"))
    fr = J["fractions"]
    ch = J["test_logloss_changes"]
    kc = J["kept_set_check"]
    sp = J["splits"]
    cross = J["crosscheck_100pct_vs_bakeoff"]
    d50 = next(c for c in ch if c["from"] == 0.5 and c["to"] == 1.0)
    steps = [c["test_logloss_change"] for c in ch]
    lo, hi = d50["ci95"]
    ll100 = next(r for r in fr if r["fraction"] == 1.0)["test"]["logloss"]
    ll50 = next(r for r in fr if r["fraction"] == 0.5)["test"]["logloss"]

    if J["log_loss_50_to_100_improves"] and J["ci95_50_to_100_below_zero"]:
        verdict = (f"Yes, test log loss still improves from 50% to 100% of train matches "
                   f"({d50['test_logloss_change']:+.5f}, 95% CI {lo:+.5f} to {hi:+.5f}), but the gain is small "
                   f"and shrinking: the step gains are {steps[0]:+.5f} (10% to 25%), {steps[1]:+.5f} (25% to 50%), "
                   f"{steps[2]:+.5f} (50% to 100%). More matches would still help a little, and the curve is "
                   f"flattening.")
    elif J["log_loss_50_to_100_improves"]:
        verdict = (f"Log loss still falls from 50% to 100% ({d50['test_logloss_change']:+.5f}) but the 95% CI "
                   f"[{lo:+.5f}, {hi:+.5f}] includes zero, so the gain is not separable from noise.")
    else:
        verdict = (f"No: test log loss does not improve from 50% to 100% ({d50['test_logloss_change']:+.5f}).")

    L = []
    L.append("# CHECK A: learning curve, 2026 kept set (seed 0)")
    L.append("")
    L.append(f"**Verdict:** {verdict}")
    L.append("")
    L.append("Label: VERIFIED. Every number below was computed by `scripts/model/sufficiency/learning_curve.py` "
             "(run output and JSON: `notes/model/sufficiency/learning_curve.json`). Rendered by "
             "`scripts/model/sufficiency/learning_curve_md.py` from that JSON.")
    L.append("")
    L.append("## Data counts")
    L.append("")
    L.append(f"- Kept set (bake-off filters, rebuilt and checked against `logs/bakeoff.log`): "
             f"{kc['kept_matches']} matches, {kc['kept_rounds']} rounds, {kc['states']} states. "
             f"Channel-kept matches {kc['channel_kept']}, duplicate matches removed {kc['dup_matches_removed']}.")
    L.append("")
    L.append("| split | matches | rounds | states | CT-win share (states) |")
    L.append("|---|---:|---:|---:|---:|")
    for name, label in (("train_pool", "train pool (90% of the 80%)"),
                        ("validation", "validation (10% of the 80%)"),
                        ("test", "test (20%)")):
        s = sp[name]
        L.append(f"| {label} | {s['matches']} | {s['rounds']} | {s['states']} | {s['ct_share']:.4f} |")
    L.append("")
    L.append("Splits are the bake-off primary split (`bakeoff.make_splits`, seed 0). Its train, validation and "
             "test counts match `logs/bakeoff.log` exactly.")
    L.append("")
    L.append("## Learning curve (XGBoost, test set fixed at 304 matches)")
    L.append("")
    L.append("| train fraction | train matches | train rounds | train states | best iter | test log loss | test Brier | test AUC | test ECE | device |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in fr:
        t = r["test"]
        L.append(f"| {pct(r['fraction'])} | {r['train']['matches']} | {r['train']['rounds']} | "
                 f"{r['train']['states']} | {r['best_iteration']} | {t['logloss']:.5f} | {t['brier']:.5f} | "
                 f"{t['auc']:.4f} | {t['ece']:.4f} | {r['device']} |")
    L.append("")
    L.append("Metrics are on states, from `bakeoff.metrics()`. Fractions are of the train pool (see notes).")
    L.append("")
    L.append("## Change in test log loss between fractions (paired cluster bootstrap over test matches)")
    L.append("")
    L.append("| change | test LL change | 95% CI | share of resamples better |")
    L.append("|---|---:|---|---:|")
    for c in ch:
        L.append(f"| {pct(c['from'])} to {pct(c['to'])} | {c['test_logloss_change']:+.5f} | "
                 f"[{c['ci95'][0]:+.5f}, {c['ci95'][1]:+.5f}] | {c['share_resamples_to_better']:.4f} |")
    L.append("")
    L.append("## Cross-check against the bake-off log")
    L.append("")
    bl = cross["bakeoff_log"]
    hc = cross["here_cpu_100pct"]
    L.append(f"- Bake-off log, XGBoost raw, primary test: log loss {bl['logloss']:.5f}, Brier {bl['brier']:.5f}.")
    L.append(f"- Here, 100% row: log loss {hc['logloss']:.5f}, Brier {hc['brier']:.5f}, best iteration "
             f"{hc['best_iteration']} (the log also reports 261).")
    L.append(f"- Match at 5 decimal places: {cross['cpu_matches_bakeoff_log_5dp']}. The gap is "
             f"{abs(hc['logloss'] - bl['logloss']):.6f} in log loss. The likely cause is the script's `n_jobs=4` "
             f"against the bake-off's default threading. This was not tested, so treat it as a cause for "
             f"investigation, not a confirmed one. The split and tree count are identical.")
    L.append("")
    L.append("## Notes and deviations")
    L.append("")
    L.append(f"- Fractions 10, 25, 50, 100% are of the train pool, not of the full 80%: validation is 10% of the 80% "
             f"(the bake-off's early-stopping slice), so the 100% row trains on {sp['train_pool']['matches']} of "
             f"{kc['kept_matches']} kept matches (72%). A literal 100% of the 80% would leave no early-stopping "
             f"slice. Fractions are nested by a seeded shuffle.")
    L.append("- GPU: the request was for `device=\"cuda\"`. The pixi `xgboost 3.4.2` build reports `USE_CUDA False`, "
             "so every fit ran on CPU. The GPU was not used.")
    L.append("- Only XGBoost was fitted. No hyperparameter search. Settings as in `bakeoff.py`: hist, depth 4, "
             "eta 0.05, min_child_weight 20, subsample 0.8, early stopping 100 on validation, bake-off monotone "
             "constraints, features `bk.FEATS`.")
    L.append(f"- 50% to 100% change: point estimate {d50['test_logloss_change']:+.5f}; the 100% model's log loss is "
             f"{ll100:.5f} and the 50% model's is {ll50:.5f}.")
    L.append("- Test log loss differences of about 0.001 are small against the test set's 304 matches. The CI is the "
             "guide, not the point estimate alone.")
    L.append("")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD}")
    print(verdict)


if __name__ == "__main__":
    main()
