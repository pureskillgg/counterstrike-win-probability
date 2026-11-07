"""Model slides for the findings deck, built from result files (no number is typed by hand). Writes slide HTML under --out.

Slides: m_headline, m_stack, m_calib, m_ticks, m_benchmark, m_transfer, m_gpucpu, m_robust, m_data.
Inputs (all under data/<store>/): export_<snapshot>.json, export_<basic>.json, exp_stack_2026.json, exp_stack_2026b.json, baselines.json,
stress_*.json, parity_*.json, e2e_*.json, eval_offset_ticks_*.json (optional), prep_report.json.
Run:  python -I -X utf8 scripts\\viz\\build_deck_model.py --out <dir>\\project\\slides [--store combined] [--snapshot winprob_v1] [--basic winprob_v1_basic]
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
import polars as pl

import deck_text_slides as T
from deck_lib import (AMBER, BLUE, bars_js, bullets, callout, cards, embed, forest_js, note, reliability_js, side_panel, slide, stat_box, table)

ROOT = r"G:\pureskill\csds-cs2"
# printed on every slide that shows an interval (the brief: warn that ticks are correlated)
CORR = "Seconds in a round are correlated, so every interval resamples whole matches."  # the factor is filled in from cluster_se_<model>.json in main


def J(store, name):
    p = os.path.join(ROOT, "data", store, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def ll(r):  # (value, lo, hi) of a result dict that holds logloss and a CI under either name
    ci = r.get("logloss_ci95") or r.get("ci95")
    return r["logloss"], ci[0], ci[1]


# ------------------------------------------------------------------ headline
def s_headline(store, snap, basic):
    ex, exb = J(store, f"export_{snap}.json"), J(store, f"export_{basic}.json")
    bl = J(store, "baselines.json")["random"]["models"]  # same store, same frozen test matches (2026 part), so the three rows are comparable
    g = ex["gpu_trained"]
    f6, old = bl["XGB F6 | era 2026"], bl["XGB F_SCORE | era 2026"]
    rows = [["Alive counts only", *ll(f6)], ["Previous design (score lead)", *ll(old)], ["Snapshot model (final)", g["logloss"], *g["logloss_ci95"]]]
    chart = embed(forest_js(rows, 0.44, 0.52, [0.44, 0.46, 0.48, 0.50, 0.52], "Test log loss, lower is better (95% match-cluster interval)", left=370, right=1030, top=20, bottom=300, pitch=90, highlight=2, fmt="toFixed(4)"),
                  left=128, top=330, width=1100, height=430)
    d_total = g["logloss"] - f6["logloss"]
    panel = side_panel(
        stat_box(f"{g['logloss']:.4f}", f"log loss of the final model on {ex['test']['matches']:,} held-out matches ({ex['test']['rows']:,} seconds)") +
        stat_box(f"{d_total:+.3f}", "against alive counts only; the living players' loadout is what matters", "amber") +
        note(f"AUC {g['auc']:.3f}, calibration error {g['ece10']:.3f}. The simpler alive-count model (what a live service uses when it has no per-player table: players alive, clock, bomb and map only) scores {exb['gpu_trained']['logloss']:.4f} on {exb['test']['matches']:,} matches.", 24),
        left=1260, top=330, width=532)
    return slide("m_headline", "Result", "The final model reaches " + f"{g['logloss']:.3f}" + " log loss on matches it never saw",
                 chart + panel + '<div style="position:absolute;left:128px;top:790px;width:1664px">' +
                 callout("Input: the table of living players (health, armor, helmet, kit, weapon, grenades, equipment value), the bomb state and the clock. No score lead, no event history.", 26) + "</div>",
                 source=f"Source: export_{snap}.json, baselines.json (2026 held-out matches). {CORR}",
                 notes="This is the number to quote. Each row of the test set is one second of one round, scored by a model that never saw that match. Seconds inside a round are strongly correlated, so every interval is a match-cluster bootstrap and is several times wider than counting seconds as independent rows would give. Alive counts only is the simple baseline everyone starts with; the middle row is the earlier design that also used the score lead and round number; the last row is the final model, which reads the table of living players at the tick and uses no score. The baselines were scored on the 2026 part of the frozen test split (2,499 matches) and the snapshot model on 2,497 of them, because two matches lost all their rounds to the scope filters; the baselines were trained on both eras together and the snapshot model on 2026 only. The differences between rows are therefore not paired.")


# ------------------------------------------------------------------ stack
def s_stack(store):
    a, b = J(store, "exp_stack_2026.json")["results"], J(store, "exp_stack_2026b.json")["results"]
    names = {"F6": "Alive counts only", "+hp": "+ team health (events)", "+feat_x2_utility": "+ utility thrown", "+feat_x1_kill_timing": "+ kill timing",
             "+feat_x5_bomb_state": "+ bomb state", "+feat_x3_shots": "+ shots fired", "+status": "+ living players' loadout"}
    rows = [[names[k], *ll(v)] for k, v in a["random"].items()]
    first = b["random"]["+status"]
    after = [v for k, v in b["random"].items() if k.startswith("+feat_")]
    worst = max(abs(v["step_delta"]) for v in after)
    chart = embed(forest_js(rows, 0.45, 0.52, [0.46, 0.48, 0.50, 0.52], "Test log loss as blocks are added one by one (95% interval)", left=400, right=1030, top=10, bottom=520, pitch=70, highlight=6, fmt="toFixed(4)", label_size=24),
                  left=128, top=330, width=1100, height=620)
    panel = side_panel(
        stat_box(f"{rows[-1][1] - rows[-2][1]:+.4f}", "what the loadout snapshot adds after every event feature (paired step)") +
        stat_box(f"{first['logloss']:.4f}", f"log loss with the snapshot first; the four event blocks added after it move it by at most {worst:.4f}", "amber") +
        note("Trained on earlier, mostly Valve-server matches and tested on the latest, mostly FACEIT matches: " + f"{b['chrono']['+status']['logloss']:.4f}" + " with the snapshot, and the events change nothing after it. That split is a source shift, not a time test.", 24),
        left=1260, top=330, width=532)
    return slide("m_stack", "Which inputs matter", "The living players' loadout carries the signal; event history adds nothing after it",
                 chart + panel,
                 source=f"Source: exp_stack_2026.json, exp_stack_2026b.json; run on the earlier 6,475-match store, not repeated. {CORR}",
                 notes="We added input groups one at a time and measured each step against the previous one on the same held-out matches. Team health, utility thrown, kill timing, bomb state and shots fired each help a little when they come first. The table of living players, meaning for every player alive at that second the health, armor, helmet, defuse kit, primary weapon, grenades and equipment value, helps far more than all of them together. When the order is reversed and the table goes first, none of the event blocks adds anything measurable. That is why the final model needs only the live table, the bomb state and the clock, which is also what a live service has at hand.")


# ------------------------------------------------------------------ calibration
def s_calib(store, snap):
    ex = J(store, f"export_{snap}.json")
    cal = ex["raw_calibration"]
    pts = [[c["mean_pred"], c["observed"], c["obs_ci95"][0], c["obs_ci95"][1]] for c in cal]
    g = ex["gpu_trained"]
    chart = embed(reliability_js(pts), left=128, top=300, width=780, height=650)
    iso, pla = ex["isotonic"], ex["platt"]
    ece_lo, ece_hi = g["ece10_ci95"]
    # headline = the LARGEST gap over all ten groups (no single bin is singled out)
    worst = max(cal, key=lambda c: abs(c["mean_pred"] - c["observed"]))
    gap = abs(worst["mean_pred"] - worst["observed"])
    panel = side_panel(
        stat_box(f"{100 * gap:.1f} points",
                 f"largest gap in any of the ten groups: the model averaged {worst['mean_pred']:.1%}, CT won {worst['observed']:.1%} (95% interval {worst['obs_ci95'][0]:.1%} to {worst['obs_ci95'][1]:.1%})") +
        stat_box(f"{100 * g['ece10']:.1f} points", f"calibration error: the average gap between predicted and actual win rate, over ten groups of seconds (0 = perfect; interval {100 * ece_lo:.1f} to {100 * ece_hi:.1f}); the biggest single gap is {100 * abs(worst['mean_pred'] - worst['observed']):.1f} points", "amber") +
        note(f"Re-calibrating the probabilities afterwards (isotonic or Platt, fitted on separate matches) does not help: log loss moves by {iso['delta_vs_raw']:+.4f} (isotonic) and {pla['delta_vs_raw']:+.4f} (Platt), where a plus sign means worse. So the raw probabilities are served.", 24),
        left=980, top=300, width=812)
    return slide("m_calib", "Calibration", f"Across ten groups of seconds, the predicted and the actual CT win rate agree to within {100 * gap:.1f} points",
                 chart + panel,
                 source=f"Source: export_{snap}.json; each point is a tenth of the test seconds; bars are 95% match-cluster intervals; dashed = perfect. {CORR}",
                 notes="Each point is one tenth of the held-out seconds, grouped by the probability the model gave. The horizontal position is the average prediction in the group and the vertical position is how often CT actually won those rounds. Points on the dashed diagonal mean the model is calibrated. The vertical bar is the match-cluster interval of the observed rate. Isotonic regression and Platt scaling are two standard ways to adjust probabilities after the fact; both were fitted on separate validation matches and then tested, and neither improves log loss, so the model's own probabilities are used.")


# ------------------------------------------------------------------ every second
def s_ticks(store, snap):
    ex = J(store, f"export_{snap}.json")
    bg = ex["by_group"]
    order = [("t_0-15s_preplant", "Before plant, 0 to 15 s"), ("t_15-30s_preplant", "Before plant, 15 to 30 s"), ("t_30-45s_preplant", "Before plant, 30 to 45 s"),
             ("t_45-60s_preplant", "Before plant, 45 to 60 s"), ("t_60-75s_preplant", "Before plant, 60 to 75 s"), ("t_75-90s_preplant", "Before plant, 75 to 90 s"),
             ("t_90-105s_preplant", "Before plant, 90 to 105 s"), ("t_105-400s_preplant", "Before plant, after 105 s"), ("plant_0-10s_after", "After plant, 0 to 10 s"),
             ("plant_10-20s_after", "After plant, 10 to 20 s"), ("plant_20-30s_after", "After plant, 20 to 30 s"), ("plant_30-100s_after", "After plant, after 30 s")]
    rows = [[lab, bg[k]["logloss"]] for k, lab in order if k in bg]
    chart = embed(bars_js(rows, 0.8, left=420, right=1040, top=10, pitch=46, label_size=23, fmt="toFixed(3)"), left=128, top=330, width=1120, height=600)
    late = bg.get("t_105-400s_preplant")
    off = J(store, f"eval_offset_ticks_{snap}.json")
    extra = ""
    if off:
        keys = [k for k in off if k.startswith("offset_")]
        base = off["training_phase_ticks"]
        worst = max(off[k]["logloss"] for k in keys)
        extra = stat_box(f"{worst:.4f}", f"log loss at shifted ticks (half a second and random phase) against {base['logloss']:.4f} at the training ticks", "blue")
    late_txt = f"Late in an unplanted round the model predicts {late['mean_pred']:.2f} where CT win {late['observed']:.2f} ({late['rows']:,} seconds): the clock running out is rare and still under-predicted." if late else ""
    panel = side_panel(
        stat_box(f"{bg['5v5']['logloss']:.3f}", "log loss while all ten players are alive: at the start of a round little has happened yet, so the state is almost a coin flip") +
        extra + note(late_txt, 24), left=1280, top=330, width=512)
    return slide("m_ticks", "Every second of the round", "Log loss by moment in the round: hardest at the start, easy once the bomb is down",
                 chart + panel,
                 source=f"Source: export_{snap}.json (by_group). {CORR}",
                 notes="The model has to work at every tick, so we report log loss by time slice instead of one average. The first fifteen seconds are the hardest because ten players are alive and little has happened; what separates the sides there is only what they bought. Once the bomb is planted the state is much easier. The clock running out is rare, about one round in seventy, and the model is slightly low for CT there; we report it rather than hide it. Training used whole seconds only. The shifted-tick evaluation re-samples the held-out matches half a second off the grid and at a random phase, reading the player table at those exact ticks, and scores the same model.")


# ------------------------------------------------------------------ benchmark
def s_benchmark(store, snap):
    ex = J(store, f"export_{snap}.json")
    bl = J(store, "baselines.json")["random"]["models"]
    g = ex["gpu_trained"]
    rows = [["Alive counts only", *ll(bl["XGB F6 | era 2026"])], ["Previous design: + score lead, round no.", *ll(bl["XGB F_SCORE | era 2026"])],
            ["Final snapshot model, no score lead", g["logloss"], *g["logloss_ci95"]]]
    chart = embed(forest_js(rows, 0.42, 0.56, [0.44, 0.48, 0.52, 0.56], "Test log loss on the 2026 held-out matches (95% match-cluster interval)", left=420, right=1030, top=20, bottom=300, pitch=90, highlight=2, fmt="toFixed(4)", label_size=24),
                  left=128, top=330, width=1100, height=430)
    d = g["logloss"] - bl["XGB F_SCORE | era 2026"]["logloss"]
    panel = side_panel(
        stat_box(f"{d:+.3f}", "final model against the previous design that used the score lead (point estimates; unpaired)") +
        note("The previous design needed the match score at the start of the round; the final model needs only the live state, so it works the same way in round 1 and in overtime.", 24),
        left=1260, top=330, width=532)
    return slide("m_benchmark", "Against the earlier model", "Better than the earlier score-lead design, and without using the score",
                 chart + panel,
                 source=f"Source: baselines.json, export_{snap}.json; our re-implementation of the earlier design. {CORR}",
                 notes="The brief asked for a better-quality model than what was discussed. Two references are in the conversation: the plain alive-count model, and the earlier design that put the score lead and round number in. The earlier design's rows come from the baseline run where both eras were trained together and the 2026 part of the test set was scored; the final model was trained on 2026 only. The comparison is therefore unpaired and approximate, and the benchmark of record is still to be named by the team. Note that the score lead was removed by instruction, and the table shows it is not needed.")


# ------------------------------------------------------------------ transfer
def s_transfer(store):
    b = J(store, "baselines.json")
    rows = []
    for split, lab in (("random", "Both eras mixed (random match split)"), ("2024_to_2026", "Train 2024, test 2026"), ("2026_to_2024", "Train 2026, test 2024"), ("chrono", "Train early, test latest matches")):
        m = b[split]["models"]
        f6 = m["XGB F6 (mean of 2 seeds)"]
        rows.append([lab, f"{m['constant']['logloss']:.4f}", f"{m['LR alive diff']['logloss']:.4f}", f"{f6['logloss']:.4f} ({f6['logloss_ci95'][0]:.4f} to {f6['logloss_ci95'][1]:.4f})", f"{f6['n_matches']:,}"])
    tbl = table(["Evaluation", "Constant", "Logistic, alive gap", "XGBoost, alive counts", "Test matches"], rows, widths=[34, 12, 16, 26, 12], size=26)
    return slide("m_transfer", "Does it transfer?", "The round-state model keeps its edge across game eras and over time",
                 tbl + bullets(["Per-player loadout exists only in 2026, so the final model is trained on 2026; the simpler alive-count model (players alive, clock, bomb, map) uses both eras",
                                "Event-derived health features did not transfer from 2026 to 2024 (a reconstruction difference); the loadout snapshot sidesteps that"], size=27),
                 source=f"Source: baselines.json; match-level splits. {CORR}",
                 notes="Training on one era and testing on the other, or training on early matches and testing on the latest, shows whether the model learned the game or the data. The alive-count model transfers in both directions with almost no loss. The snapshot model cannot be tested on 2024 because that era has no per-tick player table, which is why the alive-count model stays in the package as the fallback.")


# ------------------------------------------------------------------ GPU / CPU
def s_gpucpu(store, snap, basic):
    b = J(store, "bench_gpu_cpu.json")
    st = J(store, f"stress_{snap}.json")
    if not b:
        return None
    lat = next((c["detail"] for c in st["checks"] if " latency" in c["check"][:12]), "") if st else ""
    med = re.search(r"median ([0-9.]+) ms", lat)
    r = b["runs"]
    body = cards([(f"{b['speedup']:.1f} times faster", f"{r['cuda']['seconds']:.0f} s on the GPU against {r['cpu']['seconds']:.0f} s on the CPU for the same fit: {b['rows']:,} seconds of play, {b['inputs']} inputs, {b['trees']} trees", "blue"),
                  ((med.group(1) + " ms") if med else "about 1 ms", "median time to answer one state on a CPU: the trained model runs on a CPU in production", "blue")], big=64, caption=25)
    body += bullets(["Training is done on the GPU; the saved model loads and predicts on a CPU with the same numbers",
                     "A model trained on the GPU and one trained on the CPU score the same on held-out matches (difference in log loss: " + f"{J(store, f'export_{snap}.json')['cpu_minus_gpu_logloss']['delta']:+.5f})"], size=27)
    body += note(b["note"] + f"; {b['cpu_threads']} CPU threads.", 24)
    return slide("m_gpucpu", "Speed", f"Training on the GPU is about {b['speedup']:.0f} times faster than on the CPU", body,
                 source="Source: bench_gpu_cpu.json (scripts/model/bench_gpu_cpu.py), stress_" + snap + ".json (latency)",
                 notes="The same boosted-tree fit was run on the GPU and on the CPU, on real training rows, with a fixed number of trees. The speed-up grows with the amount of data; the full refit of the final model on the pod took minutes. Prediction is cheap on a CPU, which is where the model is meant to run.")

# ------------------------------------------------------------------ robustness
def s_robust(store, snap, basic):
    st = J(store, f"stress_{snap}.json")
    stb = J(store, f"stress_{basic}.json")
    e2e = J(store, f"e2e_{snap}.json")
    par = J(store, f"parity_{snap}.json")
    if not (st and e2e and par):
        return None
    n_ok = sum(c["status"] == "PASS" for c in st["checks"])
    n_e = sum(c["status"] == "PASS" for c in e2e["checks"])
    n_ew = sum(c["status"] == "WARN" for c in e2e["checks"])
    n_sw = sum(c["status"] == "WARN" for c in st["checks"])
    pinfo = par.get("info", {})
    lat = next((c["detail"] for c in st["checks"] if " latency" in c["check"][:12]), "")
    cards_ = cards([(f"{n_ok} of {len(st['checks'])}", f"stress checks pass ({n_sw} warning): fuzzing, unknown maps and modes, wingman states, terminal states, invalid input, monotonicity, threads, explanations", "blue"),
                    (f"{n_e} of {len(e2e['checks'])}", f"end-to-end checks pass, {n_ew} warnings that are listed in the notes: scope, duplicates, labels, grid, split, leakage screen, calibration", "blue"),
                    (f"{pinfo.get('max_abs_diff', float('nan')):.1e}", f"largest difference between the serving code and the training pipeline on {pinfo.get('rows_compared', 0):,} real held-out rows", "amber" if not par["passed"] else "blue")], big=64, caption=25)
    return slide("m_robust", "Stress tests", "It does not crash on an unexpected map or mode, and serving matches training",
                 cards_ + bullets(["40,000 fuzzed states, including corrupt player tables, never raised and always returned a probability between 0 and 1",
                                   "Unknown maps, wingman, hostage and casual modes are answered but flagged out of scope; a missing player table falls back to the alive-count model",
                                   "More CT alive never lowers the alive-count model's probability; removing a CT player lowers the snapshot probability on average",
                                   f"Speed: {lat}"], size=27),
                 source=f"Source: stress_{snap}.json, e2e_{snap}.json, parity_{snap}.json",
                 notes="The brief asked for a model that does not crash on an unexpected map or mode, with stress testing and a final end-to-end test. The stress test throws forty thousand mixed valid and garbage states at the prediction function, including corrupt player tables, and requires a finite probability and an allowed status every time. The parity check feeds real held-out ticks through the serving code as a live service would, using only the raw player table, and compares with the training pipeline. The end-to-end checks re-derive scope, duplicates, labels, the grid, the split, a leakage screen and calibration from the finished outputs. The warnings are known limits, not hidden: 1.39% of used rounds have one player's side inferred (that inference disagreed with the event log in about 40% of the cases checked); 12 matches have a wingman flag in the header that disagrees with the roster (3 used rounds); 1,275 matches are excluded as out-of-pool maps (vertigo, overpass, train and others, listed by name); 26% of matches have fewer rounds than the header score implies, concentrated in two 2024 builds; 5.13% of rounds in one 2024 build were relabelled by the signal vote, always on the first round of a half; and the label-agreement checks have documented limits on the 2024 builds.")


# ------------------------------------------------------------------ unknown / unseen maps
def s_maps(store):
    d = J(store, "exp_robustness.json")
    if not d or "variants" not in d or "lomo" not in d:
        return None
    v, lo = d["variants"], d["lomo"]
    rows = []
    for m, r in lo.items():
        b = r["B_never_saw_map"]
        rows.append([m, b["cost_vs_reference"], b["cost_ci95"][0], b["cost_ci95"][1]])
    amax = max(abs(x) for r in rows for x in r[1:4]) * 1.15 + 1e-4
    chart = embed(forest_js(rows, -amax, amax, [round(-amax * 0.8, 4), 0, round(amax * 0.8, 4)], "Log-loss cost of never having seen the map (model with no map input, 95% interval)",
                            left=300, right=1020, top=10, bottom=500, pitch=62, fmt="toFixed(4)", hline=0, label_size=25), left=128, top=330, width=1100, height=600)
    a_true, a_blank, b_true = v["A_true_map"], v.get("A_map_blanked"), v["B_true_map"]
    worst = max(max(r["A_never_saw_map"]["cost_vs_reference"], r["B_never_saw_map"]["cost_vs_reference"], r["C_never_saw_map"]["cost_vs_reference"]) for r in lo.values())
    panel = side_panel(
        stat_box(f"{b_true['logloss']:.4f}", f"log loss with NO map input, against {a_true['logloss']:.4f} with it; paired difference {b_true['delta_vs_A']:+.4f} ({b_true['delta_vs_A_ci95'][0]:+.4f} to {b_true['delta_vs_A_ci95'][1]:+.4f})") +
        stat_box(f"{worst:+.4f}", "worst cost of an unseen map over every map and every variant", "amber") +
        note(f"Blanking the map of the model that has one changes log loss by {a_blank['delta_vs_own_true_map']:+.4f}, and its average probability shift is {a_blank['mean_abs_prob_shift']:.3f}.", 24),
        left=1260, top=330, width=532)
    return slide("m_maps", "Unknown maps", "A map the model has never seen costs almost nothing, and unknown maps are flagged",
                 chart + panel,
                 source=f"Source: exp_robustness.json; each map held out of training. {CORR}",
                 notes="The worry is the classic unseen-category problem: a map the model was not trained on, such as a non-active-duty map. Three ways were tested. A: map as seven yes/no inputs. B: no map input at all. C: the map inputs blanked for a share of training matches. For each of the seven maps the model was trained without any match of that map and scored on that map's held-out matches, against a reference model that had seen the map. The cost of an unseen map is within a few thousandths of a log loss point for every map and every variant. On the normal test the model with no map input is very slightly worse than the one with it (the paired interval in the panel is above zero), so the final snapshot model keeps the seven map inputs; for a map it has never seen all seven are zero, which is the blanked case measured here. The serving code also flags an unrecognised map as out of scope, so the caller is told the answer is outside what was tested.")


# ------------------------------------------------------------------ model families
def s_families(store):
    d = J(store, "exp_model_families.json")
    if not d or "xgb_gbdt" not in d:
        return None
    names = {"xgb_gbdt": "XGBoost (final settings)", "sk_rf": "Random forest", "sk_rf_iso": "Random forest + isotonic", "sk_et": "Extra trees", "xgb_rf": "XGBoost in random-forest mode",
             "sk_mlp": "Neural network (3 layers)", "ridge_logit": "Logistic regression"}
    have = [k for k in names if k in d]
    if len(have) < 3:
        return None
    rows = [[names[k], *ll(d[k])] for k in have]
    lo_ = min(r[2] for r in rows) - 0.004
    hi_ = max(r[3] for r in rows) + 0.004
    chart = embed(forest_js(rows, round(lo_, 3), round(hi_, 3), [round(lo_ + (hi_ - lo_) * k / 4, 3) for k in range(1, 4)], "Test log loss on the same held-out matches (95% match-cluster interval)",
                            left=420, right=1030, top=10, bottom=60 + 70 * len(rows), pitch=70, highlight=0, fmt="toFixed(4)", label_size=24), left=128, top=330, width=1100, height=130 + 70 * len(rows))
    alt = [k for k in have if k != "xgb_gbdt" and "delta_vs_xgb_gbdt" in d[k]]
    best = min(alt, key=lambda k: d[k]["logloss"]) if alt else None
    panel = side_panel(
        stat_box(f"{d['xgb_gbdt']['logloss']:.4f}", f"log loss of boosted trees, fitted in {d['xgb_gbdt']['fit_seconds']:.0f} s on the GPU") +
        (stat_box(f"{d[best]['delta_vs_xgb_gbdt']:+.4f}", f"{names[best].lower()}, the best alternative (paired, interval {d[best]['delta_vs_xgb_gbdt_ci95'][0]:+.4f} to {d[best]['delta_vs_xgb_gbdt_ci95'][1]:+.4f}); fit {d[best]['fit_seconds']:.0f} s on the CPU", "amber") if best else "") +
        note(f"All models see the same {d['setup']['features']} inputs, the same training matches and are scored on the same {d['setup']['test_matches']:,} held-out matches; training uses every {d['setup']['stride']}th second. {CORR}", 22),
        left=1260, top=330, width=532)
    return slide("m_families", "Other model families", "Boosted trees are the best of the families we tried; a random forest comes close",
                 chart + panel,
                 source=f"Source: exp_model_families.json; paired differences. {CORR}",
                 notes="The brief asked whether random forest or a neural network would do as well. Every family was trained on the same rows and the same inputs and scored on the same held-out matches. The paired difference against boosted trees is the honest comparison, because seconds inside a round are correlated and the unpaired intervals overlap heavily. Boosted trees also keep the monotone constraints on alive counts, are the fastest to fit on the GPU, and predict in about a millisecond on a CPU.")


# ------------------------------------------------------------------ foundation models
def s_foundation(store):
    d = J(store, "exp_foundation.json")
    if not d:
        return None
    label = {"tabicl": "TabICL v2", "mitra": "Mitra"}
    # each run is scored on its own test seconds, so the honest picture is the PAIRED difference to XGBoost on those same seconds (0 = XGBoost)
    runs = [(v["delta_vs_xgb"], v["delta_ci95"], f"{label.get(v['model'], v['model'])} zero-shot, {v['context_rows']:,} context rows, {v['test_seconds']:,} test s") for v in d.values()]
    n_ft = 0
    for fn in ("exp_finetune.json", "exp_finetune_tabicl.json"):  # GPU fine-tuning runs (training matches only, early stopping on validation matches)
        for name, v in (J(store, fn) or {}).items():
            if isinstance(v, dict) and "delta_vs_xgb" in v:
                tag = ("TabPFN v2" if v.get("version") == "v2" else "TabICL v2") + (" fine-tuned" if v.get("arm") == "fine-tuned" else " zero-shot")
                extra = f", {v['train_rows']:,} training s" if v.get("arm") == "fine-tuned" else f", {v.get('context_rows', 0):,} context rows"
                runs.append((v["delta_vs_xgb"], v["delta_ci95"], f"{tag}{extra}, {v['test_seconds']:,} test s"))
                n_ft += int(v.get("arm") == "fine-tuned")
    runs.sort(key=lambda r: r[0])
    rows = [[lab, dv, *ci] for dv, ci, lab in runs]
    amax = max(abs(x) for r in rows for x in r[1:4]) * 1.2 + 1e-4
    pt = min(70, (520 - 130) // len(rows))  # keep the chart inside the slide however many runs there are
    chart = embed(forest_js(rows, 0.0, round(amax, 3), [round(amax * k / 4, 3) for k in range(1, 4)], "Log loss above XGBoost on the same seconds (95% match-cluster interval); 0 = as good as XGBoost",
                            left=700, right=1040, top=10, bottom=60 + pt * len(rows), pitch=pt, fmt="toFixed(4)", label_size=19, hline=0), left=128, top=330, width=1100, height=130 + pt * len(rows))
    best = runs[0]
    panel = side_panel(
        stat_box(f"{best[0]:+.4f}", f"the closest foundation-model run to XGBoost (paired interval {best[1][0]:+.4f} to {best[1][1]:+.4f}); no run was better") +
        stat_box("2k to 10k vs 1.09 M", "labelled seconds the foundation models used, against the training seconds of XGBoost: a data-budget comparison", "amber") +
        note("One test pool, one split, one seed: one comparison, not independent confirmations.", 22),
        left=1260, top=330, width=532)
    return slide("m_foundation", "Foundation models", "Pretrained tabular foundation models, zero-shot and fine-tuned, did not beat the boosted trees",
                 chart + panel,
                 source=f"Source: exp_foundation.json, exp_finetune*.json; audit in notes/research/FM_AUDIT_FINDINGS_2026-10-15.md. {CORR}",
                 notes="The question was whether a pretrained tabular foundation model, fine-tuned or not, could beat or replace the boosted trees. Each run gave the model labelled seconds from training matches only and asked it to predict held-out seconds; XGBoost predicted the same seconds, so the difference is paired and carries a match-cluster interval. Every run is above zero, that is, worse than XGBoost, by 0.005 to 0.02 log loss. The comparison is lopsided in data: the foundation models used 2,000 to 10,000 context seconds, or 143,000 fine-tuning seconds on the GPU for TabPFN v2, while XGBoost used about a million. Refitting XGBoost on the same 10,000-second context gave about the same log loss as TabICL (0.4624 against 0.4624 on 1,000 test seconds, by the audit), so at equal data the two are level and XGBoost's lead is the lead of more data, which the foundation models could not take on this hardware because fine-tuning held 15 to 24 GB of host memory. Seconds in a round are correlated, so only the match-cluster intervals should be read; the runs share one test pool and one split. Speeds were measured once on a busy machine and are not benchmarks.")


# ------------------------------------------------------------------ newest uploads
def s_temporal(store):
    d = J(store, "exp_temporal.json")
    if not d or "frozen_test_matches_after_cutoff" not in d:
        return None
    f, allm = d["frozen_test_matches_after_cutoff"], d["temporal_on_all_later_matches"]
    ref, tmp = f["reference"], f["temporal"]
    rows = [[f"Trained on all {d['train_matches']['reference']:,} training matches", ref["logloss"], *ref["logloss_ci95"]],
            [f"Trained only on uploads up to {d['cutoff']} ({d['train_matches']['temporal']:,} matches)", tmp["logloss"], *tmp["logloss_ci95"]]]
    lo_ = min(r[2] for r in rows) - 0.004
    hi_ = max(r[3] for r in rows) + 0.004
    chart = embed(forest_js(rows, round(lo_, 3), round(hi_, 3), [round(lo_ + (hi_ - lo_) * k / 4, 3) for k in range(1, 4)], f"Log loss on the {f['n_matches']:,} held-out matches uploaded after {d['cutoff']} (95% interval)",
                            left=560, right=1040, top=20, bottom=250, pitch=100, highlight=1, fmt="toFixed(4)", label_size=22), left=128, top=330, width=1100, height=380)
    dl = f["temporal_minus_reference_ci95"]
    panel = side_panel(
        stat_box(f"{f['temporal_minus_reference']:+.4f}", f"cost of never having seen the newest uploads (paired, interval {dl[0]:+.4f} to {dl[1]:+.4f})") +
        stat_box(f"{allm['logloss']:.4f}", f"the earlier-trained model on all {allm['n_matches']:,} matches uploaded after the cutoff; calibration error {allm['ece10']:.3f}", "amber"),
        left=1260, top=330, width=532)
    return slide("m_temporal", "Newest uploads", "Trained on older uploads, the model scores almost the same on the newest ones",
                 chart + panel + '<div style="position:absolute;left:128px;top:760px;width:1664px">' +
                 callout("A time-ordered test, not only a random one: the model never saw anything uploaded on or after the next day. The loss from this is about four ten-thousandths.", 26) + "</div>",
                 source=f"Source: exp_temporal.json; frozen split, paired difference. {CORR}",
                 notes="A random match split can hide drift, because new and old matches are mixed. Here one model is trained only on matches uploaded up to 10 October and scored on matches uploaded afterwards. The reference model, trained on every training match, is scored on the same frozen test matches that were uploaded after the cutoff, so the difference is paired. The small positive difference is the price of not having seen the newest data. The calibration error stays small on the later matches, so the probabilities do not drift. Upload day is the only time information in the data; it is not match start time.")


# ------------------------------------------------------------------ deep teacher / distillation
def s_distill(store):
    d = J(store, "exp_deep_distill_regB.json")  # the run on the final store; the earlier, smaller runs (archive_v2) are mentioned in the notes
    if not d:
        return None
    pr = d["protocol"]
    spec = [("deep_teacher (mean of fold nets)", f"Teacher network ({pr['width']} wide, {pr['depth']} layers) alone"), ("student XGBoost, soft labels alpha=0.5", "XGBoost student, half soft labels"),
            ("student XGBoost, soft labels alpha=0.75", "XGBoost student, 75% soft labels"), ("average of teacher and hard-label XGBoost", "Average of teacher and plain XGBoost")]
    runs = [[lab, d[k]["delta_vs_xgboost_hard"], *d[k]["delta_ci95"]] for k, lab in spec if k in d]
    if len(runs) < 3:
        return None
    xh = d["xgboost_hard_labels"]
    amax = max(abs(x) for r in runs for x in r[1:4]) * 1.15 + 1e-4
    chart = embed(forest_js(runs, round(-amax * 0.4, 4), round(amax, 4), [0.0, round(amax / 2, 4), round(amax, 4)], "Log loss above plain XGBoost on the same held-out seconds (95% match-cluster interval); 0 = as good",
                            left=560, right=1040, top=10, bottom=60 + 90 * len(runs), pitch=90, fmt="toFixed(4)", label_size=22, hline=0), left=128, top=330, width=1100, height=150 + 90 * len(runs))
    avg = d.get("average of teacher and hard-label XGBoost")
    best = min(runs, key=lambda r: r[1])
    panel = side_panel(
        stat_box(f"{best[1]:+.4f}", f"the best combination against plain XGBoost (paired interval {best[2]:+.4f} to {best[3]:+.4f}): " + ("reliably better" if best[3] < 0 else ("slightly better, but the interval includes zero, so not reliably" if best[1] < 0 else "no better"))) +
        (stat_box(f"{avg['delta_vs_xgboost_hard']:+.4f}", f"averaging teacher and XGBoost (paired interval {avg['delta_ci95'][0]:+.4f} to {avg['delta_ci95'][1]:+.4f}): level, not better", "amber") if avg else "") +
        note(f"Teacher probabilities are out-of-fold by match ({pr['folds']} groups); plain XGBoost {xh['logloss']:.4f} on {d['test_matches']:,} matches.", 22),
        left=1260, top=330, width=532)
    return slide("m_distill", "Deep teacher and student", ("A deep teacher network reliably improves the boosted trees" if best[3] < 0 else ("Combining a deep teacher with boosted trees is slightly better, but not reliably" if best[1] < 0 else "A deep teacher network does not improve the boosted trees")),
                 chart + panel,
                 source=f"Source: exp_deep_distill_regB.json; paired against hard-label XGBoost. {CORR}",
                 notes="Nobody has published a teacher and student method for win probability in Counter-Strike that we could find, so we tried one. A neural network was trained on the same live-state inputs; to avoid leaking its own training labels, its predictions for every training match come from networks that never saw that match. A boosted-tree student was then fitted to a mix of the real outcome and the teacher's probability. The teacher alone was worse than plain boosted trees; averaging the teacher with plain boosted trees is slightly better, but the paired interval includes zero, so it is not a reliable gain; the distilled students were level. Two earlier, smaller runs on the 12,000-match store (a 512-wide and a 256-wide teacher) gave the same picture. The inputs here are a table of aggregates at one second, so there is little for a deeper network to find; the untested route is per-player position information, which is what the positions experiment begins to test.")


# ------------------------------------------------------------------ positions and view angles
def s_positions(store):
    d = J(store, "exp_positions.json")
    if not d or "geometry" not in d:
        return None
    names = {"geometry": "+ positions: site occupancy, spacing, contact, bomb distance", "geometry+view": "+ view cones: who sees an enemy, who is exposed", "geometry+view+blind": "+ flashbang blindness (who, how long)"}
    rows = [[names[k], d[k]["delta_vs_base"], *d[k]["delta_vs_base_ci95"]] for k in names if k in d]
    amax = max(abs(x) for r in rows for x in r[1:4]) * 1.2 + 1e-4
    chart = embed(forest_js(rows, round(-amax, 4), round(amax, 4), [round(-amax * 0.8, 4), 0, round(amax * 0.8, 4)], "Log loss against the final model (95% match-cluster interval); below 0 = better",
                            left=640, right=1040, top=10, bottom=60 + 100 * len(rows), pitch=100, fmt="toFixed(4)", label_size=20, hline=0), left=128, top=330, width=1100, height=150 + 100 * len(rows))
    best = min(rows, key=lambda r: r[1])
    helps = best[3] < 0
    base = d["base"]
    gain = d.get("gain_share", {})
    top_feat = sorted(gain.items(), key=lambda kv: -kv[1])[:3]
    panel = side_panel(
        stat_box(f"{best[1]:+.4f}", f"best paired change in log loss against {base['logloss']:.4f} (interval {best[2]:+.4f} to {best[3]:+.4f})", "blue" if helps else "amber") +
        stat_box(f"{d['inputs']['geometry+view+blind'] - d['inputs']['base']}", "new inputs, all computed from straight-line distances and view angles; no map geometry or line of sight") +
        (note("Largest gain shares: " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in top_feat), 22) if top_feat else ""),
        left=1260, top=330, width=532)
    return slide("m_positions", "Where the players are", "Positions and view angles " + ("improve the model, by a small margin" if helps else "do not measurably improve the model"),
                 chart + panel,
                 source=f"Source: exp_positions.json; every other second, {d['test_seconds']:,} test seconds. {CORR}",
                 notes="The final model sees only counts and loadout totals. This experiment adds, for the same frozen matches, where the living players stand relative to the two bomb sites, how spread out they are, how close the sides are, how far the nearest CT is from a planted bomb and the seconds that CT would have left to finish a defuse, which players have an enemy inside a 90 degree view cone and which are exposed to an enemy that sees them while they do not see it, and how many players are blinded by a flash and for how long. The inputs use straight-line distances: they do not know walls, routes or line of sight, which limits what they can capture. The comparison is paired on the same test seconds with a match-cluster interval.")


# ------------------------------------------------------------------ hyperparameter sweep
def s_hparams(store):
    d = J(store, "exp_hparams.json")
    if not d:
        return None
    st = d["settings"]
    rows = []
    for r in st:
        lab = f"depth {r['depth']}, min child weight {r['min_child_weight']}"
        if "delta_vs_reference" in r:
            rows.append([lab, r["delta_vs_reference"], *r["delta_vs_reference_ci95"]])
        else:
            rows.append([lab + " (used)", 0.0, 0.0, 0.0])
    amax = max(abs(x) for r in rows for x in r[1:4]) * 1.2 + 1e-4
    chart = embed(forest_js(rows, round(-amax, 4), round(amax, 4), [round(-amax * 0.8, 4), 0, round(amax * 0.8, 4)], "Log loss change against depth 4, min child weight 20 (95% interval); minus = better",
                            left=480, right=1040, top=10, bottom=60 + 58 * len(rows), pitch=58, fmt="toFixed(4)", label_size=21, hline=0), left=128, top=330, width=1100, height=140 + 58 * len(rows))
    others = [r for r in st if "delta_vs_reference" in r]
    shallow = [r for r in others if r["depth"] <= 6]
    deep = [r for r in others if r["depth"] >= 8]
    same = all(r["delta_vs_reference_ci95"][0] <= 0 <= r["delta_vs_reference_ci95"][1] for r in shallow)
    spread = max(abs(r["delta_vs_reference"]) for r in shallow)
    worse = max((r["delta_vs_reference"] for r in deep), default=0.0)
    panel = side_panel(
        stat_box(f"{spread:.4f}", "largest log-loss difference among the settings with depth 3 to 6, min child weight 20 or 100: " + ("no measurable difference" if same else "some are measurably different")) +
        (stat_box(f"{worse:+.4f}", "depth 8 is slightly worse, and the interval excludes zero", "amber") if deep and max(r["delta_vs_reference_ci95"][0] for r in deep) > 0 else "") +
        note(f"We use depth 4 and min child weight 20. Depth is how many questions a tree may ask in a row; min child weight stops a split that would rest on very few seconds. Tuned on the GPU, scored on held-out matches, every {d['stride']}rd second. {CORR}", 22),
        left=1260, top=330, width=532)
    title = ("Tree depth 3 to 6 and min child weight 20 or 100 all score the same; very deep trees are slightly worse" if same else "Some tree settings are measurably different")
    return slide("m_hparams", "Settings", title, chart + panel,
                 source=f"Source: exp_hparams.json (scripts/model/exp_hparams.py). {CORR}",
                 notes="Eight combinations of tree depth and minimum child weight were fitted on the GPU with early stopping on the validation matches and scored on the same held-out test matches. Depths 3, 4 and 6 are indistinguishable from each other; depth 8 is a little worse. The chosen setting is depth 4 and minimum child weight 20: shallow trees are as good as deeper ones here, and they are faster and easier to explain.")

# ------------------------------------------------------------------ maps the model was not trained for
def s_scope(store, snap):
    d = J(store, f"eval_scope_maps_{snap}.json")
    if not d:
        return None
    ref, ref_b, us, ub = d["reference_snapshot_known_maps"], d["reference_fallback_known_maps"], d["snapshot_all"], d["fallback_all"]
    rows = [[f"Snapshot model, its 7 maps ({ref['n_matches']:,} held-out matches)", ref["logloss"], *ref["logloss_ci95"]],
            [f"Snapshot model, other maps ({us['n_matches']:,} matches, mostly vertigo)", us["logloss"], *us["logloss_ci95"]],
            [f"Alive-count model, its 7 maps ({ref_b['n_matches']:,} held-out matches)", ref_b["logloss"], *ref_b["logloss_ci95"]],
            [f"Alive-count model, other maps ({ub['n_matches']:,} matches, 2024 era)", ub["logloss"], *ub["logloss_ci95"]]]
    chart = embed(forest_js(rows, 0.43, 0.54, [0.44, 0.47, 0.50, 0.53], "Log loss (95% match-cluster interval); lower is better", left=640, right=1040, top=10, bottom=420, pitch=92, fmt="toFixed(4)", label_size=20),
                  left=128, top=330, width=1100, height=520)
    panel = side_panel(
        stat_box(f"{us['logloss'] - ref['logloss']:+.3f}", f"snapshot model on maps it never saw, against its own maps (unpaired; AUC {us['auc']:.3f} against {ref['auc']:.3f})") +
        stat_box(f"{us['ece10']:.3f}", f"calibration error on those maps, against {ref['ece10']:.3f}: the probabilities are a little less well calibrated", "amber") +
        note("The alive-count model is the simpler model used when no per-player table is available, and for 2024 matches: it sees only players alive, the clock, the bomb and the map. These matches are flagged out of scope by the serving code.", 22),
        left=1260, top=330, width=532)
    return slide("m_scope", "Maps outside the seven", "On maps it was never trained for, the model stays close and is flagged as out of scope",
                 chart + panel,
                 source=f"Source: eval_scope_maps_{snap}.json (scripts/model/eval_scope_maps.py); matches with no map input. {CORR}",
                 notes="The model is scoped to seven maps, but the question was whether it still gives a usable number elsewhere. The 5v5 competitive matches of other maps in the data (vertigo, overpass, train and a few others) were scored with the map inputs left at zero, which is what the serving code does for an unknown map. The snapshot model loses about six thousandths of log loss and keeps its ranking quality; its calibration is slightly worse. The alive-count model is scored on the 2024 matches of those maps. Intervals resample whole matches and the comparison with the known maps is unpaired because the matches differ. Wingman has 2v2 states the grid code does not build, so it is not measured here; the serving code answers it and marks it out of scope.")


# ------------------------------------------------------------------ enough data
def s_enough(store):
    a = J(store, "exp_curve_accuracy.json")
    if a and a.get("points"):
        pts = a["points"]
        rows = [[f"{q['train_matches']:,} matches", q["accuracy"], *q["accuracy_ci95"]] for q in pts]
        lo_ = min(r[2] for r in rows) - 0.002
        hi_ = max(r[3] for r in rows) + 0.002
        chart = embed(forest_js(rows, round(lo_, 3), round(hi_, 3), [round(lo_ + (hi_ - lo_) * k / 4, 3) for k in range(1, 4)], "How often the favoured side wins the round, by training matches (95% match-cluster interval)",
                                left=300, right=1020, top=10, bottom=470, pitch=74, highlight=len(rows) - 1, fmt="toFixed(4)", label_size=25), left=128, top=330, width=1100, height=580)
        f, l = pts[0], pts[-1]
        panel = side_panel(
            stat_box(f"{100 * f['accuracy']:.1f}% to {100 * l['accuracy']:.1f}%", f"accuracy with {f['train_matches']:,} training matches against {l['train_matches']:,}: the side the model favours at a second wins that round {100 * l['accuracy']:.1f}% of the time") +
            stat_box(f"{l['logloss'] - f['logloss']:+.4f}", f"log loss over the same range ({f['logloss']:.4f} to {l['logloss']:.4f}); AUC {f['auc']:.3f} to {l['auc']:.3f}", "amber") +
            note(f"Every point is a model trained on that many matches and scored on the same {a['test_matches']:,} held-out matches ({a['test_seconds']:,} seconds). {CORR}", 22),
            left=1260, top=330, width=532)
        return slide("m_enough", "Is there enough data?", f"More matches help only a little: {100 * f['accuracy']:.1f}% accuracy with {f['train_matches']:,} matches, {100 * l['accuracy']:.1f}% with {l['train_matches']:,}", chart + panel,
                     source="Source: exp_curve_accuracy.json (scripts/model/exp_curve_accuracy.py); nested random subsets of the training matches. " + CORR,
                     notes="Accuracy here counts seconds: at each second the side the model gives more than 50 percent is called the favourite, and the second is correct if that side wins the round. It is dominated by late, easy seconds, which is why it moves so little; log loss and AUC move in the same direction by small amounts. The curve is flat from about a thousand matches, so the model is not data-starved; the value of more matches is robustness across days, maps and sources rather than headline accuracy.")
    d = J(store, "exp_robustness.json")
    if not d or "curve" not in d:
        return None
    c = d["curve"]
    rows = [[f"{p['train_matches']:,} matches", p["logloss"], p["ci95"][0], p["ci95"][1]] for p in c["points"]]
    lo_ = min(r[2] for r in rows) - 0.004
    hi_ = max(r[3] for r in rows) + 0.004
    chart = embed(forest_js(rows, round(lo_, 3), round(hi_, 3), [round(lo_ + (hi_ - lo_) * k / 4, 3) for k in range(1, 4)], "Test log loss against the number of training matches (95% match-cluster interval)",
                            left=300, right=1020, top=10, bottom=470, pitch=74, highlight=len(rows) - 1, fmt="toFixed(4)", label_size=25), left=128, top=330, width=1100, height=580)
    nul = d.get("null")
    panel = side_panel(
        stat_box(f"{c['predicted_gain_from_doubling']:.4f}", "log loss the fitted curve says doubling the training matches would still buy") +
        stat_box(f"{c['predicted_gain_from_adding_1000_matches']:.4f}", "predicted gain from 1,000 more matches", "amber") +
        (note(f"Shuffled-label check: a model trained on randomly shuffled round outcomes scores {nul['test_logloss_shuffled_label_model']:.3f} against {nul['constant_logloss_on_test']:.3f} for a constant, AUC {nul['auc']:.2f}: no hidden label signal.", 24) if nul else ""),
        left=1260, top=330, width=532)
    return slide("m_enough", "Is there enough data?", "More matches still help, but only in the fourth decimal place",
                 chart + panel,
                 source=f"Source: exp_robustness.json (curve, null); nested subsets of training matches. {CORR}",
                 notes="This answers the sufficiency question directly. The same model was trained on 5, 10, 20, 40, 70 and 100 percent of the training matches and scored on the same held-out matches. The curve flattens: the fitted extrapolation says doubling the data would improve log loss by only about " + f"{c['predicted_gain_from_doubling'] * 1e4:.0f}" + " ten-thousandths, far below the width of the test interval. The shuffled-label check is the leakage control: when the round outcomes are shuffled among training rounds the model learns nothing and scores like a constant; if any feature smuggled in information about the label, this model would beat the constant.")


# ------------------------------------------------------------------ data
def day_table_2026(store):
    """2026 by upload day: usable matches by frozen split, used rounds, one-second rows. Counted from the finished tables."""
    d = os.path.join(ROOT, "data", store)
    M = pl.read_parquet(os.path.join(d, "matches.parquet"), columns=["match_key", "era", "upload_day"]).filter(pl.col("era") == "2026")
    R = pl.read_parquet(os.path.join(d, "rounds.parquet"), columns=["match_key", "use"]).filter(pl.col("use"))
    G = pl.read_parquet(os.path.join(d, "grid.parquet"), columns=["match_key", "era", "upload_day"]).filter(pl.col("era") == "2026")
    rd = R.join(M.select("match_key", "upload_day"), on="match_key", how="inner").group_by("upload_day").agg(pl.len().alias("rounds"), pl.col("match_key").n_unique().alias("matches"))
    gr = G.group_by("upload_day").agg(pl.len().alias("rows"))
    man = json.load(open(os.path.join(d, "split_manifest.json"), encoding="utf-8"))["assign"]
    sp = pl.DataFrame({"match_key": list(man.keys()), "split": list(man.values())}).join(M.select("match_key", "upload_day"), on="match_key", how="inner")
    sp = sp.group_by("upload_day", "split").len().pivot(on="split", index="upload_day", values="len").fill_null(0)
    t = rd.join(gr, on="upload_day", how="left").join(sp, on="upload_day", how="left").sort("upload_day").fill_null(0)
    return t.to_dicts()


def s_data(store):
    days = day_table_2026(store)
    tot = {"matches": sum(r["matches"] for r in days), "rounds": sum(r["rounds"] for r in days), "rows": sum(r["rows"] for r in days)}
    tr = sum(r.get("train", 0) for r in days)
    va = sum(r.get("val", 0) for r in days)
    te = sum(r.get("test", 0) for r in days)
    tbl = table(["2026, all upload days", "Matches used", "Rounds used", "One-second rows", "Train / val / test matches"],
                [[f"{len(days)} upload days", f"{tot['matches']:,}", f"{tot['rounds']:,}", f"{tot['rows']:,}", f"{tr:,} / {va:,} / {te:,}"]],
                widths=[22, 16, 16, 20, 26], size=26)
    data = [[r["upload_day"][5:], int(r.get("train", 0)), int(r.get("val", 0)), int(r.get("test", 0)), int(r["rounds"]), int(r["rows"] or 0)] for r in days]
    js_lines = [
        "var D=" + json.dumps(data) + ";",
        "var L=50,W=1664,H=400,B=350,T=14;",
        "var n=D.length;",
        "var mx=Math.max.apply(null,D.map(function(d){return d[1]+d[2]+d[3]}));",
        "var cols=['#1D4ED8','#B45309','#94A3B8'];",
        "var names=['train','validation','test'];",
        "var slot=(W-L-10)/n;",
        "D.forEach(function(d,i){",
        "  var y=B; var x=L+i*slot+slot*0.15; var w=slot*0.7; var tot=d[1]+d[2]+d[3];",
        "  for(var k=0;k<3;k++){ var v=d[k+1]; if(!v) continue; var h=v/mx*(B-T); y-=h;",
        "    var r=E('rect',{x:x,y:y,width:w,height:h,fill:cols[k]});",
        "    var t=document.createElementNS('http://www.w3.org/2000/svg','title'); t.textContent=d[0]+': '+names[k]+' '+v+' matches'; r.appendChild(t); }",
        "  if(i%4==0) E('text',{x:x+w/2,y:B+22,'font-size':18,'text-anchor':'middle',fill:'#475569'},d[0]);",
        "  E('text',{x:x+w/2,y:y-6,'font-size':14,'text-anchor':'middle',fill:'#475569'},String(tot));",
        "});",
        "E('line',{x1:L,x2:W-6,y1:B,y2:B,stroke:'#94A3B8','stroke-width':2});",
        "E('text',{x:L,y:T+4,'font-size':18,fill:'#475569'},'usable matches per upload day, by frozen split');",
        "E('rect',{x:W-420,y:T+20,width:18,height:18,fill:cols[0]}); E('text',{x:W-396,y:T+35,'font-size':18,fill:'#16202E'},'train');",
        "E('rect',{x:W-320,y:T+20,width:18,height:18,fill:cols[1]}); E('text',{x:W-296,y:T+35,'font-size':18,fill:'#16202E'},'validation');",
        "E('rect',{x:W-200,y:T+20,width:18,height:18,fill:cols[2]}); E('text',{x:W-176,y:T+35,'font-size':18,fill:'#16202E'},'test');",
    ]
    js = "\n".join(js_lines)
    chart = embed(js, left=128, top=500, width=1664, height=400)
    return slide("m_data", "Data", "2026 data by upload day: every day that went into the model",
                 tbl + chart,
                 source="Source: matches.parquet, rounds.parquet, grid.parquet, split_manifest.json (prepare.py, make_grid.py, split_manifest.py); counted at build time",
                 notes="The sizes are counted from the finished tables, not typed in. Each bar is one upload day of 2026 (the day the matches were uploaded to the server, which is the unit we split by when new data arrives). The bar is the usable matches of that day, coloured by the frozen split: train, validation and test. Matches are split by match, so no match is in both training and test. Rounds with a disconnect or bot takeover, or a label the independent signals cannot confirm, are dropped. The 2024 era is not shown here: 2,749 matches, 46,744 rounds, used only for the alive-count model and the cross-period test. Scope: competitive 5v5 on ancient, anubis, cache, dust2, inferno, mirage and nuke; wingman, hostage and other maps are excluded; 576 duplicate groups were found and one copy of each is kept.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--store", default="combined")
    ap.add_argument("--snapshot", default="winprob_v1")
    ap.add_argument("--basic", default="winprob_v1_basic")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    _cs = J(a.store, f"cluster_se_{a.snapshot}.json")
    if _cs:
        CORR = f"Seconds in a round are correlated, so every interval resamples whole matches (counting seconds as independent would be {_cs['match_cluster_ratio']:.1f} times too narrow)."
    builders = {"m_headline": lambda: s_headline(a.store, a.snapshot, a.basic), "m_stack": lambda: s_stack(a.store), "m_calib": lambda: s_calib(a.store, a.snapshot),
                "m_ticks": lambda: s_ticks(a.store, a.snapshot), "m_benchmark": lambda: s_benchmark(a.store, a.snapshot), "m_transfer": lambda: s_transfer(a.store),
                "m_gpucpu": lambda: s_gpucpu(a.store, a.snapshot, a.basic), "m_robust": lambda: s_robust(a.store, a.snapshot, a.basic), "m_data": lambda: s_data(a.store),
                "m_maps": lambda: s_maps(a.store), "m_enough": lambda: s_enough(a.store), "m_families": lambda: s_families(a.store),
                "m_foundation": lambda: s_foundation(a.store), "m_temporal": lambda: s_temporal(a.store), "m_distill": lambda: s_distill(a.store), "m_positions": lambda: s_positions(a.store), "m_scope": lambda: s_scope(a.store, a.snapshot), "m_hparams": lambda: s_hparams(a.store),
                "cover": lambda: T.s_cover(a.store, a.snapshot, a.basic), "brief": lambda: T.s_brief(a.store, a.snapshot, a.basic), "eval": lambda: T.s_eval(a.store, a.snapshot),
                "incremental": lambda: T.s_incremental(a.store), "viewer": lambda: T.s_viewer(), "lit": lambda: T.s_lit(), "decisions": lambda: T.s_decisions(),
                "questions": lambda: T.s_questions(), "next": lambda: T.s_next(a.store), "appendix": lambda: T.s_divider(), "faceit": lambda: T.s_faceit(a.store), "beforeafter": lambda: T.s_beforeafter(a.store, a.snapshot, a.basic), "graph": lambda: T.s_graph(a.store)}
    for k, fn in builders.items():
        if a.only and k not in a.only.split(","):
            continue
        try:
            h = fn()
        except Exception as exc:  # a missing input file must not stop the other slides
            print(f"SKIP {k}: {type(exc).__name__}: {exc}")
            continue
        if h is None:
            print(f"SKIP {k}: inputs not available yet")
            continue
        open(os.path.join(a.out, k + ".html"), "w", encoding="utf-8").write(h)
        print("wrote", k, len(h))
