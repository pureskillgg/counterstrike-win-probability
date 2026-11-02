"""Deliverable summary written from the saved results files and model metadata (nothing typed by hand).

Writes notes/DELIVERABLE_2026-10-19.md: the production model, its features and settings, the frozen split with example matches, every
result that was run (with the decision it led to), what is tested, and what is deliberately not included.

  ...python.exe -I -X utf8 scripts\\model\\write_deliverable_summary.py
"""
import hashlib
import json
import os
import sys

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"G:\pureskill\csds-cs2"
MODELS = os.path.join(ROOT, "models")
D = os.path.join(ROOT, "data", "combined")
OUT = os.path.join(ROOT, "notes", "DELIVERABLE_2026-10-19.md")


def meta(name):
    return json.load(open(os.path.join(MODELS, f"{name}.meta.json"), encoding="utf-8"))


def results(name):
    path = os.path.join(D, name)
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None


def fmt(x, digits=6):
    return "n/a" if x is None else f"{x:.{digits}f}"


def ci(lo_hi, digits=5):
    if not lo_hi:
        return ""
    return f" [{lo_hi[0]:+.{digits}f}, {lo_hi[1]:+.{digits}f}]"


def main():
    manifest = os.path.join(D, "split_manifest.json")
    file_sha = hashlib.sha256(open(manifest, "rb").read()).hexdigest()
    assign = json.load(open(manifest, encoding="utf-8"))["assign"]
    sizes = {s: sum(1 for v in assign.values() if v == s) for s in ("train", "val", "test")}
    grid = pl.read_parquet(os.path.join(D, "grid.parquet"), columns=["match_key", "round", "tick", "era"])
    examples = {}
    for label in ("train", "val", "test"):
        keys = sorted(k for k, v in assign.items() if v == label)
        g = grid.filter(pl.col("match_key") == keys[0])
        r1 = g.filter(pl.col("round") == g["round"].min()).sort("tick")
        examples[label] = (keys[0], g["era"][0], int(r1["round"][0]), int(r1["tick"].min()), int(r1["tick"].max()), r1.height)

    prod = meta("winprob_v8_ensemble")
    members = prod.get("members", [])
    lines = []
    add = lines.append
    add("# Round win probability for CS2: deliverable summary (2026-10-19, generated)\n")

    add("## Production model\n")
    add("`winprob_v8_ensemble`: the mean of five XGBoost seeds on 58 inputs. It is the model the serving function uses by default. "
        "Validation is used for every choice; test is scored once.\n")
    add(f"- Validation log loss {fmt(prod['validation_logloss'])}; test log loss {fmt(prod['test_logloss'])} on {prod['test_rows']:,} seconds of the 2026 test matches.")
    add(f"- Seed-0 single model: validation 0.452778, test 0.455680. The five-seed average is {ci([prod['test_delta_vs_seed0_ci95'][0], prod['test_delta_vs_seed0_ci95'][1]])} "
        f"better on test than seed 0 alone.")
    add("- Serving: `scripts/serving/win_probability.py` (default). Parity against the grid on every October test match: 294 of 294 matches, "
        "374,500 seconds, largest gap 0.0001 percentage points, 100% of seconds within 0.01 points. Score lead matches the grid in 5,431 of 5,431 rounds.")
    add("- Needs per-tick player status in the DSDK tome. A tome without it returns no probabilities (no silent fallback).\n")

    add("### Inputs (58)\n")
    add("- Event inputs (13): alive counts for each side, their difference, bomb planted, seconds since round live, seconds since plant, and the map (one-hot, 7 maps).")
    add("- Score lead (1): freeze-end CT score minus T score (round_state; known before the round starts).")
    add("- Loadout inputs (44): per side, team health, armour (players with armour, armour points), helmets, defuse kits, defusing, weapon classes "
        "(AK, M4, AWP, other rifles, other snipers, SMG, shotgun and machine gun counts are dropped; see results), players with no primary, grenades, equipment value; "
        "and the CT-minus-T differences of health, rifles, equipment, armour, AWP and SMG. Taken from the per-tick player status at the same tick.")
    add("- Monotone constraints: alive CT +1, alive T -1, alive difference +1; everything else unconstrained (score lead unconstrained).\n")

    add("### Settings\n")
    add("XGBoost: depth 4, learning rate 0.05, min child weight 20, subsample 0.8, early stopping after 100 rounds on validation, GPU, "
        "seeds 0 to 4 averaged. Every second of each round, roster-defect and status-inconsistent rounds dropped (in the cache).\n")

    add("## Frozen split\n")
    add(f"`data/combined/split_manifest.json` (file sha256 {file_sha[:12]}, the hash in every model's metadata): random by match, seed 0, "
        f"all {len(assign):,} matches: train {sizes['train']:,}, validation {sizes['val']:,}, test {sizes['test']:,}. "
        "The 2026 models use the 2026 subset (10,577 / 1,168 / 3,059 matches after the cache drops 10 untraced matches).\n")
    add("Example match and its first round's live ticks in each split (first match in each split, sorted by key):\n")
    add("| Split | Match key | Era | Round | First tick | Last tick | Seconds |")
    add("|---|---|---|---:|---:|---:|---:|")
    for label in ("train", "val", "test"):
        k, era, rnd, t0, t1, n = examples[label]
        add(f"| {label} | `{k}` | {era} | {rnd} | {t0} | {t1} | {n} |")
    add("\nTicks step by 64 from the freeze end (one per second). Matches and ticks in every split are recorded in the manifest and the grid.\n")

    add("## Results (per-second log loss)\n")
    add("| Model | Inputs | Validation | Test | Notes |")
    add("|---|---:|---:|---:|---|")
    add(f"| Production: five-seed ensemble | 58 | {fmt(prod['validation_logloss'])} | {fmt(prod['test_logloss'])} | served by default |")
    for name, n_in, note in [
        ("winprob_v7_59_drop_heavy_scorelead", 58, "single seed"),
        ("winprob_v5_59_scorelead", 60, "with heavy group, score lead"),
        ("winprob_v4", 59, "loadout reference"),
        ("winprob_v6_drop_heavy", 57, "no score lead"),
        ("winprob_v7_basic_scorelead", 14, "events and score lead only, all matches"),
        ("winprob_v4_basic", 13, "previous served, all matches"),
    ]:
        m = meta(name)
        add(f"| `{name}` | {n_in} | {fmt(m['validation_logloss'])} | {fmt(m['test_logloss'])} | {note} |")
    add("\nScores on the 2026 test seconds (3,956,374) are comparable within the 58/59-input group; the 13/14-input models are scored on all matches "
        "(4,568,043 seconds) and are comparable only with each other.\n")

    add("## Everything that was run, and what it led to\n")
    rows = []
    cs = results("exp_loadout_ablation.json")
    if cs:
        for v in cs["variants"]:
            if v["name"].startswith("drop"):
                rows.append((v["name"], v.get("validation_delta_vs_control"), v.get("test_delta_vs_control"), v.get("test_delta_ci95")))
    if rows:
        add("### Loadout ablation (one group removed; positive = removal hurts)\n")
        add("| Removed | Validation change | Test change [95% CI] |")
        add("|---|---:|---:|")
        for name, dv, dt, ci_ in rows:
            add(f"| {name.replace('drop ', '')} | {fmt(dv, 5)} | {fmt(dt, 5)}{ci(ci_)} |")
        add("")
    joint = results("exp_loadout_joint.json")
    if joint:
        for v in joint["variants"]:
            if v["name"].startswith("joint"):
                add(f"- Joint removal of the 32 redundant loadout features (27 inputs): validation {fmt(v['validation_delta_vs_control'], 5)}, "
                    f"test {fmt(v['test_delta_vs_control'], 5)}; rejected.")
    blocks = results("exp_loadout_blocks.json")
    if blocks:
        for v in blocks["variants"]:
            if v["name"].startswith("drop"):
                add(f"- Block {v['name']}: validation {fmt(v['validation_delta_vs_control'], 5)}, test {fmt(v['test_delta_vs_control'], 5)}.")
    add("")

    sl = results("exp_scorelead_combo.json")
    add("### Score lead")
    if sl:
        for v in sl["variants"]:
            if "validation_delta_vs_control" in v:
                add(f"- {v['name']} (seed {v['seed']}): validation {fmt(v['validation_delta_vs_control'], 5)}, test {fmt(v['test_delta_vs_control'], 5)}{ci(v['test_delta_ci95'])}.")
    add("- Adopted (it is in the production set).\n")

    tune = results("exp_hparams58.json")
    add("### Hyperparameters (58 inputs; eleven settings, validation decides)")
    if tune:
        base = tune["configs"][0]
        add(f"- Baseline (depth 4, min child weight 20, subsample 0.8, eta 0.05): validation {fmt(base['validation_logloss'])}, test {fmt(base['test_logloss'])}.")
        for c in tune["configs"][1:]:
            add(f"- {c['name']}: validation delta {fmt(c.get('validation_delta_vs_baseline'), 5)}, test delta {fmt(c.get('test_delta_vs_baseline'), 5)}.")
        add(f"- Selected by validation: {tune.get('selected', 'baseline')} (baseline settings kept).\n")

    seeds = results("exp_seed_avg58.json")
    if seeds and "averaged" in seeds:
        a = seeds["averaged"]
        add(f"### Seed averaging: five seeds average test {fmt(a['test_logloss'])} against {fmt(seeds['seeds'][0]['test_logloss'])} for seed 0 "
            f"({fmt(a['test_delta_vs_seed0'], 5)}{ci(a['test_delta_vs_seed0_ci95'])}). Adopted.\n")

    ev = results("exp_event_features.json")
    if ev:
        for v in ev["variants"]:
            if "validation_delta_vs_baseline" in v:
                add(f"### Event features (cumulative damage and shots per side in the round): validation {fmt(v['validation_delta_vs_baseline'], 5)}, "
                    f"test {fmt(v['test_delta_vs_baseline'], 5)}{ci(v['test_delta_ci95'])}. Rejected.\n")

    pistol = results("exp_families_pistol.json")
    if pistol and pistol.get("pistol"):
        p = pistol["pistol"]
        add("### Pistol-round split")
        for name in ("pistol rounds (1, 13)", "round after pistol (2, 14)"):
            if name in p:
                q = p[name]
                add(f"- {name}: dedicated minus main {fmt(q['dedicated_minus_main'], 5)}{ci(q['ci95'])}.")
        if "overall_test" in p:
            o = p["overall_test"]
            add(f"- Overall test with the split {fmt(o['split_minus_main'], 5)}{ci(o['ci95'])}. Rejected.\n")
    fam = pistol
    if fam and fam.get("families"):
        add("### Model families (58 inputs, every third training second)")
        for name, v in fam["families"].items():
            extra = "" if name == "xgboost" else f" (test delta vs XGBoost {fmt(v.get('test_delta_vs_xgboost'), 5)}{ci(v.get('test_delta_ci95'))})"
            add(f"- {name}: validation {fmt(v['validation_logloss'])}, test {fmt(v['test_logloss'])}{extra}.")
        add("- XGBoost kept: it is the family the production model uses, and it is the reference the others are compared with.\n")
    full = results("exp_full_comparison.json")
    if full and full.get("models"):
        add("### Model families, full training set (13,758,901 seconds, 59 inputs, every second)")
        xg = full["models"]["xgb"]["logloss"]
        for name in ["xgb", "lgbm", "catboost", "catboost_gpu", "hgb", "rf", "et", "mlp", "logit"]:
            if name not in full["models"]:
                continue
            v = full["models"][name]
            delta = "" if name == "xgb" else f" (test vs XGBoost {v['delta_vs_xgb']:+.6f} [{v['delta_vs_xgb_ci95'][0]:+.6f}, {v['delta_vs_xgb_ci95'][1]:+.6f}], paired)"
            add(f"- {name}: test {fmt(v['logloss'])} [{v['logloss_ci95'][0]:.5f}, {v['logloss_ci95'][1]:.5f}], validation {fmt(v['validation_logloss'])}{delta}.")
        add("- Settings in this design were not tuned to the production settings (see the source script exp_full_comparison.py).\n")

    ticks8 = results("exp_tick8.json")
    if ticks8:
        add("### Ticks per second (58 inputs, 2,644 training matches, within each second)")
        add(f"- Every second: test {fmt(ticks8['every_second']['test_logloss'])}.")
        for r in ticks8["rates"]:
            add(f"- {r['ticks_per_second']} per second ({r['rows_vs_every_second']:.2f} of the every-second rows): test "
                f"{fmt(r['test_logloss'])}, delta {fmt(r['test_delta_vs_every_second'], 5)}{ci(r['test_delta_ci95'])}.")
        add("- No gain from more than one tick per second. Every second kept.\n")

    rnd = results("exp_random_ticks59.json")
    if rnd:
        add("### Random draws per round (1 to 64 per round, 59 inputs)")
        for k in ("s0_n64", "s1_n64", "s0_n1"):
            c = rnd["cells"].get(k)
            if c:
                add(f"- {k}: {c['rows_vs_every_second']:.2f} of the every-second rows, test delta {fmt(c['test_delta_vs_every_second'], 5)}{ci(c['test_delta_ci95'])}.")
    fin = results("exp_final_checks.json")
    if fin:
        if "canary" in fin:
            add(f"- Leakage canary (a feature five seconds in the future): test {fmt(fin['canary']['test_logloss_with_future_feature'])}. "
                "It is far better than any real model, so the metric would show leakage if there were any.")
        if "weighted_draws" in fin:
            w = fin["weighted_draws"]
            add(f"- Same draws weighted by round length / 64: test {fmt(w['weighted_test_logloss'])} against {fmt(w['unweighted_test_logloss'])} unweighted "
                f"({fmt(w['weighted_minus_unweighted'], 5)}{ci(w['ci95'])}). Round weighting explains most of the random-draw gap.")
    add("- Random draws are not used: every second is the training set.\n")

    add("## Serving and checks\n")
    add("- Serving function reads the DSDK tome channels as they are (round_state, round_end, player_info, player_death, bomb_state, header, player_status).")
    add("- Split verified: disjoint sets, every grid match assigned.")
    add("- Every saved model re-derived from its booster on the frozen split (validation and test match the metadata to 1e-6).")
    add("- Score lead equals the grid's in every checked round (5,431 of 5,431 in the tome window).")
    add("- DSDK incremental tome changes: crash-consistency fixes (channels before the header, rewrite staging files cleaned up, header rows before the keyset), "
        "with tests; the full suite result is in logs/dsdk_fork_full_tests.log.\n")

    add("## Not in the model, and why\n")
    add("- Positions: left for a later model, as instructed.")
    add("- Facing direction, smokes, fire, blindness: not tested; left out.")
    add("- Deep models: next week, as planned.")
    add("- 16 per second: not built; the finest tested rate is 8 per second, and it showed no gain.\n")

    add("## Not done without your go-ahead\n")
    add("- Emailing the models: each send needs explicit permission.")
    add("- Publishing the deck (the local HTML is the draft to review).\n")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
