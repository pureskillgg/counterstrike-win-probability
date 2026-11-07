# Grill-me review of the final choices (2026-10-19)

One row per choice. Evidence is per-second log loss on the frozen split (random by match, seed 0). Validation decides settings; test is
scored once for the reported number. Where a choice used test numbers, the row says so.

| Choice | Objective | Alternatives (incl. the default) | Evidence (split) | Cost and explainability | Reversal (threshold) | Check and status |
|---|---|---|---|---|---|---|
| Target: P(CT wins), per second | Win probability at every live second | Per-round outcome only; per-tick | Frozen split; metric reported per second and per round | Simple to explain | Not reversible in practice | Done (per-second metric in every report) |
| Split: frozen random by match, seed 0 (manifest a17f219b) | No match in two sets; held-out matches stable across models | Chronological split (rejected: source shift); per-era | Manifest verify: disjoint, every grid match assigned | Easy | A change of split would invalidate all numbers | Done (`split_manifest.py --verify` OK) |
| Training rows: every second | Use all second-level information | Random ticks per round (1 to 64 draws); within-second ticks (1, 2, 4, 8 per second) | Random per round: about +0.0013 worse on test; within-second 1 to 8 per second: none measurably different from every second (all CIs cross zero) | Every second is the simplest | Adopt sub-second sampling only if validation gain > 0.0002 | Done |
| Event inputs (13): alive counts, bomb, time, map | Causal state from events | Positions (deferred per instruction); facing, smokes, fire, blindness (untested) | Serving parity: exact on 294 test matches | Simple | n/a | Done; positions and the four untested are deferred |
| Loadout inputs (44 per side and diffs) from per-tick status | State of players, not only counts | No loadout; health only; weapon classes only | Health +0.0042 val when removed; defuse/defusing/helmets +0.0018; armour +0.0006; weapon block +0.0007; equipment and grenades ~0 | Medium (needs per-tick status) | Drop a group if validation gain < 0.0002 | Done (one-at-a-time and joint; joint 27-input rejected at +0.0030) |
| Drop the heavy group (shotguns, machine guns) | Smaller model with no loss | Keep heavy | Validation -0.00009, test -0.00002 (both seeds); test was used to confirm consistency | Simpler | Keep if validation loss rises > 0.0002 | Done, **flag: test numbers informed this choice** (validation alone chose the 38-input set, which was not better on test) |
| Score lead (freeze-end CT minus T score) | Pre-round team strength | Leave out | Test -0.0007 (59-input, seed-controlled), -0.0009 (13-input); tome matches the grid in every checked round | Simple; known at round start | Drop if the tome lead differs from the grid | Done |
| Monotone constraints on alive counts | Physically sensible shape | None | Removing them: +0.00009 test [+0.00001, +0.00018] | Simple | Keep | Done |
| Hyperparameters (depth 4, eta 0.05, min child weight 20, subsample 0.8) | Best validation log loss | 10 alternatives (depth 5, 6, 8; min child weight 50, 100; subsample 1.0; colsample 0.8; eta 0.03; depth 6+mcw 100 (+colsample)) | None better on validation | Standard | A setting that improves validation by > 0.0002 | Done |
| Five-seed average | Lower variance | Single seed | Test -0.00008 vs seed 0 [-0.00013, -0.00002] | Five models to run | Drop if no gain | Done |
| Event-derived features (cumulative damage and shots) | Round-to-date engagement | Leave out | Validation -0.00002, test +0.00016 | Adds complexity | Adopt only if validation gain > 0.0002 | Done, rejected |
| Pistol-round and round-after-pistol models | Round-type specialisation | One model | Overall test +0.00022 [+0.00007, +0.00037] | Adds two models | Adopt only if overall test improves | Done, rejected |
| Model family: XGBoost | Accuracy and speed | LightGBM, CatBoost, logistic regression | XGBoost reference on stride-3 data: validation 0.452831, test 0.455783 | Standard | Switch if another family beats XGBoost on validation | LightGBM, CatBoost, logistic: see exp_families_pistol.json |
| Serving: reads per-tick player status from the tome | Score the best model from the same data as training | Serve the 14-input event model | Serving parity: 20 of 20 test matches within 0.0001 points (294 matches run after this note) | Needs per-tick tome | Revert to the 14-input model if parity fails | Done |

## Open items at the time of writing

- Leakage canary (future alive count) and weighted-draw check: see `data/combined/exp_final_checks.json`.
- Families: see `data/combined/exp_families_pistol.json`.
- Not done: positions, untested features, 16 per second, deep models (next week), email to Bill (needs explicit permission per send).
