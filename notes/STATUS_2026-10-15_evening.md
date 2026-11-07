# Status, 2026-10-15 evening (supersedes the open items of STATUS_2026-10-15.md)

## Done and verified
| item | result | where |
|---|---|---|
| Day 13 merged (release B) | 20,754 matches in the store; 17,563 usable (2024: 2,749; 2026: 14,814); 329,028 rounds; append 428 s, prepare 40 s, grid 35 s, split 20 s, loadout extraction 626 s, cache 486 s (12 chunks) | `logs/release_b.log`, `notes/TOME_INCREMENTAL_BUILD_DESIGN.md` (Fork implemented) |
| Final snapshot model `winprob_v3` | log loss **0.4565** [0.4534, 0.4592] on 3,059 held-out matches (3,956,374 s), AUC 0.8578, ECE10 0.0034; trained on 10,577 matches (6.93 M rows at stride 2), val 1,168 | `data/combined/export_winprob_v3.json` |
| Fallback `winprob_v3_basic` | 0.5029 [0.5005, 0.5054] on 3,611 matches | `export_winprob_v3_basic.json` |
| GPU vs CPU training | delta -0.00002 [-0.0001, 0.00007]; isotonic +0.00012, Platt +0.00001 (neither helps) | same file |
| Baselines (2026 part of the test split, 3,061 matches) | alive counts only 0.4999; earlier score-lead design 0.4761; constant 0.6929 | `baselines.json` |
| Newest uploads (cutoff 2026/10/11, 644 frozen-test matches after it) | reference 0.4531, temporal 0.4533, +0.0003 [0.0001, 0.0005] | `exp_temporal.json` |
| Learning curve | 529 -> 0.4604 ... 10,577 -> 0.4568; doubling would gain 0.0003 | `exp_robustness.json` |
| Map input | with 0.4565, none 0.4568 (+0.0003 [0.0001, 0.0005]); the final model keeps the seven map inputs | `exp_robustness.json` |
| Model families (same rows, stride 6) | XGBoost 0.4567; random forest +0.0021; extra trees +0.0028; sklearn MLP +0.0015; ridge +0.0109; xgb-rf mode +0.0253 | `exp_model_families.json` |
| Deep teacher + distillation | teacher alone worse than XGBoost, distilled students level at best | `exp_deep_distill_regB.json` |
| GPU hyperparameter sweep (depth x min child weight) | every setting within 0.0003 of the final one (depth 4, mcw 20); none distinguishable except depth 8 slightly worse | `exp_hparams.json` |
| Unseen maps | snapshot model on 246 matches of vertigo/overpass/others: 0.4629 [0.4523, 0.4737], AUC 0.854, ECE 0.023 (known maps 0.4565, 0.858, 0.0034); fallback on 1,007 matches (2024): 0.5191 | `eval_scope_maps_winprob_v3.json` |
| Serving contract | stress 35 checks, 0 failed, 1 warning (T-side kit / defusing inputs never split, expected); parity 8 of 8 (max diff 3.3e-7); e2e 62 PASS, 7 WARN (known limits) | `stress_winprob_v3.json`, `parity_winprob_v3.json`, `e2e_winprob_v3.json` |
| Mutation test of the checks | 24 of 24 faults caught | `mutation_report.json` |
| Replay viewer | two held-out matches (dust2 `711a422b`, mirage `ac0d9a57`): 8 Hz positions, view cones, smokes, fires, grenades in flight, hit flashes, blindness, weapon class, gear chips, win chance | `notes/replay/*.html` (rebuild with `scripts/replay/extract_replay.py` + `build_replay_page.py`) |
| Deck | Slides artifact Q3wq55RwipnGNHgm9UZGu6 rebuilt around v3 (first-batch slides kept as appendix) | `scripts/viz/build_deck_model.py`, `assemble_deck.py` |
| Status page | artifact FVGcKB2DMERbXpcVmpCVBh refreshed | `scripts/viz/build_status_page.py` |

## What the model does NOT see (asked about repeatedly)
Positions, rotations, view angles, smokes, grenades in flight, blindness, who damaged whom, weapon-range interactions. It reads one second of aggregate state; no history, no previous probability.
Wingman (2v2) states and maps outside the seven are answered and flagged out_of_scope; accuracy on other maps measured above; wingman accuracy NOT measured (the grid code builds 5v5 only).

## Pod
RunPod pod `3vk4kqryhklbyu` (Community, RTX A4000, $0.17/h) ran the v3 chain, the hyperparameter sweep and the positions experiment. Outputs copied out and sha256-verified (`G:\pureskill\runpod_out\v3_final`). Terminated 17:08 local after the positions result was pulled; `--list` showed 0 pods. About 1.5 h, about $0.26.

## Positions, view angles, blindness (experiment: `data/combined/exp_positions.json`)
Extraction `scripts/data/extract_positions_1hz.py` (14,804 matches, 0 failed, 3,039 s) and `extract_blind_events.py`; features `make_pos_features.py` (per-day incremental); test `exp_positions.py` on the pod, same frozen split, every other second, 3,059 test matches (1,992,866 s), paired:
| inputs | log loss | change vs base [95% match-cluster interval] |
|---|---|---|
| base (the 59 final inputs) | 0.4576 | |
| + positions: site occupancy, spacing, contact, bomb distance, defuse slack | 0.4547 | -0.0030 [-0.0033, -0.0027] |
| + view cones (sees an enemy / exposed) | 0.4544 | -0.0032 [-0.0035, -0.0029] |
| + blindness | 0.4543 | -0.0033 [-0.0036, -0.0030] |
Largest gain shares: defuse_slack 2.6%, c_min_bomb_dist 1.3%, t_min_bomb_dist 0.6%. Straight-line distances only (no walls, no line of sight). NOT in the served model: serving needs player positions in the input contract (winprob.py, snapshot_features.py) and a parity run.

## Open
* Wingman accuracy; remaining half-synced day-13 folders (none seen on E: at 16:00: 2,797 folders = the 2,797 merged); Zotero filing of the papers (list only, no write access); radar image licence (Valve assets) before sharing the viewer.
