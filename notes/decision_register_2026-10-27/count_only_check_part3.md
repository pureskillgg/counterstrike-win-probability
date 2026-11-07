# Count-only check, part 3 (rows 33 to 47 of the 47 COUNT-ONLY files)

_Saved by the parent from the finished subagent result (27 Oct 2026). Read-only: JSON only, no models run. Parts 1 and 2 are in `count_only_check_part1.md` and `count_only_check_part2.md`._

| File | Test outcome | Key path and value that shows it | Match count of that population |
|---|---|---|---|
| data/diagnostics/w2v_published_eval_w2v_gensim_repeat.json | yes | `summary_mean_over_seeds.random_leaky.B.logloss` = 0.47858; `verdicts.P1_published_gain_random_leaky.passes` = false | per seed 2657, 2659, 2653 (`random_leaky.test_matches`) |
| data/diagnostics/w2v_published_eval_w2v_gensim_depth4.json | yes | `summary_mean_over_seeds.random_leaky.B.logloss` = 0.44968; `verdicts.P1_published_gain_random_leaky.passes` = false | per seed 2657, 2659, 2653 |
| data/diagnostics/defuse_race_eval.json | yes | `verdicts.DX_adds_information.passes` = true; `verdicts.DW_adds_information.passes` = false | `test_matches` 687 (`rows_test` 92,360) |
| data/diagnostics/w2v_published_eval_w2v_pubsent.json | yes | `summary_mean_over_seeds.random_leaky.B.logloss` = 0.46683; `verdicts.P1_published_gain_random_leaky.passes` = false | per seed 2635, 2650, 2623 |
| data/diagnostics/w2v_published_eval_w2v_pubsent_depth4.json | yes | `summary_mean_over_seeds.random_leaky.B.logloss` = 0.43820; `verdicts.P1_published_gain_random_leaky.passes` = false | per seed 2635, 2650, 2623 |
| data/diagnostics/defuse_race_eval2.json | yes | `verdicts.DWX_adds_information.passes` = true (mean −0.00051) | `per_seed.0.DWX_minus_A.matches` 687 (`rows` 92,360) |
| data/diagnostics/defect_rounds_and_device_check.json | yes, but degenerate | `defect_round_effect.material` = false; `mean_INCL_minus_EXCL` = 0.0 | 687 (`rows` 92,360; `defect_test_rows` 0). Both defect log losses are NaN, so the delta is zero by construction |
| data/diagnostics/device_parity_check.json | unclear (not a performance outcome) | `agrees_1e6` = true; `max_abs_diff` = 4.2e-07 | none. Only `rows_test` 92,360. This checks CPU and GPU agreement, not model quality |
| data/diagnostics/w2v_published_eval_w2v_pubsent_sg1.json | yes | `summary_mean_over_seeds.random_leaky.B.logloss` = 0.46683; `verdicts.P1_published_gain_random_leaky.passes` = false | per seed 2635, 2650, 2623 |
| data/diagnostics/w2v_published_eval_w2v_pubsent_s1e5.json | yes | `summary_mean_over_seeds.random_leaky.B.logloss` = 0.46683; `verdicts.P1_published_gain_random_leaky.passes` = false | per seed 2635, 2650, 2623 |
| data/diagnostics/walking_approx_eval.json | yes | `verdicts.P_portal_adds_information.passes` = true (mean −0.00208) | `per_seed.0.S_minus_A.matches` 687 (`rows` 92,360); `planted_test_rows` 12,989 |
| data/diagnostics/w2v_published_eval_w2v_pubsent_ep20.json | yes | `summary_mean_over_seeds.random_leaky.B.logloss` = 0.46683; `verdicts.P1_published_gain_random_leaky.passes` = false | per seed 2635, 2650, 2623 |
| data/diagnostics/map_control_eval.json | yes | `verdicts.map_control_adds_information.passes` = false (mean −0.00019) | `per_seed.0.MC_minus_A.matches` 687 (`rows_test` 92,360) |
| data/diagnostics/allmaps_xgb_geo.json | yes | `log_loss.S` = 0.44828 (A = 0.45083); `S_passes` = true | `test_matches` 687 (`rows_test` 865,974) |
| data/diagnostics/w2v_paper_scale_honest_eval.json | yes | `pre_registered_verdicts.frozen_embedding_helps_4_features` = false; `summary.frozen_E4_minus_B4.mean` = +0.04003 | no single matches key. 20 samples, `rows_test` summing to 296 |

## Summary for part 3

- 14 yes, 1 unclear (`device_parity_check.json`, a CPU and GPU agreement check, not a model outcome), 0 no.
- `defect_rounds_and_device_check.json` is counted as yes, but its delta is zero by construction (no defect test rows).
- The word2vec files use sample-based test populations (2,623 to 2,659 matches per seed), not the 687-match test.
- These files report test outcomes under generic key names (for example `logloss`, `delta_vs_xgb`, `verdicts`), so the ledger's COUNT-ONLY class understates them. All three parts together: 46 yes, 1 unclear.
