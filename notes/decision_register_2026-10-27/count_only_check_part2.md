# COUNT-ONLY check, part 2 (ledger rows 17 to 32)

Scope: rows 17 to 32 of the COUNT-ONLY rows in `unpinned_test_ledger.md`, in table order (first is `data/combined/exp_full_comparison_part_a.json`, last is `data/diagnostics/w2v_published_eval_w2v_gensim.json`). JSON read only, no models run. Paths are relative to `G:/pureskill/csds-cs2/`.

| File | Test outcome | Key path and value | Match count |
|---|---|---|---|
| data/combined/exp_full_comparison_part_a.json | yes | models.xgb.logloss = 0.4564122947410805; models.lgbm.delta_vs_xgb = 0.0003696374170810339 | 3059 (test_rows 3956374) |
| data/combined/exp_full_comparison_part_b.json | yes | models.xgb.logloss = 0.4564122947410805; models.catboost.delta_vs_xgb = 0.00034164578142589823 | 3059 (test_rows 3956374) |
| data/combined/exp_families_torch.json | yes | models.mlp_torch_gpu.logloss = 0.4625169408498078; models.mlp_torch_gpu.delta_vs_xgb = 0.006104646108727358 | 3059 (test_rows 3956374) |
| data/combined/exp_families_cuml.json | yes | models.cuml_logit.logloss = 0.46741999387643923; models.cuml_logit.delta_vs_xgb = 0.011007699135358744 | 3059 (test_rows 3956374) |
| data/combined/exp_full_comparison.json | yes | models.xgb.logloss = 0.4564122947410805; models.catboost_gpu.delta_vs_xgb = 0.0004021201574630081 (merged from the four part files above) | 3059 (test_rows 3956374) |
| data/combined/exp_tick_sampling_test_only_old.json | yes | cells.nall_seed0.logloss = 0.45645516213396076; cells.n1_seed0.delta_vs_every_second = 0.0024348504600781346 | 3059 (test_rows 3956374) |
| data/combined/exp_tick_sampling.json | yes | cells.nall_seed0.logloss = 0.45645516213396076; cells.n1_seed0.delta_vs_every_second = 0.0024348504600781346 (sibling validation_logloss keys exist, so logloss is the test value) | 3059 (test_rows 3956374) |
| data/combined/exp_rounds_groups.json | yes | groups."All other rounds".mean = 0.46362387871225635; all_rounds.mean = 0.45560455098574587 (metric not labeled; values are on log-loss scale) | per group: 3033 / 3032 / 3047 (pistol / after pistol / other); no top-level test_matches; test_seconds 3956374 |
| data/diagnostics/general_then_finetune.json | yes | overall_all_maps.general_vs_baseline.logloss = 0.45846222074544835 (baseline 0.4564107438116982, diff 0.0020514769337501293) | 3059 (rows 3956374) |
| data/diagnostics/why_models_lag.json | yes | results[0].test.logloss = 0.45132320196856834 (C0 control, de_mirage, frozen manifest) | 687 (n_rows 865974) |
| data/diagnostics/spatial_feature_eval.json | yes | VD_minus_A.diff = -0.0007103075472145171; VD_adds_information = true | 687 (rows_test 865974) |
| data/diagnostics/nav_path_eval.json | yes | P_minus_A.diff = -0.0017480644301480774; P_adds_information = true | 687 (rows_test 92360) |
| data/diagnostics/nav_path_eval2.json | yes | verdicts.P_adds_information.mean_over_seeds = -0.0018304374197087674, passes = true | 687 (rows_test 92360) |
| data/diagnostics/emb_eval.json | yes | verdicts.z_adds_information.mean_over_seeds = -0.0005359666696348094, passes = true | 687 (rows_test 92360) |
| data/diagnostics/w2v_eval2.json | yes | verdicts.w2v_adds_information.mean_over_seeds = 0.00012062527370169478, passes = false | 687 (per-seed W_minus_A matches; rows_test 92360) |
| data/diagnostics/w2v_published_eval_w2v_gensim.json | yes | verdicts.P1_published_gain_random_leaky.mean_over_seeds = 0.03928565402822263, passes = false | 2657 / 2659 / 2653 (seeds 0 / 1 / 2, random_leaky E_minus_B, rows 5263 / 5250 / 5199) |

## Notes

- Result: 16 of 16 files report a model outcome computed on test rows. None is "no" or "unclear". Only the rows_test or test_seconds or nested "test" keys carry the test population, and the validation_* siblings in exp_tick_sampling.json show that its bare logloss key is the test value.
- exp_rounds_groups.json: the metric is not named. The values (0.456 overall, 0.519 pistol, 0.313 after pistol) are on the log-loss scale, which is the inference used here.
- w2v_published_eval_w2v_gensim.json: the file has several test sub-populations with different match counts (for example 487 in the seed-2 match_honest block). The ledger's "test_matches=487" comes from that block, not from the P1 verdict population, which is the random_leaky E_minus_B set listed above.
- Ledger consistency: the test_matches and test_rows values recorded in the ledger for rows 17 to 32 match the JSON, except the w2v_published row noted above.
