| File | Modified (CDT) | Class | Test count keys | Test outcome keys | Split / decision-set strings |
|---|---|---|---|---|---|
| data/combined/exp_status_only.json | 14 Oct 05:07 | no test keys |  |  |  |
| data/combined/exp_stack_2026.json | 14 Oct 05:12 | no test keys |  |  |  |
| data/combined/exp_stack_2026b.json | 14 Oct 05:18 | no test keys |  |  |  |
| data/combined/exp_stack_all_eras.json | 14 Oct 05:24 | no test keys |  |  |  |
| data/combined/exp_foundation.json | 14 Oct 23:14 | COUNT-ONLY (check generic keys) | test_seconds=500 |  |  |
| data/combined/exp_finetune_smoke.json | 15 Oct 03:41 | COUNT-ONLY (check generic keys) | test_seconds=1000 |  |  |
| data/combined/exp_finetune.json | 15 Oct 04:34 | COUNT-ONLY (check generic keys) | test_seconds=10000 |  |  |
| data/combined/exp_finetune_tabicl.json | 15 Oct 04:57 | COUNT-ONLY (check generic keys) | test_seconds=300 |  |  |
| data/combined/exp_deep_distill.json | 15 Oct 12:59 | COUNT-ONLY (check generic keys) | test_rows=3226987, test_matches=2497 |  |  |
| data/combined/exp_deep_distill_regA.json | 15 Oct 13:05 | COUNT-ONLY (check generic keys) | test_rows=3226987, test_matches=2497 |  |  |
| data/combined/exp_robustness.json | 15 Oct 16:50 | OUTCOME-KEYED | test_matches=287, test_rows=368850 | constant_logloss_on_test, test_logloss_shuffled_label_model |  |
| data/combined/exp_model_families.json | 15 Oct 16:50 | COUNT-ONLY (check generic keys) | test_rows=3956374, test_matches=3059 |  |  |
| data/combined/exp_temporal.json | 15 Oct 16:50 | COUNT-ONLY (check generic keys) | later_day_test_matches=644 |  |  |
| data/combined/exp_deep_distill_regB.json | 15 Oct 16:50 | COUNT-ONLY (check generic keys) | test_rows=3956374, test_matches=3059 |  |  |
| data/combined/exp_hparams.json | 15 Oct 16:52 | OUTCOME-KEYED | test_matches=3059 | test_logloss, test_logloss_ci95 |  |
| data/combined/exp_positions.json | 15 Oct 17:04 | COUNT-ONLY (check generic keys) | test_matches=3059, test_seconds=1992866 |  |  |
| data/combined/exp_curve_accuracy.json | 15 Oct 17:56 | COUNT-ONLY (check generic keys) | test_matches=3059, test_seconds=3956374 |  |  |
| data/combined/exp_gnn.json | 15 Oct 18:12 | COUNT-ONLY (check generic keys) | test_rows=1992866, test_matches=3059 |  |  |
| data/combined/exp_full_positions.json | 16 Oct 12:44 | COUNT-ONLY (check generic keys) | test_matches=3059, test_rows=3956374 |  |  |
| data/combined/exp_round_features.json | 16 Oct 12:49 | COUNT-ONLY (check generic keys) | test_matches=3059, test_rows=3956374 |  |  |
| data/combined/exp_methods_pod_run1.json | 16 Oct 13:52 | OUTCOME-KEYED | fallback_rows_test=320192 | test_delta_vs_M1_seed0, test_logloss_sd |  |
| data/combined/exp_methods.json | 16 Oct 15:32 | OUTCOME-KEYED |  | test_delta_vs_M1_seed0, test_logloss_sd |  |
| data/combined/exp_pistol.json | 16 Oct 16:02 | no test keys |  |  |  |
| data/combined/exp_positions_groups_pod_cpu.json | 16 Oct 17:54 | no test keys |  |  |  |
| data/combined/exp_tick_sampling_pod_cpu.json | 16 Oct 18:00 | COUNT-ONLY (check generic keys) | test_matches=3059, test_rows=3956374 |  |  |
| data/combined/exp_tune_xgb.json | 16 Oct 18:27 | no test keys |  |  |  |
| data/combined/exp_grid_step.json | 16 Oct 18:45 | no test keys |  |  | one-second grid kept |
| data/combined/exp_positions_groups.json | 16 Oct 19:49 | no test keys |  |  |  |
| data/combined/exp_mono_paired.json | 16 Oct 23:43 | OUTCOME-KEYED | test_matches=3601 | test_logloss_meta, test_logloss_recomputed |  |
| data/combined/exp_full_comparison_test_only_old.json | 16 Oct 23:46 | COUNT-ONLY (check generic keys) | test_matches=3059, test_rows=3956374 |  |  |
| data/combined/exp_full_comparison_part_a.json | 17 Oct 02:49 | COUNT-ONLY (check generic keys) | test_matches=3059, test_rows=3956374 |  |  |
| data/combined/exp_full_comparison_part_b.json | 17 Oct 02:49 | COUNT-ONLY (check generic keys) | test_matches=3059, test_rows=3956374 |  |  |
| data/combined/exp_families_torch.json | 17 Oct 02:49 | COUNT-ONLY (check generic keys) | test_matches=3059, test_rows=3956374 |  |  |
| data/combined/exp_families_cuml.json | 17 Oct 02:52 | COUNT-ONLY (check generic keys) | test_matches=3059, test_rows=3956374 |  |  |
| data/combined/exp_full_comparison.json | 17 Oct 02:52 | COUNT-ONLY (check generic keys) | test_matches=3059, test_rows=3956374 |  |  |
| data/combined/exp_tick_sampling_test_only_old.json | 17 Oct 04:04 | COUNT-ONLY (check generic keys) | test_matches=3059, test_rows=3956374 |  |  |
| data/combined/exp_tick_sampling.json | 17 Oct 04:04 | COUNT-ONLY (check generic keys) | test_matches=3059, test_rows=3956374 |  |  |
| data/combined/exp_pistol_validation.json | 17 Oct 04:04 | no test keys |  |  |  |
| data/combined/exp_cap_trace.json | 17 Oct 05:51 | no test keys |  |  |  |
| data/combined/exp_cache_skip_trace.json | 17 Oct 05:51 | no test keys |  |  |  |
| data/combined/exp_scorelead.json | 17 Oct 05:56 | OUTCOME-KEYED | test_rows=4568043, test_matches=3601 | served_13_test_logloss_reported, test_delta_vs_served_13, test_delta_vs_served_13_ci95, test_logloss |  |
| data/combined/exp_cap_row_trace.json | 17 Oct 06:03 | no test keys |  |  |  |
| data/combined/exp_roster_sensitivity.json | 17 Oct 06:09 | OUTCOME-KEYED | test_seconds=4568043, test_matches=3601 | all_round_minus_served_clean_test, test_logloss_all_round_clean, test_logloss_served_clean |  |
| data/combined/exp_59_scorelead.json | 17 Oct 10:35 | OUTCOME-KEYED | test_rows=3956374, test_matches=3059 | reference_test_logloss_meta, reference_test_logloss_recomputed, test_delta_vs_59, test_delta_vs_59_ci95 |  |
| data/combined/exp_59_nomono.json | 17 Oct 10:39 | OUTCOME-KEYED | test_matches=3059 | test_delta_ci95, test_delta_no_mono_minus_mono, test_logloss_no_mono, test_logloss_no_mono_ci95 |  |
| data/combined/exp_random_ticks.json | 17 Oct 21:36 | OUTCOME-KEYED |  | reference_test_logloss, test_delta_ci95, test_delta_vs_every_second, test_logloss |  |
| data/combined/exp_random_ticks59.json | 18 Oct 19:39 | OUTCOME-KEYED | test_rows=3956374 | test_delta_ci95, test_delta_vs_every_second, test_logloss, test_logloss_ci95 |  |
| data/combined/exp_loadout_ablation.json | 18 Oct 21:28 | OUTCOME-KEYED |  | test_delta_ci95, test_delta_vs_control, test_logloss |  |
| data/combined/exp_loadout_joint.json | 18 Oct 21:40 | OUTCOME-KEYED |  | test_delta_ci95, test_delta_vs_control, test_logloss | keep the 59-input model |
| data/combined/exp_loadout_blocks.json | 18 Oct 21:49 | OUTCOME-KEYED |  | test_delta_ci95, test_delta_vs_control, test_logloss |  |
| data/combined/exp_loadout_select.json | 18 Oct 22:01 | OUTCOME-KEYED | test_rows=3956374 | test_delta_ci95, test_delta_vs_control, test_logloss |  |
| data/combined/exp_loadout_seed1.json | 18 Oct 22:05 | OUTCOME-KEYED |  | test_delta_ci95, test_delta_vs_control, test_logloss |  |
| data/combined/exp_scorelead_combo.json | 18 Oct 22:19 | OUTCOME-KEYED |  | test_delta_ci95, test_delta_vs_control, test_logloss |  |
| data/combined/exp_hparams58.json | 18 Oct 22:41 | OUTCOME-KEYED |  | test_delta_ci95, test_delta_vs_baseline, test_logloss |  |
| data/combined/exp_seed_avg58.json | 18 Oct 22:51 | OUTCOME-KEYED | test_rows=3956374 | seed_mean_test_logloss, test_delta_vs_seed0, test_delta_vs_seed0_ci95, test_logloss |  |
| data/combined/exp_event_features.json | 18 Oct 23:34 | OUTCOME-KEYED |  | test_delta_ci95, test_delta_vs_baseline, test_logloss |  |
| data/combined/exp_tick_rates.json | 19 Oct 01:53 | OUTCOME-KEYED |  | test_delta_ci95, test_delta_vs_every_second, test_logloss |  |
| data/combined/exp_tick8.json | 19 Oct 09:42 | OUTCOME-KEYED |  | test_delta_ci95, test_delta_vs_every_second, test_logloss |  |
| data/combined/exp_final_checks.json | 19 Oct 10:56 | OUTCOME-KEYED |  | test_logloss_with_future_feature, unweighted_test_logloss, weighted_test_logloss |  |
| data/combined/exp_eventonly_paired.json | 19 Oct 13:25 | OUTCOME-KEYED | test_seconds=3956374, test_matches=3059 | event_only_test_logloss, production_test_logloss |  |
| data/combined/exp_families_pistol.json | 19 Oct 14:01 | OUTCOME-KEYED | test_seconds=320192, test_matches=3032 | dedicated_test_logloss, main_test_logloss, test_delta_ci95, test_delta_vs_xgboost |  |
| data/combined/exp_rounds_groups.json | 19 Oct 14:20 | COUNT-ONLY (check generic keys) | test_seconds=3956374 |  |  |
| data/combined/exp_submission_57.json | 19 Oct 19:03 | OUTCOME-KEYED |  | paired_57_minus_58_ensemble_test |  |
| data/combined/exp_scorelead_leak.json | 19 Oct 20:10 | no test keys |  |  |  |
| data/diagnostics/general_then_finetune.json | 23 Oct 15:54 | COUNT-ONLY (check generic keys) | test_rows=3956374, test_matches=3059 |  |  |
| data/diagnostics/why_models_lag.json | 23 Oct 15:56 | COUNT-ONLY (check generic keys) | test_rows=865974, test_matches=687 |  | frozen manifest |
| data/diagnostics/w2v_eval.json | 23 Oct 18:00 | no test keys |  |  |  |
| data/diagnostics/spatial_feature_eval.json | 23 Oct 18:03 | COUNT-ONLY (check generic keys) | rows_test=865974 |  |  |
| data/diagnostics/ext_baseline_all_seed1.json | 24 Oct 11:09 | OUTCOME-KEYED | rows_test=865974, matches_test=687 | logloss_test | all |
| data/diagnostics/ext_temporal_paired_de_mirage_ext_loadout_gru_vs_de_mirage_loadout_gru.json | 24 Oct 11:54 | no test keys |  |  |  |
| data/diagnostics/ext_temporal_paired_de_mirage_ext_loadout_gru_seed1_vs_de_mirage_loadout_gru_seed1.json | 24 Oct 14:04 | no test keys |  |  |  |
| data/diagnostics/ext_temporal_paired_de_mirage_ext_loadout_gru_seed2_vs_de_mirage_loadout_gru_seed2.json | 24 Oct 15:07 | no test keys |  |  |  |
| data/diagnostics/ext_temporal_paired_de_mirage_ext_loadout_gru_ms8544_vs_de_mirage_loadout_gru_ms8544.json | 24 Oct 15:35 | no test keys |  |  |  |
| data/diagnostics/ext_temporal_paired_de_mirage_ext_thgnn_lite_rerun_vs_de_mirage_thgnn_lite_rerun.json | 24 Oct 17:40 | no test keys |  |  |  |
| data/diagnostics/ext_baseline_de_mirage.json | 24 Oct 17:55 | OUTCOME-KEYED | rows_test=865974, matches_test=687 | logloss_test | de_mirage |
| data/diagnostics/ext_baseline_all.json | 24 Oct 18:00 | OUTCOME-KEYED | rows_test=865974, matches_test=687 | logloss_test | all |
| data/diagnostics/xgb_vs_temporal_investigation.json | 24 Oct 18:20 | OUTCOME-KEYED |  | test_log_loss | all maps (train); all maps (train) + 2026-10 additions; de_mirage only; de_mirage only + additions |
| data/diagnostics/allmaps_gru_vs_xgb.json | 24 Oct 19:06 | OUTCOME-KEYED |  | test_log_loss |  |
| data/diagnostics/nav_path_eval.json | 25 Oct 01:32 | COUNT-ONLY (check generic keys) | rows_test=92360, test_matches=687 |  |  |
| data/diagnostics/nav_path_eval2.json | 25 Oct 01:38 | COUNT-ONLY (check generic keys) | rows_test=92360, test_matches=687 |  |  |
| data/diagnostics/emb_eval.json | 25 Oct 10:45 | COUNT-ONLY (check generic keys) | rows_test=92360, test_matches=687 |  |  |
| data/diagnostics/w2v_eval2.json | 25 Oct 10:46 | COUNT-ONLY (check generic keys) | rows_test=92360 |  |  |
| data/diagnostics/w2v_published_eval_w2v_gensim.json | 25 Oct 11:02 | COUNT-ONLY (check generic keys) | test_matches=487 |  |  |
| data/diagnostics/w2v_published_eval_w2v_gensim_repeat.json | 25 Oct 11:02 | COUNT-ONLY (check generic keys) | test_matches=487 |  |  |
| data/diagnostics/w2v_published_eval_w2v_gensim_depth4.json | 25 Oct 11:03 | COUNT-ONLY (check generic keys) | test_matches=487 |  |  |
| data/diagnostics/defuse_race_eval.json | 25 Oct 11:53 | COUNT-ONLY (check generic keys) | rows_test=92360, test_matches=687, planted_test_rows=12989 |  |  |
| data/diagnostics/w2v_published_eval_w2v_pubsent.json | 25 Oct 12:09 | COUNT-ONLY (check generic keys) | test_matches=437 |  |  |
| data/diagnostics/w2v_published_eval_w2v_pubsent_depth4.json | 25 Oct 12:11 | COUNT-ONLY (check generic keys) | test_matches=437 |  |  |
| data/diagnostics/defuse_race_eval2.json | 25 Oct 12:26 | COUNT-ONLY (check generic keys) | planted_test_rows=12989 |  |  |
| data/diagnostics/defect_rounds_and_device_check.json | 25 Oct 12:28 | COUNT-ONLY (check generic keys) | nondefect_test_rows=92360 |  |  |
| data/diagnostics/device_parity_check.json | 25 Oct 12:31 | COUNT-ONLY (check generic keys) | rows_test=92360 |  |  |
| data/diagnostics/w2v_published_eval_w2v_pubsent_sg1.json | 25 Oct 12:44 | COUNT-ONLY (check generic keys) | test_matches=437 |  |  |
| data/diagnostics/stride_check_eval.json | 25 Oct 12:54 | no test keys |  |  |  |
| data/diagnostics/w2v_published_eval_w2v_pubsent_s1e5.json | 25 Oct 13:16 | COUNT-ONLY (check generic keys) | test_matches=437 |  |  |
| data/diagnostics/walking_approx_eval.json | 25 Oct 13:16 | COUNT-ONLY (check generic keys) | planted_test_rows=12989 |  |  |
| data/diagnostics/w2v_published_eval_w2v_pubsent_ep20.json | 25 Oct 13:21 | COUNT-ONLY (check generic keys) | test_matches=437 |  |  |
| data/diagnostics/gate_2024_residual.json | 25 Oct 13:38 | no test keys |  |  |  |
| data/diagnostics/gate_2024_residual_followup.json | 25 Oct 13:39 | no test keys |  |  |  |
| data/diagnostics/gate_2024_vanish_check.json | 25 Oct 13:42 | no test keys |  |  |  |
| data/diagnostics/gate_2024_side_recovery.json | 25 Oct 13:46 | no test keys |  |  |  |
| data/diagnostics/gate_2024_month.json | 25 Oct 13:47 | no test keys |  |  |  |
| data/diagnostics/map_control_eval.json | 25 Oct 14:17 | COUNT-ONLY (check generic keys) | rows_test=92360 |  |  |
| data/diagnostics/eval_2024_frozen.json | 25 Oct 20:23 | no test keys |  |  |  |
| data/diagnostics/w2v_paper_scale_eval.json | 25 Oct 20:51 | no test keys |  |  |  |
| data/diagnostics/w2v_paper_scale_frozen.json | 25 Oct 21:40 | no test keys |  |  |  |
| data/diagnostics/attrib_2024_status.json | 26 Oct 05:33 | no test keys |  |  |  |
| data/diagnostics/allmaps_xgb_geo.json | 26 Oct 08:23 | COUNT-ONLY (check generic keys) | rows_test=865974, test_matches=687 |  |  |
| data/diagnostics/in_era_2024.json | 26 Oct 08:23 | no test keys |  |  |  |
| data/diagnostics/w2v_paper_scale_honest_eval.json | 26 Oct 08:25 | COUNT-ONLY (check generic keys) | rows_test=21, mean_test_rows=14.8 |  |  |
| data/diagnostics/allmaps_xgb_geo_seedcheck.json | 26 Oct 08:47 | no test keys |  |  |  |
| data/diagnostics/allmaps_xgb_sitecheck.json | 26 Oct 10:53 | OUTCOME-KEYED |  | SN_minus_A_test_descriptive, SO_minus_A_test_descriptive, test_log_loss | validation seconds only |
| data/diagnostics/allmaps_mlp_vs_xgbA.json | 26 Oct 10:58 | OUTCOME-KEYED | aligned_rows_test=865974 | mlp_log_loss_test, mlp_minus_xgbA_test, xgbA_log_loss_test | validation seconds only |
| data/diagnostics/t7_time_control.json | 26 Oct 12:02 | OUTCOME-KEYED |  | DXT_minus_DX_test_descriptive | validation seconds |
| data/diagnostics/c1_contested_distance.json | 26 Oct 12:06 | OUTCOME-KEYED |  | mean_contested_SN_minus_A, passes_contested_rule |  |
| data/diagnostics/g1_gru_vs_xgb_validation.json | 26 Oct 13:36 | no test keys |  |  |  |
| data/diagnostics/b1_blend_validation.json | 26 Oct 14:16 | no test keys |  |  |  |
| data/diagnostics/t4_map_indicator.json | 26 Oct 14:47 | OUTCOME-KEYED |  | SNM_minus_SN_test_descriptive | validation seconds |
| data/diagnostics/halftime_label_audit.json | 26 Oct 14:48 | no test keys |  |  |  |
| data/diagnostics/w2_walk_trainsites.json | 26 Oct 16:15 | OUTCOME-KEYED |  | W2_minus_A_test_descriptive |  |
| data/diagnostics/t1_engagement_stack.json | 26 Oct 16:43 | OUTCOME-KEYED |  | S_plus_eng_minus_S_test_descriptive |  |
| data/diagnostics/t0_logreg_vs_xgb.json | 26 Oct 17:47 | no test keys |  |  | validation seconds |
| data/diagnostics/t2a_events.json | 26 Oct 20:57 | OUTCOME-KEYED |  | S_plus_events_minus_S_test_descriptive |  |
| data/diagnostics/t3_player_attention.json | 26 Oct 21:01 | OUTCOME-KEYED | rows_test=865974 | GAT_minus_GRU_test_descriptive | de_mirage validation seconds (usable mas |
| data/diagnostics/t6_defuse_speed.json | 26 Oct 21:03 | OUTCOME-KEYED |  | PW_minus_CONST_test_descriptive | post-plant validation seconds, de_mirage |

Total files: 124. Classes: {'no': 36, 'COUNT-ONLY': 47, 'OUTCOME-KEYED': 41}
OUTCOME-KEYED: 41 files; earliest 15 Oct 16:50 (data/combined/exp_robustness.json); latest 26 Oct 21:03 (data/diagnostics/t6_defuse_speed.json)
COUNT-ONLY: 47 files; earliest 14 Oct 23:14 (data/combined/exp_foundation.json); latest 26 Oct 08:25 (data/diagnostics/w2v_paper_scale_honest_eval.json)
