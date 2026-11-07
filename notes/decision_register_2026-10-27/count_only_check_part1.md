# Count-only check, part 1 (first 16 of 47)

_Saved by the parent from the finished subagent result (27 Oct 2026). Source: workflow wf_f3505f48-751. Subagent rules barred it from writing this file itself._

Checked the first 16 COUNT-ONLY rows of G:/pureskill/csds-cs2/notes/decision_register_2026-10-27/unpinned_test_ledger.md, in table order. Read each JSON under G:/pureskill/csds-cs2/data/combined/ (no models run). Result: 16 yes, 0 no, 0 unclear. Every file has a loss, delta, or accuracy value sitting next to an explicit test-count key, so none is truly count-only.

NOT WRITTEN: the requested file G:/pureskill/csds-cs2/notes/decision_register_2026-10-27/count_only_check_part1.md. My subagent output rules bar writing report or findings .md files, so the table is returned here for the parent to save.

File | test outcome | key path and value | match count of test population
G:/pureskill/csds-cs2/data/combined/exp_foundation.json | yes | "tabicl|ctx5000|cpu|est4|n1000".logloss = 0.46305, with sibling test_seconds in the same record (outcome key is unprefixed, weakest link) | no test_matches key; test_seconds per record = 1000, 1000, 4000, 500
G:/pureskill/csds-cs2/data/combined/exp_finetune_smoke.json | yes | "tabpfn-v2 zero-shot ctx5000".logloss = 0.46341, delta_vs_xgb = 0.01216 | test_seconds = 1000 (both records)
G:/pureskill/csds-cs2/data/combined/exp_finetune.json | yes | "tabpfn-v2 zero-shot ctx10000".logloss = 0.46928; "tabpfn-v2 fine-tuned ep6 lr1e-05 ctx10000".logloss = 0.46849, delta_vs_xgb = 0.00459 | test_seconds = 10000 (both records)
G:/pureskill/csds-cs2/data/combined/exp_finetune_tabicl.json | yes | "tabicl-v2 zero-shot ctx2000".logloss = 0.52081, delta_vs_xgb = 0.02220 | test_seconds = 300
G:/pureskill/csds-cs2/data/combined/exp_deep_distill.json | yes | top-level test_rows = 3226987 beside xgboost_hard_labels.logloss = 0.45741 and deep_teacher delta_vs_xgboost_hard = 0.00205 | test_matches = 2497
G:/pureskill/csds-cs2/data/combined/exp_deep_distill_regA.json | yes | top-level test_rows = 3226987 beside xgboost_hard_labels.logloss = 0.45741 and deep_teacher delta_vs_xgboost_hard = 0.00199 | test_matches = 2497
G:/pureskill/csds-cs2/data/combined/exp_model_families.json | yes | setup.test_rows = 3956374 beside xgb_gbdt.logloss = 0.45675 and sk_mlp.delta_vs_xgb_gbdt = 0.00147 | setup.test_matches = 3059
G:/pureskill/csds-cs2/data/combined/exp_temporal.json | yes | frozen_test_matches_after_cutoff.n_matches = 644 beside reference.logloss = 0.45306, temporal.logloss = 0.45335, temporal_minus_reference = 0.00028 | later_day_test_matches = 644 (n_rows 834165)
G:/pureskill/csds-cs2/data/combined/exp_deep_distill_regB.json | yes | top-level test_rows = 3956374 beside xgboost_hard_labels.logloss = 0.45647 and deep_teacher delta_vs_xgboost_hard = 0.00036 | test_matches = 3059
G:/pureskill/csds-cs2/data/combined/exp_positions.json | yes | top-level test_seconds = 1992866 beside base.logloss = 0.45761 and geometry.delta_vs_base = -0.00296 | test_matches = 3059
G:/pureskill/csds-cs2/data/combined/exp_curve_accuracy.json | yes | top-level test_seconds = 3956374 beside points[].accuracy (0.7586 at fraction 0.05 to 0.7606 at 1.0) and points[].logloss | test_matches = 3059
G:/pureskill/csds-cs2/data/combined/exp_gnn.json | yes | top-level test_rows = 1992866 beside models.gnn_hyb1.logloss = 0.45385, delta_vs_xgb83 = -0.00056 | test_matches = 3059
G:/pureskill/csds-cs2/data/combined/exp_full_positions.json | yes | top-level test_rows = 3956374 beside models.base59.logloss = 0.45647 and models.pos83.delta_vs_base59 = -0.00324 | test_matches = 3059
G:/pureskill/csds-cs2/data/combined/exp_round_features.json | yes | top-level test_rows = 3956374 beside models.base59.logloss = 0.45641 and split.delta_vs_base59 = 0.00050 | test_matches = 3059
G:/pureskill/csds-cs2/data/combined/exp_tick_sampling_pod_cpu.json | yes | top-level test_rows = 3956374 beside cells.nall_seed0.logloss = 0.45645 and cells.n1_seed0.delta_vs_every_second = 0.00236 | test_matches = 3059
G:/pureskill/csds-cs2/data/combined/exp_full_comparison_test_only_old.json | yes | top-level test_rows = 3956374 beside models.xgb.logloss = 0.45641 and models.lgbm.delta_vs_xgb = 0.00037 | test_matches = 3059

## Findings

- All 16 checked files are misclassified as COUNT-ONLY. Each reports loss, delta, or accuracy values beside an explicit test-count key, so they should be reclassified OUTCOME-KEYED. The ledger's Test outcome keys column was empty for these rows because only prefixed test_* keys were checked.
- Weakest link is exp_foundation.json: its outcome keys (logloss, delta_vs_xgb) are unprefixed, and test-ness rests only on the sibling test_seconds key in the same record. Verdict kept as yes, but it deserves a human look.
- exp_temporal.json has a second block (temporal_on_all_later_matches, n_matches 3010) that is not labeled as test. The verdict relies on the frozen_test_matches_after_cutoff block (n_matches 644).
- Ledger value for exp_foundation.json shows test_seconds=500, which is only the last of four records (1000, 1000, 4000, 500). Other ledger test-count values match the files.
- Nothing was run and no model was executed. Read-only JSON inspection only.
