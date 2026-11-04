# Counter-Strike round win probability: state at wrap-up (4 Nov 2026)

Code and results summary. Match data, feature tables and model outputs are not in git (parquet, npy and logs are ignored). The pin IDs below refer to the project's Meridian record.

## Final candidate

XGBoost on the full feature set: 83 inputs (59 base, plus GEO, VIEW and BLIND position groups), depth 4, eta 0.03, min_child_weight 20, subsample 0.8, 256 histogram bins, early stopping at 100 rounds, 3,000-round cap, seeds 0 to 2, train-centre position table. Chosen on validation only (pin 88d9fbb0).

## Results

| Result | Value | Pin |
|---|---|---|
| Validation, seed-mean log loss (1,168 matches, 1,515,228 seconds) | 0.450252 | 88d9fbb0 |
| Validation, sequence models vs candidate (same seconds) | all worse; closest is width-128 transformer, +0.00107 (3 seeds) | 88d9fbb0 |
| Transformer width 128, five seeds vs candidate | no claim under the pinned rule (pooled +0.00107; seed 2 interval crosses zero) | edb0aaa3 |
| GRU, 60-second vs 20-second window | no claim (pooled +0.000184) | 7085b272 |
| Holdout, Oct 26 to 31 uploads (3,315 matches, 4,320,801 seconds), seed mean | 0.452295; match-cluster 95% interval [0.449288, 0.454983]. Scored once. | 56316f69 |
| Holdout checks: duplicates, coverage | no overlap with training or holdout B; all scored matches are keepers; 10 in-scope keeper matches not in the cache, not scored | 538c3c01 |

The validation number is optimistic, because it was the selection. The holdout score is a single read, and its status depends on Dr. Freeman's go, which is not on record.

## Event-only state table (SageMaker archive)

Built on the SageMaker space by `scripts/sagemaker/build_event_state.py`: 19,498 matches, 374,073 rounds, 3,053,289 states, 10 features (the bake-off's `build_states`, copied unchanged). Holdout B (653 matches) excluded. Parity with the bake-off's own output on the same matches has not been run (pin 9053f4c2 records the build; d7d0bb65 records the data finding).

## Not done

- Dr. Freeman's go or no-go on the holdout score.
- A fresh holdout for any new candidate. The Oct 26 to 31 window is now used for this candidate.
- Older-format AWS matches (46,627): no `team_change` channel and null `max_rounds`, so they are excluded. Needs a rule decision.
- Parity check of the SageMaker state builder against the bake-off.
- Position features and per-tick features for the AWS archive.
- GDT for the sequence models.
- Holdout B is recorded as contaminated and is not used for any clean claim.

## Scope

Models use within-round information only. Cross-round history (earlier rounds' winners, economy) is excluded by the project's rules (pin 79da6d09). Validation only, except for the one holdout scoring.

## Where the code is

- `experiments/sprint/`: validation design matrix, XGBoost sprint worker and queues.
- `experiments/positions/`: position-feature validation and windowed predictions.
- `analysis/sprint_report.py`, `analysis/positions_validation_paired.py`: paired reports.
- `evaluation/build_validation_gdt_rows.py`: validation GDT rows.
- `scripts/final/`: final candidate comparison, the five-seed and 60-second checks, and the holdout build, gate and score.
- `scripts/sagemaker/`: the event-only state builder and its run script.
- `docs/eval_reports/`, `docs/sprint/`, `docs/positions_validation/`: generated reports.

The scripts use absolute paths from the authors' machines (`G:/pureskill/csds-cs2`, `C:/Users/<user>/...`, `/home/sagemaker-user/...`). Adjust them before running elsewhere.

## Licence

MIT (see `LICENSE`).
