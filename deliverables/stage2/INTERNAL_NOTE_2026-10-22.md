# Stage 2 internal note, 2026-10-22: round-state embedding and temporal models (de_mirage)

Status: internal. Nothing has been sent outside the team. Authors of the Szmida & Toka framework have not been contacted.

## What was built

- **Tomes 14 to 19 October** are complete: 8,485 matches in 20 chunks, each with BUILD_OK, and every day has CHUNKS_DONE. They are not in the combined store yet. Adding them changes the frozen split and needs owner approval.
- **de_mirage state dataset**: 4,342,609 per-second states from 43 day files, ten player slots of six values each. Split by the frozen manifest: 2,455 / 274 / 687 matches in train / val / test. Window fitted on train rows only. 246 recording-gap rows dropped. 2,004 rows (28 round pairs) have no snapshot state and are excluded from training and evaluation.
- **Loadout features**: per slot, health, armour, helmet, defuser, money, equipment value, defusing, and grenade flags, from `status_1hz`. 30,329,406 slot values written. No position row was missing its status.

## Results

| Model | Test log loss (de_mirage, 687 matches, 865,974 seconds) | Notes |
|---|---|---|
| 57-input XGBoost baseline (winprob_v6_drop_heavy) | 0.4512 (CI 0.4454 to 0.4574) | Frozen split; the reference |
| Temporal: set encoder + GRU, positions and state only | 0.4855 (CI 0.4797 to 0.4913) | 3 epochs |
| Temporal: set encoder + GRU, with loadout | 0.4524 (CI 0.4468 to 0.4584) | 3 epochs |
| Temporal: attention over players + GRU, with loadout | 0.4527 (CI 0.4470 to 0.4587) | 3 epochs |

**Paired comparison, same seconds** (`data/temporal/paired_de_mirage_gat_loadout_gru_vs_57_de_mirage.json`): the attention-plus-GRU model minus the baseline is +0.0015 log loss, 95% interval [+0.0005, +0.0026], on 865,974 paired seconds from 687 matches. The interval excludes zero, so the temporal model is worse on these seconds. The baseline reproduces its earlier score exactly (0.4512), which confirms the pairing.

**Learned positions-only embedding** (`scripts/model/train_round_state_encoder.py`, 3-second next-state prediction, change-based decoder): validation MSE 0.0150 against 0.0203 for persistence (repeating the current positions), a 26% reduction. Its drawn picture separates early from late rounds and low from high alive counts. It does not separate round winner, which is expected because the winner is not a training target. Across four nearest-neighbour queries (5 neighbours each), the neighbours share the round winner in 15 of 20 cases, about 75%, against roughly 50% by chance. That is a signal, but from four queries only, so it needs a larger check. The neighbours share the alive counts in just 1 of 20 cases, so the embedding does not capture alive count the way the situation check would expect.

## What this means

- No temporal model has beaten the 57-input baseline yet. The loadout features closed most of the gap. Attention over players added nothing at three epochs.
- The embedding is a real positions-only result. It is not yet a win-probability result, and its neighbour check is weak.
- The bar from the survey stands: a temporal model has to beat calibrated logistic regression on out-of-time data before it is worth more than a note.

## Open items for the owner

1. Approve appending tomes 14 to 19 October to the combined store, and the manifest rule for their splits.
2. Choose the next step: add the remaining 57-input features (team economy, bomb timing) and train longer, or stop the temporal line here.
3. Confirm the Zotero subfolder for the stage 2 papers.
4. Stage 1 wording: the deliverable says parity was checked on every October test match. It was checked on 294 of about 2,610. The served-flag metadata for the default model says `served: false`.
5. Calibration of the submitted model has not been reported. It can be computed from the saved predictions without retraining.

## Sources

- Baseline: `data/combined/export_winprob_v6_drop_heavy_test_predictions.parquet`, `data/combined/internal_baseline_de_mirage_test.json`
- Temporal metrics: `data/temporal/de_mirage_gru_metrics.json`, `data/temporal/de_mirage_gru_loadout_metrics.json`, `data/temporal/de_mirage_gat_loadout_gru_metrics.json`
- Paired comparison: `data/temporal/paired_de_mirage_gat_loadout_gru_vs_57_de_mirage.json`
- Encoder: `data/round_state/de_mirage_h3_metrics.json`, `deliverables/stage2/ROUND_STATE_CHECKS_DE_MIRAGE_LEARNED_H3.txt`
- Papers: `deliverables/stage2/ARCHITECTURES_2026-10-22.md`, `deliverables/stage2/TEMPORAL_AND_FOUNDATION_2026-10-22.md`
