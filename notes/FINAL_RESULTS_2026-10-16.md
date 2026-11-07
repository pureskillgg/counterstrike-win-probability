# Final results, 2026-10-16

Supersedes the model and metric sections of `DELIVERABLE_2026-10-16.md`. Everything below was produced in this session. Split: the current manifest `data/combined/split_manifest.json`, sha256 starts `a17f219b`. Every second is used. Log loss is per second, with 95% match-cluster intervals.

## 1. What ships

| Role | Model | Inputs | Test log loss | 95% interval | Test matches | Servable from the DSDK tome |
|---|---|---|---|---|---|---|
| **Served default** (`win_probability(data)`) | `winprob_v4_basic` | 13 event inputs | 0.50220 | [0.49980, 0.50459] | 3,601 | Yes |
| Best model | `winprob_v4` | 59 snapshot inputs | 0.45641 | [0.45333, 0.45914] | 3,059 (2026) | **No**: needs per-tick player status, which the 2026 tome lacks |

- `winprob_v4_basic` is the default now. The old default, `winprob_v3_basic`, is still in `models/` but is no longer the default, and its metadata cites an out-of-date manifest.
- Parity: `win_probability(model_name="winprob_v4_basic")` matches the same booster applied to the grid rows to **0.0001 percentage points** on three test matches (`scratchpad/parity_v4.py`).
- The old parity test still passes against the v3 export (0.0001 pp), with the model pinned to `winprob_v3_basic`.

## 2. Final model settings

Both final models use the baseline settings: depth 4, eta 0.05, min child weight 20, subsample 0.8, early stopping after 100 rounds, monotone alive counts, seed 0, CUDA on the local RTX 3080.

| Model | Train | Validation | Test | Best iteration | Validation log loss | Test AUC | Test ECE (10 bins) |
|---|---|---|---|---|---|---|---|
| `winprob_v4` | 10,577 matches / 13,758,901 seconds (2026) | 1,168 / 1,515,228 | 3,059 / 3,956,374 | 1,051 | 0.45369 | 0.8578 | 0.0032 |
| `winprob_v4_basic` | 12,509 / 15,933,894 (2024 + 2026) | 1,383 / 1,757,999 | 3,601 / 4,568,043 | 443 | 0.49962 | 0.8142 | 0.0043 |

The basic model's train, validation and test counts are smaller than the manifest's (12,563 / 1,389 / 3,611): matches with no usable rounds are dropped by `drop_roster_defect_rounds`. A CPU run of the same settings gave test 0.50224, so the device has no effect on the result.

## 3. Tuning (40 random trials + baseline, validation only)

Run on the local GPU, training every 3rd second of the 2026 training matches. The rule was fixed before the search: adopt the best-validation setting only if it beats the baseline by at least 0.0005.

- Baseline validation log loss 0.45584. Best tuned 0.45569, a gain of 0.00015, which is below the bar. **Baseline kept.**
- On test, the top three tuned settings are no better: deltas −0.00004 [−0.00017, +0.00009], −0.00008 [−0.00022, +0.00007], and −0.00015 [−0.00025, −0.00004]. The third's interval excludes zero by a sliver, but it was not the validation pick and it is smaller than the bar. Not adopted.

## 4. Positions and the every-n-seconds sweep (both on the local GPU)

Both were re-run on the local RTX 3080. The earlier runs on the pod's CPU agree with these to within 0.0001 and are kept as cross-checks (`*_pod_cpu.json`).

**Positions by round type** (full test set, 3,956,374 seconds; `exp_positions_groups.json`, device cuda)

| Group | Base (59) | With positions (83) | Difference | 95% interval |
|---|---|---|---|---|
| Overall | 0.45654 | 0.45326 | **−0.00328** | [−0.00358, −0.00296] |
| Pistol (rounds 1, 13) | 0.52038 | 0.51770 | −0.00267 | [−0.00378, −0.00165] |
| After pistol (2, 14) | 0.31462 | 0.31218 | −0.00244 | [−0.00332, −0.00158] |
| Rest | 0.46450 | 0.46109 | −0.00341 | [−0.00373, −0.00308] |

Positions help in every round group. They are not servable: the tome carries no per-tick positions. The GPU run used the memory-mapped script `scripts/model/exp_positions_np.py`; the polars script `exp_positions_groups.py` ran out of memory locally.

**Seconds per round** (`exp_tick_sampling.json`, device cuda, 2 seeds each, all test seconds)

| Seconds per round | Training seconds | Test log loss | vs every second |
|---|---|---|---|
| 1 | 201,592 | 0.45879 | +0.00232 |
| 2 | 403,184 | 0.45824 | +0.00177 |
| 4 | 806,367 | 0.45795 | +0.00148 |
| 8 | 1,612,731 | 0.45780 | +0.00134 |
| 16 | 3,225,346 | 0.45766 | +0.00119 |
| 32 | 6,393,284 | 0.45753 | +0.00106 |
| 64 | 11,261,263 | 0.45675 | +0.00029 |
| every second | 13,758,901 | 0.45647 | 0 |

This table is from the 2026-10-17 rerun, which also scores validation (section 4c-validation). The earlier run's test figures differ by at most 0.00006 (GPU run-to-run variation); the old file is kept as `exp_tick_sampling_test_only_old.json`.

More seconds per round help, and the curve is flat near 64. Use every second; sampling buys nothing.

## 4b. Grid resolution: one second against a quarter second

Basic model (13 inputs), trained on one-second rows or on quarter-second rows (89,657,037 rows; the quarter-second grid is built from the same rounds and deaths). Both models are scored on the same quarter-second validation and test rows. Rule, fixed before the run: adopt the quarter-second grid only if its validation log loss is at least 0.0005 lower.

| Set | One second | Quarter second | Difference | 95% interval |
|---|---|---|---|---|
| Validation (1,389 matches) | 0.49851 | 0.49854 | +0.00003 | [0.00000, 0.00006] |
| Test (3,611 matches, reported only) | 0.50110 | 0.50112 | +0.00001 | [−0.00001, 0.00003] |

**Decision: one-second grid kept.** A finer grid adds rows without adding information, which matches the every-n-seconds sweep in section 4.

The first run of this comparison reported a quarter-second loss of 1.10. That was a bug in the comparison script, not the model: the two models were read in different row orders. The fix sorts every read by match and tick, and the script now asserts that both models see identical rows. The one-second rebuild matches `grid.parquet` exactly on rows, alive counts, bomb state, time and labels.

## 4c. Seed variance, early stopping, monotone constraints (local GPU)

All fits use the split above; only the stated setting changes. Log loss per second, all test seconds unless marked validation.

**Seed variance** (XGBoost seed, which sets the row subsampling)

| Model | Seed | Validation | Test |
|---|---|---|---|
| `winprob_v4` (59 inputs) | 0 | 0.45369 | 0.45641 |
| `winprob_v4_s1` | 1 | 0.45359 | 0.45647 |
| `winprob_v4_s2` | 2 | 0.45359 | 0.45651 |
| `winprob_v4_basic` (13 inputs) | 0 | 0.49962 | 0.50220 |
| `winprob_v4_basic_s1` | 1 | 0.49963 | 0.50221 |
| `winprob_v4_basic_s2` | 2 | 0.49960 | 0.50219 |

Spread over three seeds: 59 inputs, 0.00010 on validation and 0.00010 on test. 13 inputs, 0.00003 on validation and 0.00002 on test. Seed noise is far below the differences the choices were judged on.

**Early stopping** (patience on validation log loss; default 100)

| Model | Patience | Best iteration | Validation | Test |
|---|---|---|---|---|
| `winprob_v4` | 50 | 882 | 0.45371 | 0.45645 |
| `winprob_v4` | 100 (default) | 1,051 | 0.45369 | 0.45641 |
| `winprob_v4` | 200 | 1,051 | 0.45369 | 0.45641 |
| `winprob_v4_basic` | 50, 100, 200 | 443 in all three | 0.49962 | 0.50220 |

Patience 50 costs 0.00002 on validation for the 59-input model. The 13-input model stops at the same iteration under all three settings.

**Monotone constraints on the alive counts** (paired on the same seconds; `scripts/model/exp_mono_paired.py`, `data/combined/exp_mono_paired.json`). The difference is log loss without the constraints minus with them, so a negative value means dropping them is better. Intervals resample matches.

| Model | Split | Difference | 95% interval | Matches |
|---|---|---|---|---|
| `winprob_v4_basic` (13 inputs) | validation | −0.00019 | [−0.00036, −0.00002] | 1,383 |
| `winprob_v4_basic` (13 inputs) | test | −0.00044 | [−0.00059, −0.00031] | 3,601 |
| `winprob_v4` (59 inputs) | validation | −0.00006 | not computed (single fits: 0.45363 vs 0.45369) | 1,168 |
| `winprob_v4` (59 inputs) | test | +0.00009 | not computed (single fits: 0.45650 vs 0.45641) | 3,059 |

The recomputed test log losses from the saved boosters match their meta files exactly (0.502198 and 0.501755). For the served 13-input model, dropping the constraints lowers log loss on both splits, and both intervals exclude zero. For the 59-input model the two are within 0.0001.

**Open decision: keep or drop the monotone constraints in the served model.** It is still served with them. The earlier note in this session applied the 0.0005 bar, which was set for tuning, to this choice after seeing the result, so it is not a pre-set test. Dropping them gains 0.00044 on test. Keeping them guarantees that more CT players alive never lowers CT's chance, and that more T players alive never raises it. That guarantee is worth more than 0.00044 only if someone reads the output as a probability curve, so the choice is the user's.

## 4e. Close-out checks (2026-10-17, `scripts/model/exp_closeout_checks.py`, `data/combined/closeout_checks.json`)

- **Like-for-like.** On the same 3,059 2026 test matches, the 13-input model scores 0.49950 [0.49688, 0.50204] and the 59-input model 0.45641 [0.45333, 0.45914]. Gap 0.04309. Not a paired interval, but the intervals do not overlap. So the two models' headline numbers differ by about 0.043 on the same matches; the earlier "different test sets" caveat was about the match counts, not the size of the gap.
- **The 115-to-116-second band.** 94 test rows (79 matches) have no bomb planted and sit between 115 and 116 seconds of live play. All 94 ended with a CT win, and the model's mean prediction there is 0.96. Dropping them changes test log loss from 0.50220 to 0.50221, so the extra second does not move the score. The same check finds 17 unplanted test rows past 116 seconds. Sixteen belong to six rounds planted after 116 seconds: the cap applies only to rounds with no plant recorded, so those rounds keep their pre-plant rows (`data/combined/exp_cap_trace.json`). One row, in a round recorded as never planted, is not explained.
- **Matches that differ between sources.** The manifest has 17,563 matches, and every one has grid rows. The 2026 snapshot cache has 14,804, all of them in the manifest. The grid's 2026 era has 14,814: the 14,804 plus 10 that the cache lacks (7 train, 1 validation, 2 test). Every round of those 10 has a missing player, and the roster-defect rule drops all of them, so the cache build skips them and the 13-input model drops them too (`data/combined/exp_cache_skip_trace.json`).

## 4h. The winning model: score lead and monotone constraints, on the 59 inputs (2026-10-17, local GPU)

Same split, same settings, same test seconds as `winprob_v4`. The reference is the saved `winprob_v4` fit, reloaded and scored on those rows; its test figure reproduces the recorded 0.456412.

- **Score lead added (`scripts/model/exp_59_scorelead.py`, `data/combined/exp_59_scorelead.json`):** test −0.00070 [−0.00101, −0.00040]; validation −0.00094 [−0.00144, −0.00047]. It still helps on top of the 59 inputs, by less than on the 13-input model (−0.00094).
- **Monotone constraints dropped (`scripts/model/exp_59_nomono.py`, `data/combined/exp_59_nomono.json`):** test +0.00009 [+0.00001, +0.00018]; validation −0.00006 [−0.00018, +0.00007]. The two splits disagree, and both effects are about 0.0001. Keeping the constraints costs nothing measurable, and they guarantee the monotone behaviour, so they stay.
- **Final model saved (`scripts/model/train_final_59_scorelead.py`, `models/winprob_v5_59_scorelead.json` and `.meta.json`):** the 59 inputs plus score lead, with the monotone constraints on the alive counts, the baseline settings, the frozen split, early stopping on validation, seed 0. It is not the served default. The served function reads only event channels, and the 59 inputs need per-tick player status.

## 4g. Roster-defect drop, sensitivity (2026-10-17, `scripts/model/exp_roster_sensitivity.py`, `data/combined/exp_roster_sensitivity.json`)

Rounds with a missing player: 3,841 of 329,028 rounds across both eras (about 1.2%, not the 0.4% the earlier notes gave). Training the served 13-input model with them included (CPU, same settings, early stopping on the clean validation seconds) and scoring both models on the same clean test seconds: the all-round model is worse by 0.00004 [+0.00002, +0.00006]. The drop helps slightly, and it is kept.

## 4f. Score lead as a 14th input (2026-10-17, `scripts/model/exp_scorelead.py`, `data/combined/exp_scorelead.json`)

The grid carries `score_lead_diag` (CT score minus T score at the freeze end) as a diagnostic column, not a model input. Added to the 13 inputs with the served model's settings (CPU, best iteration 498, monotone constraints on the alive counts; score lead unconstrained) and scored on the same seconds as the served model:

- Test: 0.50126 [0.49883, 0.50365], against 0.50220 for the served model. Difference −0.00094 [−0.00124, −0.00065].
- Validation: 0.49863, against 0.49962. Difference −0.00099 [−0.00145, −0.00051].

Not adopted, pending your decision. A check on five 2026 matches (95 rounds, `scripts/serving/check_score_lead_tome.py`) found that the tome's `round_state` scores give the grid's freeze-end lead in every round, so the tome supports it. Adopting it means retraining the served model with the extra input, updating the serving function, and rerunning the parity test. The gain is real on both splits.

## 4c-validation. Sampling sweep and round-type models, validation numbers (added 2026-10-17)

Sampling sweep (`exp_tick_sampling.json`, GPU, 2 seeds each, all matches). The rule for these choices is validation first; test was reported before. Validation deltas are against every second.

| Seconds per round | Training seconds | Test log loss | Test delta | Validation log loss | Validation delta |
|---|---|---|---|---|---|
| 1 | 201,592 | 0.45879 | +0.00232 | 0.45554 | +0.00195 |
| 2 | 403,184 | 0.45824 | +0.00177 | 0.45498 | +0.00139 |
| 4 | 806,367 | 0.45795 | +0.00148 | 0.45466 | +0.00107 |
| 8 | 1,612,731 | 0.45780 | +0.00134 | 0.45459 | +0.00100 |
| 16 | 3,225,346 | 0.45766 | +0.00119 | 0.45451 | +0.00092 |
| 32 | 6,393,284 | 0.45753 | +0.00106 | 0.45425 | +0.00066 |
| 64 | 11,261,263 | 0.45675 | +0.00029 | 0.45369 | +0.00010 |
| every second | 13,758,901 | 0.45647 | 0 | 0.45359 | 0 |

Validation agrees with test: fewer seconds per round is worse at every level, and the gap closes as more seconds are used. Every second stays the choice. (The sweep's every-second row averages two seeds, so its test figure, 0.45647, is not the served model's 0.45641, which is seed 0 alone. The two differ by 0.00006, the same size as the seed spread in section 4c.)

Round-type models, validation (`exp_pistol_validation.json`), against one model for every round (M1, 0.45363):

| Model | Overall validation delta [95%] | Round group | Group delta [95%] |
|---|---|---|---|
| V1 pooled, with round-type inputs | +0.00015 [−0.00013, +0.00047] | all | — |
| V2 pistol-only, small-group settings | +0.00010 | pistol (1, 13) | +0.00128 [−0.00022, +0.00281] |
| V2b after-pistol-only, small-group settings | −0.00005 | after pistol (2, 14) | −0.00066 [−0.00273, +0.00129] |

None of the three gains on validation, and every interval includes zero. The decision to use one model for all rounds holds. The after-pistol model's test result (+0.00251, section 7 of the methodology) is not confirmed on validation (−0.00066), so both numbers are noise at these group sizes.

## 4d. Model families, validation and test (same split, every second; `data/combined/exp_full_comparison.json`)

Device column: the families on GPU are CatBoost (its own GPU mode), a PyTorch MLP, and cuML logistic regression and random forest. Everything else ran on CPU. XGBoost is the saved GPU fit `winprob_v4`, reloaded and scored on the same rows. Deltas are test log loss minus XGBoost's, with match-cluster intervals.

| Model | Device | Validation | Test | Test minus XGBoost [95%] | Notes |
|---|---|---|---|---|---|
| XGBoost (`winprob_v4`) | GPU | 0.45369 | 0.45641 | — | served family |
| CatBoost | CPU | 0.45373 | 0.45675 | +0.00034 [+0.00015, +0.00054] | monotone constraints on |
| CatBoost | GPU | 0.45354 | 0.45681 | +0.00040 [+0.00023, +0.00058] | **no monotone constraints** (the GPU refused them), so not like-for-like |
| LightGBM | CPU | 0.45387 | 0.45678 | +0.00037 [+0.00024, +0.00052] | |
| Histogram boosting | CPU | 0.45403 | 0.45691 | +0.00049 [+0.00033, +0.00066] | 600 iterations, no early stopping |
| Random forest (sklearn) | CPU | 0.45786 | 0.46034 | +0.00393 [+0.00356, +0.00430] | 10% of training rows |
| Random forest (cuML) | GPU | 0.45781 | 0.46059 | +0.00418 [+0.00379, +0.00460] | quantile-binned splits, max depth 16 (cuML default): a different algorithm |
| ExtraTrees | CPU | 0.45904 | 0.46168 | +0.00527 [+0.00483, +0.00576] | 10% of training rows |
| MLP (sklearn) | CPU | 0.46083 | 0.46407 | +0.00766 [+0.00696, +0.00845] | 25% of training rows |
| MLP (PyTorch) | GPU | 0.45900 | 0.46252 | +0.00610 [+0.00539, +0.00682] | same architecture; sklearn's L2 penalty not replicated |
| Logistic regression (sklearn) | CPU | 0.46373 | 0.46741 | +0.01100 [+0.01042, +0.01159] | |
| Logistic regression (cuML) | GPU | 0.46373 | 0.46742 | +0.01101 [+0.01042, +0.01160] | |

On validation, the boosting models sit within 0.0003 of each other. None beats XGBoost by 0.0005, the bar used for the tuning run (section 3); this comparison has no bar of its own. The family choice holds on validation and on test. The CPU-only families (LightGBM, histogram boosting, ExtraTrees) stayed on CPU by choice. Where a family has both a CPU and a GPU row, the GPU version is a different implementation (CatBoost without monotone constraints, a PyTorch MLP, cuML's random forest), so its difference from the CPU row is about the algorithm, not the hardware.

## 5. Tests

- DSDK (`incremental-tomes`, commit `e8d0e24`): **372 passed**, exit 0 (`logs/dsdk_pytest.log`).

## 6. Not done, and why

- **Email of the models.** Not sent. It needs your explicit go, per send. This is the only item left that is not training or running.
- **RunPod as a GPU backup.** Every pod I launched (five in all) landed on one of two hosts (`64.119.209.250` and `206.41.93.58`) where CUDA initialisation fails. Each was terminated after its check. No RunPod work ran on GPU, and the local GPU did all the GPU work.
- **Pod CPU runs.** The first positions run and the first sweep ran on the pod's CPU before the GPU rule was applied. They were re-run on the local GPU (section 4). The CPU results are kept as `*_pod_cpu.json`. An early CPU copy of the basic model and an unfinished CPU run of the 59-input model were also made; the reported models are the GPU ones.
- **Hyperparameter search on the basic model.** The search covered the 59-input model only. The basic model keeps the baseline settings.

## 7. Known differences and caveats

- The served function scores some rounds the grid drops (rounds with four or five players on a side). It warns about them in `last_warnings`, so on those matches its tick count is larger than the grid's (e.g. 950 ticks against 151).
- Earlier figures (0.45641 in the full comparison, 0.50289 on 3,611 matches for the old basic model) came from different splits or stride settings. The table above is the one to quote.
- Every manifest match has grid rows. The 13-input model's counts are lower because 70 matches (54 train, 6 validation, 10 test) have no rounds left after the roster-defect drop. Ten 2026 matches are in the grid but not in the 2026 snapshot cache (7 train, 1 validation, 2 test). Every round of each one has a missing player, so the roster-defect rule drops them; neither model uses them.

## 8. Provenance (sha256, first 16 hex characters)

| File | sha256 |
|---|---|
| `models/winprob_v4.json` | b7f666a8c769cd20 |
| `models/winprob_v4.meta.json` | 3fc4155a745a033c |
| `models/winprob_v4_basic.json` | 65a38bd16f180859 |
| `models/winprob_v4_basic.meta.json` | e20d27161d70eb37 |
| `data/combined/exp_positions_groups.json` (GPU) | 75db4cd9b199dd3b |
| `data/combined/exp_positions_groups_pod_cpu.json` (pod CPU cross-check) | 2c6e59188d12cef6 |
| `data/combined/exp_tick_sampling.json` (GPU) | 7051daa337ec0249 |
| `data/combined/exp_tick_sampling_pod_cpu.json` (pod CPU cross-check) | dbe1755d769d8e58 |
| `data/combined/exp_tune_xgb.json` | bae0cf111ea522aa |
| `data/combined/exp_grid_step.json` | f19327b5377f5a5f |
| `scripts/model/train_final.py` | da0d9602dc2baeb5 |
| `scripts/model/exp_tune_xgb.py` | f9e56f108a5fa9b2 |
| `scripts/model/exp_positions_np.py` (GPU positions) | a819a0e10e2d53dc |
| `scripts/model/exp_positions_groups.py` (pod / polars version) | 8ee59b456f630b86 |
| `scripts/model/exp_tick_sampling.py` | 5124ca92fee80fdd |
| `scripts/model/grid_compare.py` | b7fcc8f714155d8c |
| `scripts/data/build_grid_step.py` | 21c833f7c9e90a3c |
| `scripts/model/wp_common.py` | dc478d29a1e81e45 |
| `scripts/serving/win_probability.py` | 879c8ac2c22f5a70 |
| `data/combined/split_manifest.json` (manifest the models were trained on) | a17f219b5287ff1e |
