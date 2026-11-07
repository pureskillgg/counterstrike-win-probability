# Round win probability for CS2: deliverable summary (2026-10-19, generated)

## Submission (decided 19 October)

`winprob_v6_drop_heavy`: 57 inputs (59 minus the heavy-weapon group), no score lead, one seed, all round types. Validation 0.453557 [0.448692, 0.458152] on 1,168 matches; test 0.456411 [0.453311, 0.459138] on 3,059 matches (`data/combined/exp_submission_57.json`). Against the score-lead comparison model on the same test seconds: +0.000806 [+0.000508, +0.001105]. Served: no (`models/winprob_v6_drop_heavy.meta.json`, served false); serving parity was checked on 19 October (294 October test matches in the 1 to 8 October tome; maximum gap 0.0001 percentage points; `logs/test_win_probability_v6_drop_heavy_2026-10-19.log`). Needs per-tick player status. The score-lead check is in `notes/METHODOLOGY_2026-10-19.md`, section 1. In this note, "production model" below means the score-lead comparison model, the served default; it is not the submission.

## Production model

`winprob_v8_ensemble`: the mean of five XGBoost seeds on 58 inputs. It was the model the serving function used by default until 19 October; the function's default is now the 57-input submission. Validation is used for every choice; test is scored once.

- Validation log loss 0.452722; test log loss 0.455605 on 3,956,374 seconds of the 2026 test matches.
- Seed-0 single model: validation 0.452778, test 0.455680. The five-seed average is  [-0.00013, -0.00002] better on test than seed 0 alone.
- Serving: `scripts/serving/win_probability.py`. The check below was run before the default changed on 19 October. Parity against the grid on every October test match: 294 of 294 matches, 374,500 seconds, largest gap 0.0001 percentage points, 100% of seconds within 0.01 points. Score lead matches the grid in 5,431 of 5,431 rounds.
- Needs per-tick player status in the DSDK tome. A tome without it returns no probabilities (no silent fallback).

### Inputs (58)

- Event inputs (13): alive counts for each side, their difference, bomb planted, seconds since round live, seconds since plant, and the map (one-hot, 7 maps).
- Score lead (1): freeze-end CT score minus T score (round_state; known before the round starts).
- Loadout inputs (44): per side, team health, armour (players with armour, armour points), helmets, defuse kits, defusing, weapon classes (AK, M4, AWP, other rifles, other snipers, SMG, shotgun and machine gun counts are dropped; see results), players with no primary, grenades, equipment value; and the CT-minus-T differences of health, rifles, equipment, armour, AWP and SMG. Taken from the per-tick player status at the same tick.
- Monotone constraints: alive CT +1, alive T -1, alive difference +1; everything else unconstrained (score lead unconstrained).

### Settings

XGBoost: depth 4, learning rate 0.05, min child weight 20, subsample 0.8, early stopping after 100 rounds on validation, GPU, seeds 0 to 4 averaged. Every second of each round, roster-defect and status-inconsistent rounds dropped (in the cache).

## Frozen split

`data/combined/split_manifest.json` (file sha256 a17f219b5287, the hash in every model's metadata): random by match, seed 0, all 17,563 matches: train 12,563, validation 1,389, test 3,611. The 2026 models use the 2026 subset (10,577 / 1,168 / 3,059 matches after the cache drops 10 untraced matches).

Example match and its first round's live ticks in each split (first match in each split, sorted by key):

| Split | Match key | Era | Round | First tick | Last tick | Seconds |
|---|---|---|---:|---:|---:|---:|
| train | `csds/2024/02/01/m_9805aa614020949e127f8783/csds` | 2024 | 2 | 6798 | 9294 | 40 |
| val | `csds/2024/02/01/m_a86189876e2b9990b7dd6efa/csds` | 2024 | 2 | 10481 | 13745 | 52 |
| test | `csds/2024/02/01/m_7091cd950390b7f8e8dc280d/csds` | 2024 | 2 | 5295 | 7599 | 37 |

Ticks step by 64 from the freeze end (one per second). Matches and ticks in every split are recorded in the manifest and the grid.

## Results (per-second log loss)

| Model | Inputs | Validation | Test | Notes |
|---|---:|---:|---:|---|
| Score-lead comparison: five-seed ensemble | 58 | 0.452722 | 0.455605 | served by default until 19 October |
| `winprob_v7_59_drop_heavy_scorelead` | 58 | 0.452778 | 0.455680 | single seed |
| `winprob_v5_59_scorelead` | 60 | 0.452750 | 0.455712 | with heavy group, score lead |
| `winprob_v4` | 59 | 0.453695 | 0.456412 | loadout reference |
| `winprob_v6_drop_heavy` | 57 | 0.453557 | 0.456411 | no score lead |
| `winprob_v7_basic_scorelead` | 14 | 0.498608 | 0.501242 | events and score lead only, all matches |
| `winprob_v4_basic` | 13 | 0.499617 | 0.502198 | previous served, all matches |

Scores on the 2026 test seconds (3,956,374) are comparable within the 58/59-input group; the 13/14-input models are scored on all matches (4,568,043 seconds) and are comparable only with each other.

## Everything that was run, and what it led to

### Loadout ablation (one group removed; positive = removal hurts)

| Removed | Validation change | Test change [95% CI] |
|---|---:|---:|
| heavy (shotguns, machine guns) | -0.00009 | -0.00002 [-0.00010, +0.00005] |
| armor | 0.00058 | 0.00069 [+0.00046, +0.00094] |
| ak + m4 split (keeps all rifles) | -0.00011 | 0.00019 [+0.00009, +0.00029] |
| all rifles (incl. ak, m4) | -0.00008 | 0.00011 [-0.00003, +0.00024] |
| awp | -0.00001 | 0.00007 [-0.00001, +0.00015] |
| smg | 0.00007 | 0.00005 [-0.00006, +0.00017] |
| other snipers (scout, autos) | 0.00007 | 0.00005 [-0.00004, +0.00014] |
| no primary (pistol-only players) | -0.00003 | 0.00007 [-0.00001, +0.00013] |
| health | 0.00425 | 0.00463 [+0.00407, +0.00523] |
| equipment value | 0.00001 | 0.00013 [-0.00001, +0.00028] |
| grenades (flash, smoke, he, fire, decoy) | -0.00000 | 0.00008 [-0.00003, +0.00018] |
| defuse kits, defusing, helmets | 0.00180 | 0.00160 [+0.00141, +0.00180] |

- Joint removal of the 32 redundant loadout features (27 inputs): validation 0.00300, test 0.00286; rejected.
- Block drop weapon types + equipment (22): validation 0.00260, test 0.00270.
- Block drop weapon types + grenades (29): validation 0.00059, test 0.00039.

### Score lead
- + score lead (60) (seed 0): validation -0.00084, test -0.00071 [-0.00101, -0.00041].
- + score lead, drop heavy (58) (seed 0): validation -0.00087, test -0.00075 [-0.00107, -0.00046].
- + score lead (60) (seed 1): validation -0.00083, test -0.00079 [-0.00107, -0.00051].
- + score lead, drop heavy (58) (seed 1): validation -0.00093, test -0.00093 [-0.00123, -0.00061].
- Adopted (it is in the production set).

### Hyperparameters (58 inputs; eleven settings, validation decides)
- Baseline (depth 4, min child weight 20, subsample 0.8, eta 0.05): validation 0.452778, test 0.455680.
- depth 5: validation delta 0.00001, test delta -0.00002.
- depth 6: validation delta 0.00011, test delta 0.00012.
- depth 8: validation delta 0.00026, test delta 0.00033.
- min child weight 50: validation delta 0.00003, test delta 0.00011.
- min child weight 100: validation delta 0.00015, test delta 0.00014.
- subsample 1.0: validation delta 0.00014, test delta 0.00018.
- colsample 0.8: validation delta 0.00001, test delta 0.00004.
- depth 6, mcw 100: validation delta 0.00023, test delta 0.00020.
- depth 6, mcw 100, colsample 0.8: validation delta 0.00012, test delta 0.00004.
- eta 0.03: validation delta 0.00009, test delta 0.00004.
- Selected by validation: baseline d4 mcw20 sub0.8 (baseline settings kept).

### Seed averaging: five seeds average test 0.455605 against 0.455680 for seed 0 (-0.00008 [-0.00013, -0.00002]). Adopted.

### Event features (cumulative damage and shots per side in the round): validation -0.00002, test 0.00016 [+0.00002, +0.00029]. Rejected.

### Pistol-round split
- pistol rounds (1, 13): dedicated minus main 0.00110 [+0.00011, +0.00211].
- round after pistol (2, 14): dedicated minus main 0.00175 [+0.00013, +0.00325].
- Overall test with the split 0.00022 [+0.00007, +0.00037]. Rejected.

### Model families (58 inputs, every third training second)
- xgboost: validation 0.452831, test 0.455783.
- catboost (GPU, no monotone constraints): validation 0.452827, test 0.455882 (test delta vs XGBoost 0.00010 [-0.00007, +0.00027]).
- lightgbm: validation 0.452717, test 0.455712 (test delta vs XGBoost -0.00007 [-0.00016, +0.00002]).
- XGBoost kept: it is the family the production model uses, and it is the reference the others are compared with.

### Model families, full training set (13,758,901 seconds, 59 inputs, every second)
- xgb: test 0.456412 [0.45333, 0.45914], validation 0.453695.
- lgbm: test 0.456782 [0.45363, 0.45951], validation 0.453869 (test vs XGBoost +0.000370 [+0.000237, +0.000523], paired).
- catboost: test 0.456754 [0.45364, 0.45943], validation 0.453731 (test vs XGBoost +0.000342 [+0.000149, +0.000542], paired).
- catboost_gpu: test 0.456814 [0.45379, 0.45956], validation 0.453545 (test vs XGBoost +0.000402 [+0.000231, +0.000577], paired).
- hgb: test 0.456905 [0.45378, 0.45963], validation 0.454028 (test vs XGBoost +0.000493 [+0.000329, +0.000656], paired).
- rf: test 0.460339 [0.45726, 0.46301], validation 0.457856 (test vs XGBoost +0.003926 [+0.003564, +0.004299], paired).
- et: test 0.461680 [0.45880, 0.46424], validation 0.459040 (test vs XGBoost +0.005268 [+0.004830, +0.005762], paired).
- mlp: test 0.464069 [0.46080, 0.46693], validation 0.460831 (test vs XGBoost +0.007656 [+0.006963, +0.008448], paired).
- logit: test 0.467410 [0.46429, 0.47036], validation 0.463729 (test vs XGBoost +0.010998 [+0.010421, +0.011591], paired).
- Settings in this design were not tuned to the production settings (see the source script exp_full_comparison.py).

### Ticks per second (58 inputs, 2,644 training matches, within each second)
- Every second: test 0.457776.
- 1 per second (1.00 of the every-second rows): test 0.457743, delta -0.00003 [-0.00014, +0.00007].
- 2 per second (2.00 of the every-second rows): test 0.457737, delta -0.00004 [-0.00015, +0.00007].
- 4 per second (3.99 of the every-second rows): test 0.457740, delta -0.00004 [-0.00014, +0.00007].
- 8 per second (7.95 of the every-second rows): test 0.457772, delta -0.00000 [-0.00011, +0.00010].
- No gain from more than one tick per second. Every second kept.

### Random draws per round (1 to 64 per round, 59 inputs)
- s0_n64: 0.94 of the every-second rows, test delta 0.00131 [+0.00095, +0.00164].
- s1_n64: 0.94 of the every-second rows, test delta 0.00118 [+0.00087, +0.00150].
- s0_n1: 0.01 of the every-second rows, test delta 0.00255 [+0.00218, +0.00290].
- Leakage canary (a feature five seconds in the future): test 0.441276. It is far better than any real model, so the metric would show leakage if there were any.
- Same draws weighted by round length / 64: test 0.456572 against 0.457688 unweighted (-0.00112 [-0.00144, -0.00079]). Round weighting explains most of the random-draw gap.
- Random draws are not used: every second is the training set.

## Serving and checks

- Serving function reads the DSDK tome channels as they are (round_state, round_end, player_info, player_death, bomb_state, header, player_status).
- Split verified: disjoint sets, every grid match assigned.
- Every saved model re-derived from its booster on the frozen split (validation and test match the metadata to 1e-6).
- Score lead equals the grid's in every checked round (5,431 of 5,431 in the tome window).
- DSDK incremental tome changes: crash-consistency fixes (channels before the header, rewrite staging files cleaned up, header rows before the keyset), with tests; the full suite result is in logs/dsdk_fork_full_tests.log.

## Not in the model, and why

- Positions: left for a later model, as instructed.
- Facing direction, smokes, fire, blindness: not tested; left out.
- Deep models: next week, as planned.
- 16 per second: not built; the finest tested rate is 8 per second, and it showed no gain.

## Not done without your go-ahead

- Emailing the models: each send needs explicit permission.
- Publishing the deck (the local HTML is the draft to review).

