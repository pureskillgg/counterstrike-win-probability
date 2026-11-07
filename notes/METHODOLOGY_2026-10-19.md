# Round win probability for CS2 competitive matches: methodology

Written 2026-10-19 for the project owner and for a reviewer who has not seen the work. It states what the models read, what they were trained on, which choices were made and on what evidence, and what is still open. Paths are relative to `G:\pureskill\csds-cs2`. Every number is followed by the file that holds it. Where a number is computed from a file rather than stored in one, the computation is named. Intervals are 95% match-cluster bootstrap intervals (1,000 resamples, seed 0) unless stated otherwise. Log loss is per second of live play; lower is better. A "delta" is a paired difference on the same seconds.

## 1 Summary

**Submission decision (19 October, meeting).** The submission model is the 57-input model `winprob_v6_drop_heavy`: the 59-input set minus the heavy-weapon group, with no score lead, trained on all round types, one seed. Its numbers are validation 0.453557 [0.448692, 0.458152] and test 0.456411 [0.453311, 0.459138] on 3,059 matches (`data/combined/exp_submission_57.json`). Against the score-lead production model on the same test seconds it is worse by +0.000806 [+0.000508, +0.001105]. The score-lead model described in the rest of this document is the comparison, not the submission. The submitted model needs per-tick player status, as the production model does, and serving parity was checked on 19 October: the scoring function reproduces the grid's predictions on the 294 October test matches in the 1 to 8 October tome, with a maximum gap of 0.0001 percentage points (`logs/test_win_probability_v6_drop_heavy_2026-10-19.log`).

**Score-lead check (19 October).** On the 2026 grid there are 259,948 consecutive round pairs (14,814 matches; 282,284 rounds), and the score lead changes in every pair. In the 245,136 pairs with no side swap, the change equals the previous round's CT-win sign in every pair; the current round's share there (62.91%) equals how often the same side wins two rounds in a row. Over all pairs the previous-round share is 96.79% and the current-round share is 62.22%. The earlier quick-check figures (95.13%, 0.37%, 59.93%, 35.58%) did not reproduce and are withdrawn. This is consistent with the score lead being the running score of earlier rounds; it does not show that the score lead contains the current round's result. The reason given at the meeting for leaving the score lead out was round leakage; that reason needs owner confirmation before it is repeated. Sources: `data/combined/exp_scorelead_leak.json`, `scripts/model/exp_scorelead_leak.py`.

**Open before sending:** the owner's reason for leaving out the score lead; the origin of the day 14 and 15 folders on the E: drive; serving parity for the submission was checked on the 294 test matches of the 1 to 8 October tome; coverage of the other October test matches is not shown in the record.

In this document, "production model" means the score-lead comparison model (58 inputs). Until 19 October it was the served default; since then the scoring function's default is the submission (57 inputs), and the comparison is passed by name.

The model reads a live competitive 5v5 round once per second and returns the probability that CT wins the round. T's probability is 100 minus CT's, because every round has exactly one winner.

**Production model: 58 inputs.** 13 event inputs, 1 score lead, and 44 loadout inputs. The loadout inputs describe the living players' health, armour, helmets, defuse kits, weapon classes, grenades and equipment value (section 8). The model is the mean of five XGBoost models that differ only in their random seed.

**Event-only model: 14 inputs.** The 13 event inputs and the score lead. It has no loadout inputs, so it runs on tomes without per-tick player status, including the 2024 matches.

Headline results on the same test seconds (3,059 matches, 58,059 rounds, 3,956,374 seconds):

| Model | Test log loss | Interval | Source |
|---|---|---|---|
| Production, five seeds | 0.455605 | [0.45243, 0.45841] | `data/combined/exp_seed_avg58.json`; interval recomputed from `data/combined/export_winprob_v8_ensemble_test_predictions.parquet` (section 13) |
| Production, seed 0 alone | 0.455680 | not recorded | `models/winprob_v8_ensemble.meta.json` members |
| Event-only, same seconds | 0.498695 | not recorded | `data/combined/exp_eventonly_paired.json` |
| Event-only minus production, paired | +0.043090 | [+0.041449, +0.044884] | `data/combined/exp_eventonly_paired.json` |

The event-only figure on its own test set (3,601 matches, 2024 and 2026) is 0.501242 (`models/winprob_v7_basic_scorelead.meta.json`). It is not comparable with the production figure (section 13.2).

Three things a reader should know before going further:
1. The test set was looked at many times during the work. Section 11 says which numbers were chosen on validation, and which were reported only as the final figure.
2. Dropping the heavy-weapon group was partly decided on test consistency, not validation alone (section 11).
3. Section 16 lists what is unknown or unresolved, including two decisions that are yours to make (monotone constraints in the event-only model, and whether 2024 data belong in training).

## 2 The question

At every second of live play the model answers: given the state of the round at this second, how likely is it that CT wins this round? The label is the round's final winner and is the same on every row of a round. Early rows are uncertain (probabilities near 0.5); late rows are nearly decided. The score is the average over all rows.

Scope: competitive 5v5 matches on seven maps (de_ancient, de_anubis, de_cache, de_dust2, de_inferno, de_mirage, de_nuke). Wingman, hostage and other maps are not scored (`notes/data/COMBINED_STORE.md`, section "Catalog and rounds"; `models/winprob_v3.meta.json`, scope.maps).

The T = 100 minus CT rule is checked in three places: the end-to-end check passes check D3 (every round has one constant label) and C1 (the final label is 2 or 3 on each of 329,028 used rounds) in `data/combined/e2e_winprob_v3.json`, run 2026-10-15; the label audit finds no winner code outside {2, 3} in the 2024 collection or in 2026-10-01 to 08 (`logs/swarm2/A6_label_audit.log`); and the serving function sets T to 100 minus CT (`scripts/serving/win_probability.py`).

## 3 Data and time windows

The store is one combined folder built from several collections (`notes/data/COMBINED_STORE.md`, table of collections): 2024-02-g (4,435 matches, vetted), 2026-09, 2026-10-01 to 08 (1,752 matches, vetted), and 2026-10-09 to 12 (one collection per upload day). The era of a match is the year in its upload folder (`scripts/data/prepare.py`).

| | 2024 | 2026 |
|---|---|---|
| Matches in the frozen split | 2,749 | 14,814 (`data/combined/split_manifest.json`) |
| Upload folders of those matches | 2024/02/01 to 2024/02/25 | 2026/09/01 to 2026/10/13 |
| Header match dates | not checked here | 2026-08-02 to 2026-10-13 (read from the tome header `match_date` during the claims audit of 2026-10-19) |
| Per-tick player table | none (`data/combined/eval_scope_maps_winprob_v3.json`, notes) | per-second extraction for all 14,814 matches (`data/status_1hz/`, 43 daily files) |
| Used by | event-only model | production and event-only models |

Of the 14,814 used 2026 matches, 2,223 (15.0%) are September uploads and 12,591 (85.0%) are October uploads (`data/combined/matches.parquet`, upload_day). The built store (`tomes/`) has no matches from 14 or 15 October. Raw match folders for both days exist on the E: drive (950 and 835 match folders, each with a sibling zip). Their origin and completeness are not verified: `notes/STATUS_2026-10-14.md` records approval for days 12 and 13 only. They are not used in any result. The 2024 matches have no per-tick player table, so they support only the event inputs. The production model is therefore trained and scored on 2026 matches only. The event-only model uses both years.

Two cautions. First, the upload-folder dates are upload days, not play days. Second, for hand-uploaded matches (mostly FACEIT servers) the header match date is the processing day, not the day played (comment in `scripts/model/wp_common.py`, function `make_split`). The 2026 catalogue by server is 154 FACEIT competitive, 12,222 FACEIT unknown and 3,914 Valve competitive matches (`data/combined/faceit_audit.json`, run 2026-10-15).

## 4 Map scope and the pool

The scope is seven maps. It was chosen from the data: these were the 5v5 maps with real volume in the first eight days of October 2026 (1,561 of 1,603 non-wingman matches) (`notes/data/COMBINED_STORE.md`).

Whether the seven match the game's Active Duty pool was checked on 2026-10-19:
- Wikipedia's list of Active Duty maps as of July 2026 gives the same seven.
- A community wiki lists de_vertigo where Wikipedia lists de_cache. The two sources disagree on that map.
- No Valve notice dated after July 2026 was found.
- Third-party guides disagree on the dates of earlier pool changes (Vertigo, Overpass, Train, Anubis and Cache). Those dates are not used anywhere in the model.

So the seven maps are the pool as of July 2026 according to Wikipedia. They are not confirmed for October 2026 at the source (section 16).

The scope is applied by map name (`scripts/data/prepare.py`, `IN_SCOPE_MAPS`), so out-of-pool matches are not used. Their counts (`data/combined/prep_report.json`, scope_by_era, reason map_out_of_pool): 1,032 matches in 2024 and 263 in 2026.

## 5 What was excluded, and why

Three exclusions apply to the snapshot-feature models (production) and the event-only model alike. They are applied to rounds, not matches.

**(a) Roster defect.** In some rounds the roster lacks one player, so the side of that player has to be inferred. `drop_roster_defect_rounds` in `scripts/model/wp_common.py` removes every round whose `rounds.parquet` column `missing_side` is set. In the sample checked, the inferred side disagreed with the event log in about 40% of cases (37 of 90 sampled rows). The share of rounds affected is 0.095% of 2026 rounds and 7.5% of 2024 rounds (computed from `data/combined/rounds.parquet`).

**(b) Table versus events disagreement.** Rounds where the player table and the event-derived alive counts disagree are dropped by the cache build (`scripts/model/make_snapshot_cache.py`, docstring).

**(c) Ten matches with no usable round.** Ten 2026 matches used in the split are absent from `data/combined/cache_snapshot_2026.parquet` (14,804 of 14,814 present). They are 7 train, 1 validation and 2 test matches. Every grid round of these ten matches (146 rounds) has `missing_side` set, so the roster defect removes all of them (`data/combined/exp_cache_skip_trace.json`). The production training and test sets are therefore 10,577 train, 1,168 validation and 3,059 test matches, not the 10,584, 1,169 and 3,061 in the manifest.

The event-only test set has 3,601 matches, 10 fewer than the manifest's 3,611. The 10 are 8 matches from 2024 and 2 from 2026: none of their grid rounds survives the roster filter. This was checked against the grid and `rounds.parquet` directly. The validation and training gaps for the event-only model are of the same kind but are not separately listed here. Measured effect of the rule: keeping the 3,841 roster-defect rounds in training makes clean test log loss worse by 0.000042 [0.000020, 0.000062] for the event-only model (`data/combined/exp_roster_sensitivity.json`), so the rule is also supported by the data.

## 6 Rows: one per second

The unit that is scored is one second of live play in one round. Rows are sampled from the grid (`data/combined/grid.parquet`), one per second: tick = freeze end + 64 × k, for k = 0, 1, 2, and so on. Rows stop at the decisive event (the bomb outcome, or the death that wipes a side), as recorded in `notes/data/COMBINED_STORE.md`, section "Effective end".

Round length, in seconds from freeze end, over the 386,496 in-scope rounds (`data/combined/rounds.parquet`): 10th percentile 37.3, median 65.5, 90th percentile 104.4.

Finer rows were tested: 1, 2, 4 and 8 rows per second, against every second, on the same matches. None beats one row per second (`data/combined/exp_tick8.json`, 2,644 matches, test deltas and intervals in its `rates` list). Sampling random rows within each round, from 1 to 64 per round, is worse than every second in every case (`data/combined/exp_random_ticks59.json`). Weighting each draw by its round length closes most of that gap (`data/combined/exp_final_checks.json`: weighted minus unweighted -0.001116 [-0.001440, -0.000790]).

## 7 The split

**Rule.** Matches are assigned at random, by match, not by second and not by date. A second of one match is almost identical to the next, so a split by second would hand the answer to the test set. The assignment is frozen in `data/combined/split_manifest.json` (sha256 a17f219b5287ff1ebae9c4bf976ef9927a9c2f2daa6107f739f390c6777b2ef6). Two rules produced it: a legacy per-era shuffle for the first 6,475 matches (20% test, then 10% of the remainder as validation), and a hash rule for later matches (test if sha256(seed:match_key) is below 0.20, validation if below 0.28, otherwise train). The history records a freeze of 6,475 matches on 2026-10-15 at 09:00, then extensions of +8,393 and +2,695. Existing entries are not changed.

**Shares.** The design target for all matches is 72% train, 8% validation and 20% test (`scripts/model/wp_common.py`, HASH_TEST_FRAC and HASH_VAL_FRAC). The target is not a tuned value. It comes from the defaults of the first training code (`_holdout` in `scripts/model/wp_common.py`: 20% of matches as test, then 10% of the remaining 80% as validation, which is 8% of all matches). No recorded experiment or decision gives a reason for those values, and no experiment in the record varied them. They are not a named standard: the guides cite 80/20 for a two-way split and 60/20/20, 70/15/15 or 80/10/10 for three-way splits. They were carried over from the first models for comparability, and no fraction sweep was run.

**Outside sources.** Published guidance is general and does not name these values. Blog guides list 80/20 as the usual two-way split and 60/20/20, 70/15/15 or 80/10/10 as three-way splits, with the choice depending on dataset size: [Lightly, train/validation/test split](https://lightly.ai/blog/train-test-validation-split); [MetricGate, split best practices](https://metricgate.com/blogs/train-test-validation-split-best-practices/); [APXML, common split ratios](https://apxml.com/courses/basics-model-evaluation-metrics/chapter-4-preparing-data-evaluation/common-split-ratios). A summary of Andrew Ng's *Machine Learning Yearning* suggests dev and test sets of about 1,000 to 10,000 independent examples ([summary by Chelsea Troy](https://chelseatroy.com/2017/06/29/takeaways-machine-learning-yearning-by-andrew-ng-chapters-1-14/)). The primary text was not retrieved. Dietterich (1998) compares tests for a single held-out evaluation (McNemar's test) and for repeated splits (the 5x2 cross-validation test) ([abstract](https://mlanthology.org/neco/1998/dietterich1998neco-approximate/)); the paper does not set a split fraction.

**What our own numbers support.** The independent units are matches, not seconds. The test set has 3,059 matches, and its match-cluster 95% interval for the production model is about plus or minus 0.0030 log loss. The validation set has 1,168 matches. A paired validation interval for a family against XGBoost is about plus or minus 0.00015 (LightGBM, stride 3, `data/combined/exp_families_pistol.json`), and about plus or minus 0.0005 for the score-lead comparison (`data/combined/exp_scorelead_combo.json`). So the validation set resolves differences of roughly 0.0002 to 0.0005, which covers the setting changes that were compared (all within 0.00026 on validation). That is the justification the record supports: the validation set is large enough to choose among settings at the scale that mattered, and the test set is large enough for a usable interval. It is not a justification that 20% and 8% are optimal; no fraction sweep was run. The shares actually obtained:

| Matches | Train | Validation | Test | Total |
|---|---|---|---|---|
| All matches in the manifest | 12,563 (71.53%) | 1,389 (7.91%) | 3,611 (20.56%) | 17,563 |
| 2026 in the manifest | 10,584 (71.45%) | 1,169 (7.89%) | 3,061 (20.66%) | 14,814 |
| 2026 used by the production model | 10,577 (71.45%) | 1,168 (7.89%) | 3,059 (20.66%) | 14,804 |

| Seconds, production model | Train | Validation | Test | Total |
|---|---|---|---|---|
| Rows (`data/combined/exp_full_comparison.json`) | 13,758,901 (71.55%) | 1,515,228 (7.88%) | 3,956,374 (20.57%) | 19,230,503 |

Against the 72/8/20 target, the test share is 0.66 points high and the train share 0.55 points low. The validation share is 0.11 points low. The split was frozen before any result was known and was not adjusted to hit the target.

**Checks.** The manifest check confirms that the sets are disjoint and that every grid match is assigned (`scripts/model/checks/split_checks.py`; its only saved output is `data/combined/e2e_winprob_v3.json`, run 2026-10-15). Every model's metadata records the same manifest hash.

## 8 Inputs: the production model (58)

The list is in `models/winprob_v8_ensemble.meta.json` (features). Definitions are in `scripts/serving/win_probability.py` (`_loadout_from_players`, with the item-id tuples RIFLE, M4, AK, AWP, SNIPER_OTHER, SMG and HEAVY), in `scripts/model/features_status.py`, and in `scripts/data/make_grid.py` for the event inputs.

**Composition.** 13 event inputs + 1 score lead + 44 loadout inputs = 58. The 44 loadout inputs are 19 per side (CT and T, 38 in all) and 6 CT-minus-T differences.

**The 13 event inputs.**
- `alive_ct`, `alive_t`: living players on each side at this second.
- `ct_alive_minus_t`: the difference of the two.
- `bomb_planted`: 1 once the bomb is planted, otherwise 0.
- `seconds_since_round_live`: (tick - freeze end) / 64.
- `seconds_since_plant`: (tick - plant tick) / 64 after a plant, and -1 before it.
- Seven map indicators: `map_is_ancient`, `map_is_anubis`, `map_is_cache`, `map_is_dust2`, `map_is_inferno`, `map_is_mirage`, `map_is_nuke`. All zero for an out-of-scope map.

**The score lead (1 input).** `score_lead_diag`: the CT score minus the T score at the freeze end of the round (round_state, `round_freeze_end`). It is constant within a round (`data/combined/grid.parquet`).

**The 44 loadout inputs.** For each side, from the living players at this second:

| Input | Meaning |
|---|---|
| `hp` | sum of health |
| `armor_n` | living players with armour above 0 |
| `armor_sum` | sum of armour |
| `helmet_n` | living players with a helmet |
| `kit_n` | living players with a defuse kit (always 0 for T; see below) |
| `defusing` | living players who are defusing (always 0 for T) |
| `ak`, `m4`, `awp` | players carrying an AK, an M4, or an AWP |
| `rifle` | players carrying any rifle (includes AK and M4) |
| `sniper_other` | other snipers (scout and autosnipers) |
| `smg` | players carrying an SMG |
| `no_primary` | players with no primary weapon |
| `flash`, `smoke`, `he`, `decoy` | grenades carried, counted |
| `fire` | molotov plus incendiary grenades carried |
| `equip` | sum of equipment value |

The six CT-minus-T differences are `hp`, `rifle`, `equip`, `armor_n`, `awp` and `smg`.

**Constant inputs.** `st_kit_n_t` and `st_defusing_t` are zero on every row of the cache (`data/combined/cache_snapshot_2026.parquet`, checked on 2026-10-19). Two of the 58 inputs therefore carry no information; the other 56 vary.

**Monotone constraints.** Three inputs are constrained: `alive_ct` +1 (more CT alive never lowers CT's probability), `alive_t` -1, and `ct_alive_minus_t` +1. No other input is constrained. The evidence for keeping these constraints is in section 11.

**Dropped.** The heavy-weapon group (shotguns and machine guns, item ids 14, 25, 27, 28, 29 and 35) was in the first 59-input model and is out of the 58. Section 11 gives the evidence.

**Not used.** Round number, pistol and overtime flags (pistol and after-pistol rounds appear only as evaluation slices), money, positions, view direction, smokes in play, fire zones, blindness, player or team identity, and rank. Positions and blindness were tested in a deferred experiment (section 13.7).

**Where the values come from.** Event inputs are grid columns, built from `rounds.parquet` and `deaths.parquet`. Loadout inputs come from the per-second player status extraction (`data/status_1hz/`), and the cache is built by `scripts/model/make_snapshot_cache.py`. `scripts/model/wp_items.py` holds the training-side item lists. `scripts/serving/win_probability.py` keeps its own copy of the same constants, which must be kept in step by hand.

## 9 Inputs: the event-only model (14)

**Composition.** The 13 event inputs of section 8 plus `score_lead_diag`: 14 in all. No loadout inputs. The monotone constraints are the same three (`alive_ct` +1, `alive_t` -1, `ct_alive_minus_t` +1). List and metadata: `models/winprob_v7_basic_scorelead.meta.json`.

**Training and scoring rows.** Grid rows of both years, with roster-defect rounds removed (`drop_roster_defect_rounds`, true in the metadata). The 2024 matches contribute the 2024 rows; the 10 excluded test matches are section 5(c).

**Model.** One seed. XGBoost with depth 4, eta 0.05, min_child_weight 20, subsample 0.8, early stopping 100. Best iteration 548 (metadata).

**Served role.** It is the model to request for a tome that has no per-tick player status. It is not chosen automatically: the default serving model (the submission) needs the status channel, and a tome without it returns an empty result with a warning (section 14).

## 10 The production model: XGBoost and the five-seed ensemble

**The learner.** Gradient-boosted decision trees (XGBoost, binary logistic objective, histogram method). Trees are added one after another, each fitted to the errors left by the earlier ones; the sum is passed through a logistic function to give a probability.

**Settings.** Depth 4, learning rate (eta) 0.05, minimum child weight 20, subsample 0.8, early stopping after 100 rounds without gain on validation, at most 3,000 trees. Monotone constraints as in section 8. Trained on the GPU (CUDA), served on the CPU (`models/winprob_v8_ensemble.meta.json`, params).

**The five seeds.** Seeds 0 to 4 train the same model; they differ only in the random row subsampling. Best iterations: 1,408, 1,402, 1,181, 1,331 and 1,289. Validation per seed runs from 0.452767 to 0.452838; test from 0.455634 to 0.455732 (`data/combined/exp_seed_avg58.json`).

**The output.** The ensemble is the arithmetic mean of the five probabilities, second by second. Averaged: validation 0.452722, test 0.455605 with interval [0.45243, 0.45841] (`data/combined/exp_seed_avg58.json` and the predictions file). Against seed 0 alone, the average is better on test by 0.000076, with interval [-0.000127, -0.000023] (`test_delta_vs_seed0_ci95`). That gain is small, but its interval excludes zero.

**Why five.** Averaging removes most of the variation from the random subsampling without changing the settings. The spread between seeds is 0.00007 on validation and 0.00010 on test, so the single-seed choice would move a result by about that much.

## 11 How choices were made

**The rule.** Choices were made on validation. Test numbers were reported for the final choices, and the test set was not used to pick between settings. The history is less clean than the rule. The test set was computed many times during the work: 31 of the 62 experiment files record both validation and test numbers, and 31 record test numbers alone (`data/combined/exp_*.json`). "Test scored once" describes the reporting rule for the final figures, not the history.

**Settings.** Eleven configurations were compared on validation: the baseline and ten alternatives (depth 5, 6 and 8; min child weight 50 and 100; subsample 1.0; column sampling 0.8; eta 0.03; and combinations). None beat the baseline on validation; the closest were +0.00001 (`data/combined/exp_hparams58.json`, field selected). The test numbers for all configurations are in the same file.

**Heavy-weapon group.** Removing the heavy group changed validation by -0.00009 and test by -0.00002 [-0.00010, +0.00005] in the 59-input model (`data/combined/exp_loadout_ablation.json`). Validation alone does not separate the two; the drop was partly decided on test consistency. The grill-me notes say so (`notes/GRILLME_FINAL_2026-10-19.md`).

**Score lead.** Adding the score lead together with the heavy-group drop (seed 0) changed validation by -0.000865 [-0.001378, -0.000372] and test by -0.000754 [-0.001066, -0.000459] (`data/combined/exp_scorelead_combo.json`). Both splits moved; the score lead is kept.

**Monotone constraints.** Two experiments disagree in direction, and the event-only model keeps its constraints although its own experiment favours removing them.
- 59-input model, removing the constraints: validation -0.000064 [-0.000183, +0.000066] (interval includes zero) and test +0.000092 [+0.000009, +0.000176] (`data/combined/exp_59_nomono.json`). Neutral on validation, a small cost on test. Kept.
- 13-input event model (without the score lead), removing the constraints: validation -0.000188 [-0.000357, -0.000018] and test -0.000443 [-0.000588, -0.000315] (`data/combined/exp_mono_paired.json`). Removing them is better on both splits. The 14-input model that was served keeps the constraints, and this experiment did not test the 14-input model.

The production model keeps its constraints because they are neutral on validation. The event-only model is the open question (section 16).

**Rejected alternatives.** Each was compared with the same reference on the same seconds.
- Finer rows: none beats one row per second (section 6).
- Event features (cumulative damage and shots so far): validation -0.000024, test +0.000156 [+0.000020, +0.000290] (`data/combined/exp_event_features.json`). Rejected.
- Dedicated pistol-round models: overall test +0.000224 [+0.000075, +0.000371] against seed 0 (`data/combined/exp_families_pistol.json`, key pistol). Rejected.
- Joint removal of 32 loadout features (27 inputs left): validation +0.002996 [+0.002125, +0.003911] and test +0.002855 [+0.002329, +0.003352] (`data/combined/exp_loadout_joint.json`). Rejected.
- Other model families: section 13.4.

## 12 Evaluation and uncertainty

**Metric.** For a row with CT-win probability p and label y (1 if CT won), the loss is -[y ln p + (1 - y) ln(1 - p)]. The score is its mean over all rows. `scripts/model/wp_common.py` (`evaluate`) clips p to [1e-15, 1 - 1e-15]; the pairing and family scripts clip to [1e-7, 1 - 1e-7]. The two conventions agree for the production ensemble; they were not compared for the other models.

**Intervals.** Seconds within a round, and rounds within a match, are correlated. Every interval therefore resamples whole matches with replacement (1,000 resamples, seed 0; 2.5th and 97.5th percentiles of the resampled means), not individual seconds.

**Round-weighted view.** Averaging per round instead of per second gives the long-round-heavy scores a lower weight. Round-weighted figures are in `data/combined/exp_full_comparison.json` (field logloss_round_weighted) and in `data/combined/exp_final_checks.json`.

**Leakage canary.** A test-time copy of a feature that reads a later second (the future alive count) drops test loss to 0.4413 (`data/combined/exp_final_checks.json`, canary). The metric is sensitive enough to show leakage of that size.

## 13 Results

### 13.1 Production model

Test (3,059 matches, 3,956,374 seconds, 58,059 rounds): 0.455605 [0.45243, 0.45841]. Single seed 0: 0.455680. Validation: 0.452722.

### 13.2 Like-for-like with the event-only model

On the same 3,956,374 seconds of the 3,059 test matches (`data/combined/exp_eventonly_paired.json`, `scripts/model/exp_eventonly_paired.py`):

| | Log loss | Paired difference, event-only minus production |
|---|---|---|
| All rounds | production 0.455605; event-only 0.498695 | +0.043090 [+0.041449, +0.044884] |
| Pistol rounds (1, 13), 294,872 seconds | | +0.012046 [+0.008448, +0.015786] |
| Rounds after pistol (2, 14), 320,192 seconds | | +0.139070 [+0.129061, +0.148363] |
| Other rounds, 3,341,310 seconds | | +0.036632 [+0.034767, +0.038517] |

The gap mixes two things. One is the loadout inputs, which the event-only model lacks. The other is the training data: the event-only model trains on 2024 and 2026, the production model on 2026 alone. No experiment in the record separates the two. The gap is therefore not a pure measure of the loadout inputs.

### 13.3 Round groups (production model, test)

Recomputed from the saved predictions and labels (`data/combined/exp_rounds_groups.json`, written by `scripts/viz/build_rounds_slide.py`):

| Group | Seconds | Matches | Log loss | Interval |
|---|---|---|---|---|
| Pistol rounds (1, 13) | 294,872 | 3,033 | 0.519138 | [0.511379, 0.527875] |
| Rounds after pistol (2, 14) | 320,192 | 3,032 | 0.313411 | [0.301668, 0.325398] |
| All other rounds | 3,341,310 | 3,047 | 0.463624 | [0.460438, 0.467064] |

Separate models for pistol and after-pistol rounds were worse on the same seconds (section 11).

### 13.4 Model families

Two designs, both on the same test seconds (`data/combined/exp_families_pistol.json`, key families; `data/combined/exp_full_comparison.json`; `data/combined/exp_families_torch.json`).

**Design A: matched settings, every third training second, 58 inputs.** Each family is paired against the XGBoost reference:
- LightGBM, monotone constraints: validation -0.000115 [-0.000259, +0.000033]; test -0.000071 [-0.000161, +0.000020].
- CatBoost on the GPU, no monotone constraints: validation -0.000004 [-0.000282, +0.000292]; test +0.000099 [-0.000067, +0.000274].
- CatBoost on the CPU with monotone constraints, and logistic regression: not run in this design.

**Design B: the full training set, 13,758,901 seconds, 59 inputs, each family with the settings in its run.** Each family is paired against the XGBoost reference on the test seconds (`delta_vs_xgb`, `delta_vs_xgb_ci95`):

| Model | Test log loss | Test difference to XGBoost | Interval |
|---|---|---|---|
| XGBoost (reference) | 0.456412 | | |
| LightGBM, 31 leaves | 0.456782 | +0.000370 | [+0.000237, +0.000523] |
| CatBoost, CPU, monotone | 0.456754 | +0.000342 | [+0.000149, +0.000542] |
| CatBoost, GPU, no monotone | 0.456814 | +0.000402 | [+0.000231, +0.000577] |
| Logistic regression | 0.467410 | +0.010998 | [+0.010421, +0.011591] |

Random forest, extra trees and two neural networks are also in the file, all further behind. The settings in design B were not tuned to each family.

**Reading.** Design A shows no clear difference. Design B shows XGBoost ahead of every boosted family with the settings used, and every paired interval excludes zero. The two designs use different rows and different settings, so they are not one experiment. XGBoost is kept: it is the family the production model uses, and in the one design where the families were run on all the data, it was ahead.

**Run-to-run noise.** The GPU CatBoost configuration was run twice in design A. The first run gave validation 0.452679 (`logs/families_cb_gpu.log`) and the repeat 0.452827 (`logs/families2_cb_gpu.log`, stored in the JSON). The difference, about 0.00015 on validation, is the scale of noise to expect.

### 13.5 Loadout groups

Removing groups from the 59-input control (validation 0.453643, test 0.456435) (`data/combined/exp_loadout_ablation.json`, `exp_loadout_blocks.json`, `exp_loadout_joint.json`):

| Removed | Test change | Interval |
|---|---|---|
| Health | +0.004630 | [+0.004070, +0.005230] |
| Defuse kits, defusing, helmets | +0.001600 | [+0.001410, +0.001800] |
| Armour | +0.000690 | [+0.000460, +0.000940] |
| Weapon types and equipment (22 features) | +0.002698 | [+0.002211, +0.003188] |
| Weapon types and grenades (29 features) | +0.000388 | [+0.000110, +0.000675] |
| All 32 redundant loadout features (27 left) | +0.002855 | [+0.002329, +0.003352] |
| Heavy weapons | -0.000020 | [-0.000100, +0.000050] |

Single weapon groups (AWP, SMG, other snipers, pistol-only players) and grenades on their own changed test by less than 0.0001 and are not separately significant.

### 13.6 Earlier models

The event-only model with the score lead (`winprob_v7_basic_scorelead`, 14 inputs) replaced the earlier 13-input served model (`winprob_v4_basic`). On the all-match test set (3,601 matches) they scored 0.501242 and 0.502198 (`models/*.meta.json`). These figures are on the same test set and may be compared with each other, not with the production figure.

### 13.7 Positions (deferred, not in the model)

Adding 24 position-derived inputs to the 59-input base (view-cone exposure, "sees enemy" and "exposed" counts, and flash-blindness counts) lowered test loss by 0.003269 [-0.003560, -0.002952] and validation by 0.003434 [-0.003901, -0.002964] (base test 0.456492, with positions 0.453223) (`data/combined/exp_positions_groups_pod_cpu.json`, a RunPod run). The owner has deferred positions to a later model, so they are not in the production or event-only models.

## 14 Serving

`scripts/serving/win_probability.py`: `win_probability(data, team, model_name="winprob_v8_ensemble", unit)`.
- The default model is the submission, `winprob_v6_drop_heavy` (57 inputs). It needs the per-tick `player_status` channel in the tome; without it the function returns an empty result and records a warning. The score-lead comparison, `winprob_v8_ensemble`, is passed by name and has the same requirement; its metadata says "no fallback: no status, no probabilities" (`models/winprob_v8_ensemble.meta.json`, serving.needs).
- The event-only model is not chosen automatically. It must be requested by name (`model_name="winprob_v7_basic_scorelead"`).

**Parity.** For the event-only model, the served function was run on the raw tome channels of the 294 October test matches and compared with the grid predictions at the same ticks: 294 of 294 matches, 374,541 seconds, maximum gap 0.0001 percentage points (`models/winprob_v7_basic_scorelead.meta.json`, serving block; `logs/test_win_probability_scorelead.log`). The production model's serving block cites a parity result for the ensemble (`logs/test_win_probability_ensemble_full2.log`); that result was not re-checked on 2026-10-19.

**Metadata provenance.** `scripts/model/finalize_ensemble58.py` writes `served: false` and no serving block into the production metadata. The current `models/winprob_v8_ensemble.meta.json` says `served: true` and carries a serving block. No script in `scripts/` writes `served: true`. The change was made by hand or by a step not recorded in the repository. This is an open item (section 16).

## 15 Limitations

- **Data.** Two windows: 2024 matches from February 2024 uploads (used only by the event-only model), and 2026 matches played August to October 2026 (used by the production model). Matches come from FACEIT and Valve servers. One year of play is not covered.
- **Maps.** Seven maps. Neither model has been measured on other maps. An earlier 59-input model was run on de_vertigo and de_overpass with the map inputs set to zero, which is an approximation and is not a measure of the models here (`data/combined/eval_scope_maps_winprob_v3.json`).
- **Pool.** The seven maps match Wikipedia's Active Duty list as of July 2026. They are not confirmed against a Valve notice for October 2026.
- **Not in the model.** Positions, facing, smokes in play, fire zones, money and buy choices, and player identity. Positions and flash blindness were tested in a deferred experiment (section 13.7).
- **Rows.** One row per second. Finer rows were tested only up to 8 per second.
- **Constant inputs.** Two of the 58 production inputs are always zero (section 8).
- **Test reuse.** The test set was computed many times in the work (section 11).
- **Headers.** Header match dates are processing days for hand-uploaded matches, so the true play dates for those matches are unknown.
- **Event-only fallback.** It is not automatic (section 14).

## 16 Open items

These are the things a reviewer should decide or check. Items 1 and 2 need the owner's decision.

1. **Monotone constraints in the event-only model.** Removing them improves both validation and test in the 13-input model (section 11). The served 14-input model keeps them and has not been tested without them. Decision: test the 14-input model without constraints on validation, and change it only if validation improves.
2. **2024 in the event-only training.** No experiment separates the effect of the 2024 training rows from the effect of the loadout inputs (section 13.2). A 2026-only versus both-years event-only fit on the same features would settle it.
3. **Family choice.** Design A shows no difference. Design B shows XGBoost ahead with the settings used. Neither design tunes each family. CatBoost on the CPU with monotone constraints and logistic regression were never run in design A (only in design B).
4. **Metadata.** The production metadata's `served: true` and its serving block were not written by the script that is named for them (section 14).
5. **Pool.** The seven maps are confirmed only to Wikipedia's July 2026 list. A Valve notice would settle October 2026.
6. **Stale notes and comments.** `scripts/model/wp_common.py` says roster-defect rounds are 0.4% of 2026; the current count is 0.095%. `notes/data/COMBINED_STORE.md` gives 14% for 2024 player_info gaps; the current rounds share is 7.5%. The feature list in `scripts/data/make_grid.py` omits the score lead, which the production model uses. These should be corrected in place.
7. **Clipping.** The metric clips at 1e-15 in `wp_common.py` and at 1e-7 in the pairing and family scripts. They agree for the production ensemble only.
8. **Constant inputs.** `st_kit_n_t` and `st_defusing_t` are always zero. Remove them or keep them explicitly.
9. **Test reuse.** Section 11 describes it. The final figures are reported once per model choice, not once per experiment.
10. **Family intervals in design A.** CatBoost on the CPU at stride 3 and logistic regression at stride 3 were not run.

## 17 Reproducing

Commands run from `G:\pureskill\csds-cs2` with `.pixi\envs\default\python.exe -I -X utf8` unless the script's docstring names a GPU environment.

| Step | Script | Writes |
|---|---|---|
| Split manifest | `scripts/model/split_manifest.py` | `data/combined/split_manifest.json` |
| Snapshot cache | `scripts/model/make_snapshot_cache.py` | `data/combined/cache_snapshot_2026.parquet` |
| Cache exclusions | `scripts/model/exp_cache_skip_trace.py` | `data/combined/exp_cache_skip_trace.json` |
| Round groups | `scripts/viz/build_rounds_slide.py` | `data/combined/exp_rounds_groups.json` |
| Like-for-like | `scripts/model/exp_eventonly_paired.py` | `data/combined/exp_eventonly_paired.json` |
| Families, design A | `scripts/model/exp_families_pistol.py --only lgb,cb,logistic --device cpu` (LightGBM and CatBoost; CUDA for the GPU CatBoost) | `data/combined/exp_families_pistol.json` |
| Families, design B | `scripts/model/exp_full_comparison.py` (RunPod runs; see the JSON `merged_from` field) | `data/combined/exp_full_comparison.json`, `exp_families_torch.json` |
| Serving parity | `scripts/serving/test_win_probability.py` | `logs/test_win_probability_*.log` |
| Deck slides | `scripts/viz/build_*.py` | `deliverables/deck_2026-10-19/project/slides/*.html` |
| Deliverable summary | `scripts/model/write_deliverable_summary.py` | `notes/DELIVERABLE_2026-10-19.md` |

The full list of experiments, with status and decisions, is in `notes/EXPERIMENTS_2026-10-19.md`.
