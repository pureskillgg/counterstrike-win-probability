# Deployment and player impact: CS2 round win probability (Task E)

Date: 2026-10-13. Status: design spec. Nothing here is validated outside the 2026-09-18 to 2026-10-08 window.
Scripts: `scripts\spec\time_deploy_predict.py` (timing and parity). Output: `notes\spec\deployment_timing_2026-10-13.json`.
Evidence labels: VERIFIED (say how) or UNVERIFIED. Timings are on synthetic inputs, so they measure speed, not accuracy.

## 0. Direct answers

- **Where are the tomes?** On G:, not E:. The header and basic channels are at `G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds`. E: holds the raw per-match files, including per-tick `player_status` and `player_vector` (VERIFIED: `notes\schema_dictionary.md` profiles one E: match; `notes\spec\feature_spec.md` section 6 says the tome has no per-tick channels). This task did not read E:.
- **Latency is not the constraint.** Native XGBoost scores one state in about 0.18 ms on one thread (VERIFIED, timed).
- **Correction to the request text.** It says "24-round halves". MR12 is 24 rounds in two 12-round halves, and the win threshold is 13 (`max_rounds//2+1`). VERIFIED: header `max_rounds` 24 (`notes\schema_dictionary.md` section 1) and swaps after round 12 in 24-round matches (`feature_spec.md` section 3).
- **Player credit from this model is leverage, not skill.** The 10 features contain no player identity, position, HP, weapon or economy. See section 5.

## 1. Model facts

VERIFIED: read `notes\model\MODEL_CARD.md` and `models\xgb_primary_seed0_meta.json`, and loaded `models\xgb_primary_seed0.json` in xgboost 3.4.2.

- File `models\xgb_primary_seed0.json`, 599,074 bytes. The file holds 362 trees; the model uses the first 262 (`iteration_range=(0, 262)`).
- Max depth of the used trees is 4. Max nodes per tree in the file is 31 (a full depth-4 tree).
- Objective is binary logistic, P(CT wins the round). Base score is stored as `[5.213736E-1]` (base margin 0.08555).
- No calibration. Primary split, states: test logloss 0.4366, Brier 0.1464, AUC 0.8703, ECE 0.0089; validation ECE 0.0124 (`MODEL_CARD.md` and meta metrics). Sigmoid calibration raised test ECE to 0.0408 and isotonic was slightly worse, so the raw model is what ships.
- Monotone constraints: alive_ct +1, alive_t -1, ct_alive_minus_t +1, ct_score_minus_t_score +1, the rest 0.

## 2. Deployment options

| Option | Install weight | Latency per state | Status |
|---|---|---|---|
| A. Native XGBoost Booster (JSON), Python | pixi: py-xgboost 3.4.2 (180 KiB), libxgboost 1.54 MiB, numpy 7.04 MiB (VERIFIED: `pixi list`) | 0.18 ms median, 1 thread, `inplace_predict`, p95 0.47 ms (VERIFIED, 2,000 calls). 0.63 ms median with a new DMatrix per call, 1 thread (VERIFIED). | Recommended for Python serving. Pin xgboost==3.4.2. |
| B. Pure-Python walker over the JSON (stdlib only) | None (VERIFIED: uses `json` and `math` only) | 0.13 ms median per state, 1 thread (VERIFIED). About 1.9 s per 10,000 rows (VERIFIED). | Reference implementation and fallback. Parity: max probability difference 4.1e-7 vs Booster on 10,000 rows; NaN rows 5.5e-8 (VERIFIED). |
| C. ONNX (xgboost-to-ONNX, onnxruntime) | Not installed; not in the pixi env (VERIFIED by `pixi list`). Weight UNVERIFIED. | UNVERIFIED. Expected sub-millisecond per state (see note). | Use only if the runtime is not Python. Parity test required. Conversion of NaN default directions and monotone constraints is UNVERIFIED. |
| D. Treelite (compiled C) | Not installed. Weight UNVERIFIED. | UNVERIFIED. Expected microseconds per state. | Best option for a compiled service with no Python. Parity test required. |

Note on estimates (UNVERIFIED, from tree count and depth): a state takes at most 262 trees x 4 comparisons = 1,048 comparisons. Compiled code should run that in microseconds. Measured Python cost (0.13 to 0.18 ms) is mostly interpreter and call overhead.

Serving rules:
- Call with `iteration_range=(0, 262)` every time. Omitting it would use all 362 trees. The size of that change was not tested here (UNVERIFIED).
- Set `nthread=1` per process for per-state calls. Batch scoring of 10,000 rows took about 12 to 15 ms on 16 threads (VERIFIED).
- Release gate: any deployed scorer must match the Booster within 1e-5 on a fixed state set (the synthetic 10,000 rows plus real held-out states).
- Timing gotcha found while measuring: re-predicting the same DMatrix hits XGBoost's prediction cache and gives about 0.2 ms. Each timed call needs a fresh DMatrix (VERIFIED; fixed in the script).

## 3. Input contract

Feature order (must match `models\xgb_primary_seed0_meta.json`):
`alive_ct, alive_t, ct_alive_minus_t, bomb_planted, seconds_since_round_live, seconds_since_plant, round_number, ct_score_minus_t_score, is_pistol_round, is_overtime`

| Feature | Type | Allowed values (by definition, from `feature_spec.md` section 5) |
|---|---|---|
| alive_ct | int | 0 to 5. 5 at live start minus CT deaths in window. |
| alive_t | int | 0 to 5. Same rule for T. |
| ct_alive_minus_t | int | -5 to 5. Must equal alive_ct - alive_t; reject otherwise. |
| bomb_planted | {0,1} | 1 if a plant in the live window is at or before the state tick. |
| seconds_since_round_live | float | >= 0. (tick - round_freeze_end tick) / 64. |
| seconds_since_plant | float | -1 exactly when bomb_planted = 0; >= 0 when 1. |
| round_number | int | >= 1, 1-based within the match. |
| ct_score_minus_t_score | int | Score at live start. Bounded by round_number - 1, since t_score + ct_score = round - 1 held at all 35,953 live starts (VERIFIED: `feature_spec.md` section 3). |
| is_pistol_round | {0,1} | 1 iff round is 1 or max_rounds//2 + 1 (13 for MR12). |
| is_overtime | {0,1} | 1 iff round > max_rounds. |

Missing values:
- Do not impute. Any null feature means no score is returned and an error is logged. The features are counts and times from event rows, so a null means a missing event or team. Imputing would silently shift alive counts.
- NaN was not a training case by design (all features defined for every kept state). The model would accept NaN through XGBoost's learned default direction, but that is not a principled fill (UNVERIFIED that no NaN appeared in training).
- seconds_since_plant = -1 is a sentinel for "no plant", not a missing value.

Round gates (refuse to score, do not guess):
- Any `player_disconnect` or `bot_takeover` in the round. Training dropped 1,767 such rounds (`MODEL_CARD.md` filters), so the model never saw these states. A disconnect is not a death, so alive counts can be too high (`feature_spec.md` caveat: 29 sample rounds).
- Not 5 vs 5 at start (training dropped 476 rounds from `player_info`). Wingman matches (`is_wingman` True) were excluded from the batch-effect comparison and should be refused.
- `max_rounds` outside the trained set. The bakeoff note mentions 149 matches in the 16-round format. Whether any survived the training filters is UNVERIFIED here.

Output: P(CT wins) in (0, 1), plus a version tag. Record the model file's SHA-256 at release (not computed here).

## 4. Monitoring

- **Calibration drift.** Track ECE (10 bins), logloss and Brier per rolling window of completed rounds, with bootstrap intervals over matches. Also report round-level metrics: the model card reports state-level metrics only, and long rounds carry more states. Baselines: test ECE 0.0089, validation ECE 0.0124. Alert thresholds are a proposal, UNVERIFIED: for example, ECE above 0.02 with the interval above 0.01, or mean predicted vs observed CT-win gap above 3 points over 300 rounds.
- **Recalibration policy.** None now. Recalibrate (isotonic on a recent holdout) only after a confirmed drift. Sigmoid is excluded (`MODEL_CARD.md`).
- **Input distribution.** Daily PSI or KS per feature against the training split. Rate checks: plant rate per round, alive-count distribution, round_number distribution, pistol and overtime share, 5v5 share, disconnect and takeover share, null rate on team fields, duplicate share.
- **Metadata as a source signal.** `match_type` and `platform` are unknown for 92.7% of 2026-10-08 matches and 3.9% of other days (`batch_effect_2026-10-08.md`). The model does not use them. Keep them out of the features: the batch EDA says they would make the batch trivially identifiable. Monitor them as a sign the feed changed.
- **Batch effect** (`notes\eda\batch_effect_2026-10-08.md`, VERIFIED by reading). 2026-10-08 is 72% of 5v5 rounds. The winner base rate does not shift (T 47.8% vs 48.0%, p = 0.69). Plant rate is +2.6 points on 5v5 (plant time +3.5 s), Target_Bombed +2.6 points, kills per round -0.13. Round-start state identifies the batch only weakly (AUC 0.565). Rounds per match are about 2.8 higher on 5v5 (21.0 vs 18.2).
- **Gap in the evidence.** The primary split is random over matches, so it mostly tests within-batch error. The model card's chronological split has all test matches on 2026-10-08, so it does not test cross-day shift either. The gate before any deployment claim is: train on 2026-10-01 to 10-07, test on 2026-10-08, with death and plant features included. Not run here.

## 5. Player impact

### 5.1 Method in the source

Source: Xenopoulos, Doraiswamy and Silva, "Valuing Player Actions in Counter-Strike: Global Offensive", arXiv 2011.01324 v2 (4 Nov 2020). The arXiv abstract page and the HTML v2 were both fetched. Quotes below are from the fetch tool's extraction (targeted passes), not my own reading of the PDF, which the fetch tool returned as binary. Check wording before citing.

- **Win probability model.** Target = 1 if CT wins. States are recorded at each footstep, damage and bomb event. Features: map, ticks since round start, round-start equipment value, players remaining, HP per team, bomb planted flag, bomb site, minimum player distance to each site (graph distance on nav meshes). Models: logistic regression, CatBoost, XGBoost. Data: 4,682 LAN matches. Extraction gives XGBoost test logloss 0.5353, Brier 0.1842, AUC 0.7913 (UNVERIFIED digits). These are not comparable to our numbers: different data, unit and features.
- **Action value.** V(a) = Ŷ(t+1) - Ŷ(t) (extracted: "We can value the player's action as the difference in win probability between the two states").
- **Which actions.** Quoted: "we only define damage events as actions for our framework" and "to calculate WPA, we only consider damage events". Kills, plants and defuses are not valued separately (extraction; listed as future work).
- **Credit.** The player dealing damage gets V, normalized to their own team. The player receiving damage gets the negative of V. Player WPA = sum of action values over all games.
- **Assists and trades.** Not addressed.
- **Validation.** Calibration over 100 bins; month-to-month correlation of WPA with ADR, KAST, KDR and HLTV Rating 2.0 (n = 479); correlation with KDR; top-10 lists; bootstrap over rounds (100 samples). No direct check of summed credit against scoreboard totals.

Caveats for CS2 and for our data:
- Source is CS:GO, MR15 (halves of 15). Ours is CS2 MR12 (halves of 12).
- Our tome has no per-tick positions, HP, equipment or money (`feature_spec.md` section 6). The paper's feature set cannot be reproduced.
- The paper's unit is damage. Damage-level states need HP, which our model lacks. Our `basic_player_hurt` rows exist but cannot be used as states.
- CS2 uses sub-tick updates. Our timestamps are 64-tick (`second` = tick / 64), so events within one tick have no reliable order.
- Population differs: 2016 to 2019 pro LAN matches vs 2026 steam, faceit and unknown-platform matches.

### 5.2 Our version: what is credited

Constraints:
1. States are event-anchored: live start plus each in-window death (`feature_spec.md`). There are no damage states, so the action unit here is the kill.
2. The model has no player identity or skill feature. Per-player differences come only from who died and who killed, and when.

Definitions (side-normalized):
- W(s) = model P(CT wins) at state s.
- For a kill e at tick t_e, with counts before and after the kill:
  - Time component: T_e = W(t_e, counts before) - W(t_prev, counts before). Goes to a time ledger, not to a player.
  - Action component: X_e = W(t_e, counts after) - W(t_e, counts before). This is the only part credited to players.
  - Killer on side s_k gets credit = X_e x (+1 if s_k is CT, -1 if T). This equals the change in the killer's own team's win probability.
  - Victim gets the negative of the killer's credit (the paper's receiver rule). Each kill is zero-sum across the two players.
- Monotone constraints (alive_ct +1, alive_t -1) should make every kill credit weakly non-negative for the killer. This is an expected property of XGBoost, not yet tested (UNVERIFIED): grid-test the Booster before relying on it.
- Assists: default is no credit (assist ledger). Sensitivity: killer keeps (1 - a) of the credit and the assister gets a x credit (a = 0.5 is an arbitrary choice, UNVERIFIED). Zero-sum holds in both. Flash assists (`is_flash_assist`) go in their own column.
- Trades: flag a kill as a trade if the victim had just killed a teammate within N seconds (N = 5 is a proposal, UNVERIFIED). Descriptive only; no credit change.
- Plant: evaluate the model at the plant tick. Features are valid there (bomb_planted = 1, seconds_since_plant = 0). The planter gets the plant's X from the model. UNVERIFIED for credit: training had no plant-tick states unless a death followed (`MODEL_CARD.md`), so calibration at plant-tick states must be checked first.
- Defuse: the model has no defuse feature, so X = 0 at `bomb_begin_defuse`. The completion (`bomb_defused`) ends the round, so it is an outcome. Its credit is realized, based on the outcome, and must be kept separate from model credit. It cannot be used for deployed prediction (leakage).
- Round end with no kill (timer, or a bomb explosion nobody caused): the change from the last state to the outcome goes to the time or bomb ledger, not to players.

Identity checks:
- Per round, sum of (T_e + X_e) plus the final outcome change equals 1{CT wins} - W(live start). This telescopes exactly and is testable in code.
- Per match, the sum of all player net credits is zero (zero-sum by construction). Any nonzero total is a bug.

### 5.3 Side swaps and player slots

- Side for each event comes from the round's `player_info` (or `player_team_code`, which should agree; check). Team-normalized credit uses the actor's side at that event, so no extra sign flip is needed.
- Halftime swap is after round 12 in 24-round matches; overtime swaps follow (`feature_spec.md` section 3).
- Player IDs are per match (raw index 0 to 9, or the `*_fixed` alternative; the dictionary says `player_id_fixed` is 1:1 with `player_id` in one checked match, so confirm per match). Within a match, a slot should map to one human. Confirm this on every match before use.
- No stable cross-match player key was found. For the profiled match, `schema_dictionary.md` says steam_id, name and clan_tag are redacted or anonymized, and `csds_dictionary.csv` marks name as redacted (VERIFIED by reading; the tome's own player_personal fields were not checked). Sum credits only within a match. Across matches, compare players with within-match z-scores.
- Starters: group the five players who started on each side. The header's `t_starters_score_final` and `ct_starters_score_final` reconcile with identity-based totals, not side totals (`schema_dictionary.md` section 7, one match). Use that to validate the side logic.

### 5.4 Validation

1. Unit tests: per-match zero-sum, round additivity identity (section 5.2), and the monotone grid test.
2. Starter-identity score check: rounds won per starter group equals the header finals, for every match.
3. Scoreboard tracking, adapted from the paper: correlate net credit per player-match with K minus D (from `player_death`). ADR is possible from `basic_player_hurt` health removed, if the tome carries that column (UNVERIFIED). Report residuals.
4. Clutch sanity: in 1-vs-N rounds, the lone survivor's kills should carry large credit when their side's win probability is low. Check the sign and size.
5. Stability: split by batch day (10-08 vs earlier), by split (primary vs chronological), and across models. Use bootstrap over rounds for intervals, as the paper does.
6. Baseline: a count-only lookup (W as a function of alive counts, time, bomb and score only). If the XGBoost credit matches it closely, the model adds little for player credit.

Limitation: summed credit is not skill. The model cannot see positions, utility or HP, so the credit ranks players by their part in count changes at the moment of the change. Claims about player impact need the stability and scoreboard checks above, with intervals, before anyone reads them as skill.

## 6. Recommendations

1. Ship the raw XGBoost baseline as the first deployable model. Score it in Python with native Booster, 1 thread, explicit `iteration_range`, plus the pure-Python walker as a parity reference.
2. Do not deploy until the batch gate in section 4 is run.
3. Build player credit in two separate ledgers: model credit (deployable) and realized credit (outcome-based, post-hoc only).
4. Report player credit only within a match, with intervals, and only after the section 5.4 checks pass.

## 7. Unverified list

- ONNX and treelite install weight and latency (not installed; not timed).
- Effect size of omitting `iteration_range` (not tested).
- Monotone-constraint grid test on kill credit (not run).
- Plant-tick calibration (not checked).
- Training NaN count (not checked).
- Assist fraction, trade window, alert thresholds (proposals).
- Paper numbers from the fetch extraction (XGBoost LL/Brier/AUC, 4,682 matches, 100 bins, 479 players); quotes need checking against the PDF.
- Whether the tome carries `player_hurt` health fields for ADR.
- Whether 16-round matches survive the training filters.
