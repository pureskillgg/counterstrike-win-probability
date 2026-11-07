# Economy snapshot prototype (TASK D)

Date: 2026-10-13. Game: CS2 (not CS:GO). Scope: 40 sampled 5v5 matches, 1,045 rounds.
Main script: `scripts\eda\economy_snapshot.py`. Outputs (G:, not E:): `logs\eda\economy_snapshot_results.json`, `..._rounds.csv`, `..._rounds_buyend.csv`, `..._round_diag.csv`, `..._e_timing.csv`, `..._matches.csv`.
Run log: `logs\eda_economy_snapshot_run.log`. Nothing was written under E:.

## 1. Findings

1. **Team economy at round_freeze_end predicts CT win.** Univariate AUC for CT win (ct minus t, match-cluster bootstrap 95% CI): equipment value 0.621 (0.589 to 0.652); armor count 0.621 (0.590 to 0.654); money 0.533 (0.492 to 0.573). Helmets 0.619, rifle or AWP count 0.617, defuse kits 0.575 (extra). VERIFIED (run).
2. **The score sign flips with match-length selection, so the 40-match score result is not usable as is.** Round-start score difference (ct_score minus t_score) gives AUC 0.405 (0.384 to 0.422) in the 40 sampled matches. The sample is restricted to matches with at least 24 rounds, which selects on a future outcome (whether the match lasts to round 24). In the 24-round format (1,602 matches, 33,919 rounds), the same AUC is 0.568 for all matches, 0.616 for the 1,242 matches that ended early (24,114 rounds), and 0.415 for the 360 that reached 24 rounds (9,805 rounds). VERIFIED (`economy_probe_score_by_format.py`, `economy_probe_score_subsets.py`). Do not read the score sign off the 40-match sample. The same selection may also affect the economy AUCs above; they were measured only on the selected matches.
3. **Economy adds to a constant guess, and adds to the score control.** Per-fold grouped CV (5 folds by match):
   - Constant (alive difference is 0 at round start): logloss 0.690, Brier 0.248.
   - Economy 3 (equipment, money, armor count diff): fold AUC 0.619 (sd 0.027), logloss 0.662, Brier 0.235. Logloss gain over constant 0.028 (cluster bootstrap 0.012 to 0.041).
   - Score only: fold AUC 0.600 (sd 0.014), logloss 0.679.
   - Economy 3 plus score: fold AUC 0.661 (sd 0.036), logloss 0.645, Brier 0.228. Over score only: AUC +0.06, logloss gain 0.035 (0.018 to 0.049).
   - Economy 3 over score only, logloss gain 0.017 (-0.001 to 0.034). Not clearly positive on its own.
   - Holdout (10 matches, 268 rounds, seed 0): constant logloss 0.700; economy 3 AUC 0.591, logloss 0.678; economy 3 plus score AUC 0.662, logloss 0.648. VERIFIED (run).
   - Calibration of economy 3 (OOF, 5 bins): predicted vs observed 0.33 vs 0.27 (n=101), 0.52 vs 0.55 (n=612), 0.67 vs 0.63 (n=303), 0.82 vs 0.83 (n=29). Extreme bins are thin. VERIFIED (run).
4. **Equipment and money at freeze end are close to buy-end values.** On the 364 rounds where both snapshots are complete, the correlation is 0.96 to 0.99 and mean changes are under about 1k. Buy-end econ3 is slightly better (fold AUC 0.605 vs 0.586; logloss 0.666 vs 0.677). The differences are small and no test was run on 364 rounds. VERIFIED (run, `economy_probe_freeze_vs_buy_levels.py`). Whether freeze-end already contains all pre-round purchases was NOT checked (it would need pre-freeze player_status rows, which were not read).

Bottom line for the model: team economy at freeze end is a real pre-live signal (AUC about 0.62 alone; economy 3 has fold AUC above 0.5 in all five folds, 0.57 to 0.65). The score lead is informative in the full 24-round format (AUC 0.62 when the match ends early), but its sign in this 40-match sample is an artifact of the length rule. Economy plus score scored best here (fold AUC 0.66), but its score coefficient is not trustworthy until the snapshot runs on a sample that is not selected on match length. The next step is to run the same code on a random sample of all matches of each format, which needs more E reads than this task allowed.

## 2. Method and sample

- Eligible: header is_wingman false; match key day 01 to 07; at least 24 rounds (round_end rows, unique round numbers); round_end.player_count = 10 in every round. 48 eligible matches. Sample: `random.Random(0).sample` over sorted eligible keys, 40 matches (`logs\eda\economy_snapshot_matches.csv`).
- Snapshot: player_status rows at the round_freeze_end tick (basic_round_state). Nearest player_status tick used if no exact row. Exact-tick match in 1,045 of 1,045 kept rounds (max gap 0).
- A round is kept only if the snapshot has 5 T and 5 CT. 1,045 of 1,046 freeze rows kept (1 dropped: 5 T and 4 CT, 9 players).
- Features per side: sum of equipment_value_calc; sum of money; count of armor > 0; count of has_helmet; count of has_defuser; count of rifle or AWP. Rifle or AWP set: inv_primary 7, 8, 9, 10, 11, 13, 16, 38, 39, 60 (VERIFIED: names checked against basic_weapon_fire.weapon_name for the 40 matches via `economy_probe_weapon_codes.py`; 8 aug, 11 g3sg1, 38 scar20, 39 sg556 were not in the older notes table). SSG 08 (40) excluded.
- Label: winner_team_code 3 = CT win (round_end).
- Team membership (see section 4): basic_player_info team_code, joined on match, round and player_id. Zero duplicate keys.
- Models: logistic regression with standardization, C=1. Constant baseline = training prior. Evaluation: 10-match holdout plus GroupKFold(5) by match; cluster bootstrap (1,000 resamples of matches). Use per-fold AUC. The pooled out-of-fold AUC of the constant model is 0.45 only because fold priors differ, so it is an artifact.
- Sample CT win rate: 55.2% (small sample; full collection 52.4%).

## 3. Selection and reads

- E reads: 40 player_status files, one read each, 13 projected columns (about 57% of compressed file bytes). Mean file 5.8 MB.
- Earlier probe (`economy_probe_schema.py`) listed the day-01 folder on E and read the footer and folder listing of match 0f044ff9, which is NOT among the 40. Metadata only, zero rows read. Disclosed.

## 4. Deviations from the task text (please note)

- **Team membership came from the G-side basic_player_info tome.** The task limited reads to player_status, round_state and round_end. Raw player_status has no team column (schema checked; the raw columns are tick, round, player_id, money, armor, inv_*, equipment costs, has_* flags and similar, with no team). I used the G-side tome, not E:. VERIFIED.
- **Other G-side reads:** basic_weapon_fire filtered to the 40 matches (the task named this as a check for weapon codes); round_state buytime_ended rows for the buy-end comparison; the header. No further E reads.
- **Score fields are team-attached, not side-attached.** For each of 1,046 freeze rows in the 40 matches, ct_score equals the prior-round wins of the team on CT, and t_score equals the prior-round wins of the team on T (1,046 of 1,046 for both). A side-cumulative reading matches only 61%. feature_spec.md section 5 labels ct_score_minus_t_score as "side-attached"; that label should read "team-attached (value follows the team, indexed by current side)". VERIFIED (`economy_probe_score_semantics.py`).
- **Halves are 12 rounds (MR12), not 24.** All 40 sampled 24-round matches flip team sides at round 13 (`economy_probe_halftime.py`). The task text said "CS2 uses 24-round halves, MR12"; the data show MR12 means 12 rounds per half, with 24 regulation rounds. VERIFIED.

## 5. Buy-end comparison and data-quality finding

- buytime_ended comes 1,281 ticks (20.016 s at 64 per second) after round_freeze_end: median, p10 and p90 are all 1,281 ticks across the 1,045 rounds with a buy tick. VERIFIED (`economy_snapshot_round_diag.csv`).
- The player_status rows at the buy-end tick are NOT complete: 681 of 1,045 rounds have 3 to 9 players at that exact tick (median 8). The freeze tick is complete in 1,045 of 1,046 rounds. player_status is therefore not a complete per-tick panel. A full build must choose ticks with a complete 10-player set, not just the nearest tick. VERIFIED (`economy_probe_buy_drops.py`, round diagnostics).

## 6. Score sign across subsets (VERIFIED, `economy_probe_score_by_format.py`, `economy_probe_score_subsets.py`, `economy_probe_score_all_rounds.py`)

| subset (max_rounds 24 unless stated) | matches | rounds | AUC of ct_score - t_score for CT win |
|---|---:|---:|---:|
| all matches, all formats | 1,751 | 35,953 | 0.568 |
| all matches, max_rounds 24 | 1,602 | 33,919 | 0.568 |
| max_rounds 24, ended early | 1,242 | 24,114 | 0.616 |
| max_rounds 24, reached 24 rounds (the task's rule) | 360 | 9,805 | 0.415 |
| the 40 sampled | 40 | 1,046 | 0.405 |

The length rule is the cause: the selected subset has the opposite sign. Any mechanism (for example loss-bonus mean reversion) is UNVERIFIED. Matches that reach 24 rounds are selected on the future outcome, so this subset should not be used to estimate the score effect.

## 7. CS:GO vs CS2 caveats

- Format: CS2 here is MR12 with 12-round halves (section 4). The literature in `notes\research` is CS:GO-era (Xenopoulos 2020 and ESTA, per the notes). Its round format and economy rules may not transfer; the literature was taken from summaries and not opened by me (UNVERIFIED).
- Economy tuning: CS2 prices and loss-bonus rules may differ from CS:GO (UNVERIFIED here). The equipment_value_calc values are the ones recorded in the data, so the AUCs above are CS2 data results. The loss-bonus explanation for the score sign is UNVERIFIED.
- The score sign depends on the length rule (section 6), not on the format alone. Do not take a sign from this sample, from CS:GO, or from CS2 without a non-selected sample.
- Ticks: the demo tick rate is 64 (feature_spec: second = tick / 64; freeze ticks land exactly on player_status ticks in 1,045 of 1,045 kept rounds). CS2 sub-tick updates may make the tick stamp coarser than event timing; not checked here (UNVERIFIED).

## 8. Cost for all 1,752 matches

- E read time per match (open, read and to_pandas), three runs: run 1 cold (first touch of these files) 17.1 s for the first match, then median 0.137 s; run 2 warm median 0.068 s; run 3 warm median 0.043 s, p90 0.068 s.
- Full build estimate for the file reads (1,752 matches): about 1.2 min warm (0.043 s x 1,752), about 4 min cold (0.137 s x 1,752), plus the one-off first-touch cost. Listing an E day folder cost 16.9 s cold; building paths from match keys avoids listing.
- Volume: player_status mean 5.8 MB per match, about 10.2 GB in total; projected columns about 3.3 MB per match (about 5.8 GB).
- The snapshot step (group-by and nearest-tick selection) was not timed separately. The whole 40-match run took about 100 s wall, but the E reads total about 2 s of that. The rest (pandas steps, the buy-end pass, bootstraps and fits) was not split out. The full-build cost of the snapshot step is UNVERIFIED.

## 9. Not verified

- Literature claims (economy as strongest pre-round signal; Xenopoulos 2020 gain shares; ESTA feature list) are from notes\research and were not checked against sources.
- Loss-bonus or mean-reversion explanation of the negative score sign.
- Whether freeze-end equipment already includes all purchases made before round_freeze_end (pre-freeze rows not read).
- Whether the 8, 11, 38, 39 rifle codes hold for other days (checked only in these 40 matches).
- Cold-cache timing beyond the first run; E I/O throughput on a fresh disk.
- Model quality beyond 40 matches: the holdout has 10 matches, and the bootstrap CIs are wide.
- Whether the economy AUCs (about 0.62) survive without the match-length selection. They were measured only on the 40 selected matches (section 1, item 2).
- Whether the score-field team attachment holds outside these 40 matches (checked on 1,046 rounds only).
