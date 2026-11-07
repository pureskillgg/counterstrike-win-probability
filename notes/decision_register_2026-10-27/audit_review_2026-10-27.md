# Test-provenance audit (27 Oct 2026)

Status: draft for owner review. Nothing has been read from holdout B. No model has been retrained. No pins were added, changed or superseded. Times are Central (CDT). Pin times from the Meridian dump are UTC and have been converted.

## Verdict

The candidate B read (pin 56e6e642) should not run as pinned. Its own gate says that if this audit finds an earlier test read that changed a choice in B or A, the rule is re-pinned before the read. The audit finds exactly that:

- The 687-match de_mirage test was read directly by six result files, and more results read its rows under other keys.
- The distance, walking, defuse-race and word2vec conclusions were made on that test set. For the walking and defuse-race conclusions, no validation-only re-check is pinned.
- Candidate B was chosen after the test numbers for its two components (GRU and SN) were on record.

Recommended path: point candidate B at the fresh holdout B (653 matches, all maps, built and unread), retire the 687-match test for decisions, re-pin, and then give your go.

## 1. Test-set exposure

Each test number has to name its population. The records use several.

| Population | Size | Where it appears | Status |
|---|---|---|---|
| de_mirage frozen test | 687 matches (92,360 paired seconds; 865,974 all seconds) | `allmaps_xgb_geo`, `defuse_race_eval`, `emb_eval`, `nav_path_eval`, `nav_path_eval2`, `why_models_lag` (verified by key). Pins 7b4f7a87, 4f868e95, 428961a1, 93d46735, 28cfd049, 5ba17087 | Read repeatedly since 23 Oct |
| All-maps test, older freeze | 3,059 matches | `exp_full_comparison.json` (17 Oct, verified) | Read in the 17 Oct comparison |
| 2026 frozen test, by manifest | 3,611 before the 22–25 Oct extension; 4,264 after (3,611 + 653). Corrected from an earlier 3,714, which was wrong | `split_manifest.json` | Not read as a whole. The 653 new test matches are holdout B |
| 2024 February | separate set | `eval_2024_frozen` (25 Oct), `in_era_2024` (26 Oct) | Pinned as separate (afa62544, f4520d78) |
| Holdout B | 653 matches, all maps (155 de_mirage) | `data/combined/holdout_22_25_test_keys.json` | Built and features verified. Not scored |

**Ledger of the 124 unpinned result files** (full table in [unpinned_test_ledger.md](unpinned_test_ledger.md)):

- 41 name a test outcome directly (for example `test_logloss`).
- 47 carry test counts, but their outcome numbers sit under generic keys (`logloss`, `delta`). Each needs a per-file check.
- 36 carry no test keys.

So 88 of 124 touch test-split numbers.

**Timeline**

- 14 Oct: integrity rules pinned (9ec38dc9). They cover splits, bootstrap and exclusions. They set no limit on test reads.
- 15 Oct: first outcome-keyed test reports (for example `exp_robustness`).
- 23 Oct 15:56: first read of the 687-match test (`why_models_lag`).
- 24 Oct: all-maps GRU and paired distance comparisons on test (`allmaps_gru_vs_xgb` 19:06; `ext_temporal_paired_*` 11:54 to 17:40).
- 25 Oct: nav-path, defuse-race, embedding, walking-approximation, stride and map-control runs on test. Pins 7b4f7a87 and 5ba17087 at 10:54, 28cfd049 at 11:53, 93d46735 at 15:47.
- 26 Oct 08:23 to 08:47: `allmaps_xgb_geo` and its seed check on test. Pins 4f868e95 (08:26) and 428961a1 (08:47).
- 26 Oct 10:14: first validation-only pre-registration (56028667). From here on, decisions are pinned as validation-only.
- 26 Oct 23:36: candidate B pinned, with a read-once rule (56e6e642).
- 27 Oct: you chose B.

## 2. Findings

**F1. Distance, walking and embedding conclusions were made on the test set.**
- 7b4f7a87 (25 Oct, 10:54) says walking distance does not beat straight-line distance. The evidence is paired test results on the 687-match test (`nav_path_eval2`). In the pins I read, no validation-only check of walking against straight-line is pinned. W2 (8697407b) tests walking against the 57-input baseline on validation, not against straight-line.
- 28cfd049 (defuse-race margins, 25 Oct) comes from `defuse_race_eval` on test. Its 250 units per second rests on an unconfirmed source (Huang et al. 2026).
- 5ba17087 (word2vec, 25 Oct) comes from `emb_eval` on test.
- `walking_approx_eval` (25 Oct) also reads test rows (`planted_test_rows`).
- *Partly mitigated:* the straight-line distance claim was re-checked on validation with train-only centres (56028667, then ee61ba3a at 10:53: "distance gain holds"). The walking-versus-straight, defuse-race and embedding conclusions have no validation-only re-check.

**F2. The site centres were fitted with the test matches included.**
- `sitecheck_geo.py` says the centres are fitted "on every match's plant positions, including the frozen test". The same all-match centres appear in `build_nav_path_features.py:61`, `train_temporal_allmaps.py:59` (the GRU distance arm in 93d46735) and `build_nav_walk_allmaps.py:76`.
- So the walking and original straight-line distance features behind 4f868e95, 428961a1, 7b4f7a87 and 93d46735 were built with test-informed centres. Their numbers cannot be cited as clean test results.
- Candidate B is clean on this point. Its SN arm uses train-only centres (stated in 56e6e642), and its GRU has no distance features.

**F3. Candidate B was chosen with its components' test numbers already seen.**
- GRU test numbers existed on 24 and 25 Oct (`allmaps_gru_vs_xgb`; 93d46735).
- SN test numbers existed on 26 Oct, 08:26 to 08:47 (4f868e95, 428961a1).
- B was pinned on 26 Oct at 23:36 CDT and chosen by you. Its validation result (993fc8bb) was pinned before the choice.
- So a read of the 687-match test cannot be a clean confirmation of B.

**F4. The pinned read scope is stale.**
- 56e6e642 names the de_mirage test (865,974 seconds, 687 matches) as the read population. You chose holdout B (653 matches, all maps) on 27 Oct. The pin has not been updated, and its gate requires a re-pin.
- The body says "Owner's choice on 27 Oct", but the pin was created 26 Oct 23:36 CDT (27 Oct 04:36 UTC). Clarify the date.

**F5. "Frozen" is ambiguous about the training manifest.**
- The all-maps scripts (`allmaps_xgb_geo.py:48`, `allmaps_xgb_geo_seedcheck.py:47`, `sitecheck_xgb.py:58`, `sitecheck_geo.py:64`) read `data/combined/split_manifest.json`. On 26 Oct that file was the pre-extension state, with 21,468 training matches. The backup `_backup_pre_22_26_20261027` has the same count. The arithmetic checks out: 12,563 + 8,905 = 21,468, where 8,905 is the 23 Oct `extend_train`.
- The shipped 57-input model (`models/winprob_v6_drop_heavy.meta.json`) records manifest a17f219b, with 12,563 training matches.
- So the "frozen recipe" pins (4f868e95, 56028667, dc66a027) were trained on 21,468 matches, not on the shipped model's 12,563. dc66a027 says 21,468 is frozen, which is right for the all-maps runs and wrong for the shipped model. A correction pin should say which.
- Checked by counting rows (27 Oct, see `manifest_check_2026-10-27.md`). The saved all-maps row counts (25,582,731 train; 1,515,228 val; 865,974 test; 687 matches) match the 21,468-entry manifest exactly. The 12,563-entry manifest gives 13,758,901 train rows and does not match. The current manifest also matches, because its 3,417 new entries have no rows in the snapshot yet, so the counts cannot separate the two. The manifest history puts the 21,468 version in force on 26 Oct. Most likely input: the 21,468 version. Still unconfirmed: the JSON records no manifest hash.
- In-scope training matches (with rows in the seven-map join) are 19,478 for the 21,468 version, and 10,577 for the 12,563 version. The pin's 21,468 counts manifest entries, not matches with data.

**F6. The reproduction gate fails on CPU, and the GPU is busy.**
- The S4 CPU retrain missed the 1e-4 prediction gate. Validation log loss was 0.4533433 against a saved 0.4533213 (2e-5 apart), but max |Δp| was 0.126 and mean |Δp| was 0.0038.
- Close loss with large individual differences looks like a different random subsample. The CPU and GPU samplers differ under `subsample` 0.8. That is a hypothesis, not yet tested.
- The GPU is held by another project's WSL job (adam, pid 599). Stopping it is not ours to decide.
- If the gate is changed after it failed, that is a post-hoc change and must be pinned before the rerun, with the failure disclosed. Keeping the gate and waiting for the GPU is cleaner.

**F7. Holdout B has 11 duplicated matches.**
- 11 holdout matches have non-keeper copies that sit in no split. They do not leak into training. They double-count if every copy is scored.

**F8. Stale and inconsistent text.** See section 4.

## 3. Before any read of holdout B

1. You pick the population (D1). Recommended: holdout B.
2. Duplicates are resolved (D2).
3. The SN and A reproduction gate is decided and met for seeds 0 to 2 (D3). The GRU already reproduces on validation for seeds 0 to 2.
4. Holdout inference is wired. The GRU join reads the `data/combined` cache, which has no holdout rows. Holdout distances for SN are built (`h_05_geo`).
5. 56e6e642 is re-pinned, or superseded, with: the holdout population and key-file hash; the reproduction results; this audit's findings; the read-once rule; and the scope.
6. You give the go. Gate 1 of 56e6e642 (the audit reviewed by you) is met when you review this document.

## 4. Editing list

**Register and papers**
- `README.md`: the "no read has run" line is still true. Add that holdout B is built and that the reproduction status is as above.
- `README.md`: the claim that `exp_model_families.json` reports 3,059 test matches is not verified. `exp_full_comparison.json` is verified.
- `README.md`: the owner-approval list includes 2702982d, which is a request, not an approval. 56e6e642 is a gate. 2569aac7 is open. Label them.
- `papers_status.md`: remove the "Correction to the README" section. The README already says 4 papers tested.
- `papers_status.md`: Szmida and Toka year. Use "2025 by paper ID, 2026 by listing" to match the deck.

**Deck (22 slides)**
- `sources-open` (slide 18): change the Szmida year to "open: 2025 by paper ID, 2026 by listing".
- `test-set-issue` (slide 21) and `next-steps` (slide 4): add the 88-of-124 count, the six direct reads of the 687-match test, and holdout status. Not reviewed in this pass.
- `daily-log` (slide 22): no change found.

**Register doc (MYgtho68m9mcALBdDQBQFB, rev 11)**
- Add the audit results and correct "21,468 frozen" to say which model.

**Meridian pins to add (proposed, not written)**
- Correction to 56e6e642: population and gates (F4).
- Correction to 7b4f7a87: walking-versus-straight was decided on test, and a validation check is pending (F1).
- Correction to 4f868e95 and 428961a1: centres were fitted including the test set. The distance claim was re-checked on validation in ee61ba3a (F2).
- Correction to dc66a027: which runs used 21,468 and which model file uses 12,563 (F5).
- Retirement note: the 687-match test is retired for decisions and kept for descriptive reporting only (D5).

**Result files**
- `unpinned_results.md`: label the 88 test-touched files "descriptive, test-touched". Label the 36 others "no test numbers". Check the 47 count-only files one by one.

## 5. Decisions for you

- **D1. Population for the candidate-B read.** Recommended: holdout B (653 matches). The alternative is the 687-match de_mirage test, which is not recommended. It has six direct reads, and B was chosen with its components' test numbers visible.
- **D2. Holdout duplicates.** Recommended: score the keeper copies only. The alternative is to score all copies.
- **D3. Reproduction gate for SN and A.** Recommended: keep the 1e-4 prediction gate and run on the GPU once it is free. If you want a metric-level gate on CPU, pin the new gate before the rerun and disclose the S4 failure.
- **D4. The 88 test-touched unpinned files.** Recommended: label them descriptive. The alternatives are pin or retire.
- **D5. The 687-match de_mirage test.** Recommended: retire it for decisions and keep it for descriptive reporting only.

Still open from before, and not decided here: whether 2024 enters any pre-registered test (recommended no for the final read), and the split ratio (2569aac7).

## Method and limits

- **Ledger.** A script read all 124 files listed in `unpinned_results.md`. "Outcome-keyed" means a test key with a loss, accuracy, delta or pass-type name. "Count-only" means test counts with no named outcome. Key-name matching is a heuristic. The 47 count-only files were then checked one by one (27 Oct): 46 report a test outcome under generic key names, and 1 (`device_parity_check.json`) is a CPU and GPU agreement check. See `count_only_check_part1.md`, `part2.md` and `part3.md`.
- **Pins.** Read from a Meridian dump filtered to pins containing "test". Only the pins named above were read in full. The rest are from the titles file.
- **Code.** Checked by grep for the manifest path and centre fitting. No code was run.
- **Not checked.** The `exp_model_families` claim. The 3,714 count (from the 27 Oct notes). The GRU prediction manifest. The training keys behind the SN prediction files.
