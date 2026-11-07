# Review: the 57,858-match training export (export_clean_v2)

Written Nov 6, 2026. Reproduce every number with [scripts/data/review_export.py](../../scripts/data/review_export.py).

**Verdict: not ready for review as a training or evaluation set.** The stored fields hold no personal data. The split labels have problems that the owner has to resolve first.

## 1. What is being reviewed

- **Archive:** `export_clean.tar`, 302,520,320 bytes, written 4 Nov 2026. It is on this laptop at `C:/Users/<user>/AppData/Local/Temp/`. Temp folders can be cleared, so copy it somewhere durable before anyone relies on it.
- **Contents:** per-second alive counts (72,630,483 seconds); per-round freeze and plant ticks, label, match index, round number and map (1,059,624 rounds); per-round loadout totals (30 columns); 57,858 match keys; split codes (`match_split.npy`).
- **Self-check written with the export:** zero mismatches against the grid on the checked fields (second, bomb planted, seconds since round live, seconds since plant, label).
- **Dates:** 23 Nov 2023 to 25 Oct 2026, with a gap from Aug 2024 to Dec 2025. 23,683 of the 57,858 matches are from October 2026.
- **Full copy on the remote space:** not checked from this laptop. The copies were made with scp.

## 2. Personal data

- **Stored fields:** no player names, Steam IDs or account IDs. Loadout columns are per-side counts and totals (`loadout_names.json`).
- **Match keys:** opaque paths of the form `csds/YYYY/MM/DD/<uuid>/csds`. The upload date is in the path.
- **Duplicate player IDs:** `loadout_report.json` flags 934 rounds with a duplicate player ID. The IDs were used to build the loadouts and are not stored. The build script was not part of this review, so confirm that there.
- **Raw tomes are different.** `pipeline/sagemaker/holdout_build.py` names raw tables `player_name` and `player_personal`. The raw data is on G:, outside the repo, and it is where personal data lives.
- **Repo scan (61 files since `196ba15`):** no credentials and no player-level data. Three things to clean up before external review:
  - your email in `pixi.toml` under `authors`;
  - your Windows username (`<user>`) in the default paths of 18 files;
  - pod, bucket and S3 identifiers in `sequence/notes/provenance_2026-11-06/PROVENANCE.md`.

**Verdict on personal data:** none in the export as stored.

## 3. Split problems (blocking)

1. **Frozen test matches are labelled as training.** 2,372 of the 3,059 keys in `data/benchmark/frozen_test_keys.json` are in the export. Their labels: 1,891 train, 244 validation, 237 test.
2. **The split codes disagree with `split_manifest_v2`.** The export and the manifest copy agree on 39,226 of 57,858 keys. No code-to-label mapping fixes this. `docs/final/MIRAGE_TEST_DEFINITION.md` says the codes agree on all 57,811 keys, so either there are two manifest versions or one is stale. The owner has to say which.
3. **The counts do not reconcile.**

   | Source | Matches | Train | Validation | Test |
   |---|---|---|---|---|
   | Export codes (`match_split.npy`) | 57,858 | 46,288 | 5,785 | 5,785 |
   | `MIRAGE_TEST_DEFINITION.md` | 57,811 | 47,501 | 4,633 | 5,677 |
   | `split_manifest_v2` copy, for the export keys | 57,858 | 47,538 | 4,637 | 5,683 |

4. **The GDT window is training data in this export.** Matches dated 1 Sep to 8 Oct 2026: 3,542. Labels: 2,874 train, 332 validation, 336 test. The 83-input model was scored on this window, so the export does not hold it out.
5. **The scorer's test uploads are training data in this export.** Matches dated 14 to 25 Oct 2026: 11,668. Labels: 9,328 train, 1,172 validation, 1,168 test.
6. **Holdout B and October 26 onward are clean.** None of the 653 holdout B keys are in the export, and no match is dated 26 October or later.

**Consequence:** a model trained on the export's train split and scored on the frozen test, the GDT window or the 14 to 25 October uploads would be scored on data it trained on. The planned full-archive XGBoost sweep on this export (`docs/final/MODELS_TO_RUN.md` row 8, `handover/deep-model/README.md` row 8) inherits this. Do not run it until items 1 to 5 are resolved.

## 4. Requests to the reviewer

1. Confirm the split of record: which `split_manifest_v2` copy, or the export's own codes.
2. Confirm the frozen test definition (`data/benchmark/frozen_test_keys.json`) and whether the 14 to 25 October uploads are the scorer's test set.
3. Reconcile 57,858 with 57,811 matches.
4. Confirm the duplicate player-ID handling in the build script, and that no identifier reaches the export.
5. Decide where the export lives, and copy it out of Temp.
6. Decide on the repo cleanup: the email in `pixi.toml`, the username in paths, and the pod and S3 identifiers.
