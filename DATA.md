# Data

The models learn from CS2 match recordings processed into one row per second of each round. None of the data is in git because it's too large. This page says where it is, which days each model used, and what's known about personal data. Paths, build commands and table details are in [DATA.agents.md](DATA.agents.md).

## Where the data lives

| What | Where | Size |
|---|---|---|
| Raw per-match files | `E:\csds_final\csds` (USB drive) | about 990 GB, 1.74 million files |
| Tomes (raw data packed per upload day) | `G:\pureskill\csds-cs2\tomes`, backed up to `C:\Users\13144\Backups` | 351 GB |
| Processed tables (matches, rounds, per-second grid, caches) | `G:\pureskill\csds-cs2\data` (a link to `C:\Users\13144\Documents\pureskill_internship\csds-cs2\data`), backed up to `C:\Users\13144\Backups` | about 79 GB |
| Training inputs used on GPUs | RunPod network volume `z7yvc40ykx` (EU-RO-1): `cs2data/` and `cs2data_gz/` | 10.4 GiB of 37.4 GiB |
| Model handover | `s3://temp-model-transfer-0/adam/cs2-finished-models-v1/` (models only, temporary) | 83 MiB |

The raw E: drive isn't copied anywhere else. It's too large for any local drive.

## Which days each model used

Days are upload days. Each match key starts `csds/YYYY/MM/DD/`, and tome folders use the same dates. Full per-day counts are in [data/manifests/](data/manifests/README.md).

| Data | Upload days | Matches |
|---|---|---|
| XGBoost v3 to v8 and the 83-input model | 1 Sep to 13 Oct 2026 (43 days) | 14,804: 10,577 train, 1,168 validation, 3,059 test |
| Sequence models and the all-maps temporal models | 1 Sep to 21 Oct 2026 (51 days) | 23,716: 19,486 train, 1,169 validation, 3,061 test |
| Everything in the combined store | 21 days in Feb 2024, plus 1 Sep to 31 Oct 2026 | 37,119 |
| The training export (`export_clean_v4`/`v5`, not used by any model here) | 23 Nov 2023 to 25 Oct 2026 | 57,811 |

## Test and holdout sets

- **Frozen test:** 3,059 matches from the same 43 days, chosen at random by match. It's in `data/benchmark/frozen_test_keys.json`. Every XGBoost test score uses it.
- **Holdout B:** 653 matches uploaded 22 to 25 Oct, in `data/benchmark/holdout_22_25_test_keys.json`.
- **October 26 to 31 holdout:** 3,315 matches. It was scored once, for the 83-input model, and wasn't used for any choice.

## Personal data

An automated audit on 6 November found **no names, Steam IDs, emails, IP addresses or chat text** in the training tables. Names and Steam IDs are already redacted in the source. That doesn't make the data anonymous:

- Each row carries a per-match player number and a match key. The match key points back to the raw files on E:.
- The raw tomes hold personal tables (redacted), two free-text header fields, and install paths that include usernames. These weren't fully screened. The E: drive was only sample-checked (20 matches).
- This repo contains an author email (`pixi.toml`), the Windows account name and user paths in many scripts, match-key lists, and one Steam64-shaped number in a test fixture in the git history.

Treat the data as pseudonymous. Review the raw tomes before sharing anything beyond the processed tables.

## Open issues

- **Protected batch:** 2,848 matches uploaded 21 to 25 Oct needed owner sign-off before training use, and written sign-off isn't on file. 84 of them, all from 21 Oct, are in the sequence training data. None are in the XGBoost training data.
- **Export labels:** a review found that `export_clean_v2` labels some frozen-test matches as training, and its split codes disagree with the split manifest. See [docs/data/EXPORT_REVIEW.md](docs/data/EXPORT_REVIEW.md).
- **Split drift:** the split manifest on the laptop was extended on 4 Nov with a different rule. Use the one the models recorded; see [TRAINING.md](TRAINING.md).
- **9 and 10 Oct:** match counts for those days are unreconciled (see `HANDOFF.md`).
