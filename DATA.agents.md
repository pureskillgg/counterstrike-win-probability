# DATA (agent notes)

Human summary: [DATA.md](DATA.md). Retraining recipes: [TRAINING.agents.md](TRAINING.agents.md).

## Roots

- Data root `CS2_ROOT` (default `G:\pureskill\csds-cs2`). `data/` there is a junction to `C:\Users\<user>\Documents\pureskill_internship\csds-cs2\data`; `tomes/` is a junction too. Backup: `C:\Users\<user>\Backups\csds-cs2_2026-11-06` (tree, 31.7 GB) plus sibling copies of `data` (78.9 GB) and `tomes` (351 GB). The E: junction targets (989 GB) were not copied.
- Raw: `E:/csds_final/csds/YYYY/MM/DD/<match_id>/` (43 files per complete match).
- RunPod volume `z7yvc40ykx`, S3 endpoint `https://s3api-eu-ro-1.runpod.io`, region `eu-ro-1` (credentials are local, never in git). Prefixes: `cs2data/` (uncompressed training inputs: combined caches, pos24 site grid, keys), `cs2data_gz/` (gzipped `data/combined`, `data/diagnostics`, `data/round_state`; manifest with sha256 per file; 29.7 GB raw), `export_clean_v4/` and `v5/` (57,811-match export arrays), `results/` (run metrics, predictions, weights backup), `models/gdt_objective/`, `pod_runner/`, `gdtscore/`, `envs/`, `repo/`.

## Pipeline (tomes to model inputs)

| Step | Script and command | Output |
|---|---|---|
| Tomes | `G:\pureskill\envs\dsdk-fork\Scripts\python.exe -I -X utf8 scripts\data\build_tomes_incremental.py --raw <folder> --out tomes\<name>` | `tomes/<D>` or `<D>_cNN` chunks (~500 matches), `BUILD_OK` |
| Day queue | `scripts/data/extend_tomes_chunks.py --days 2026-10-14,... [--chunk 500]` (uses `tome_ledger.py`) | `tomes/D_cNN`, `D.CHUNKS_DONE` |
| Combined store | `scripts/data/build_combined.py` (full) / `append_combined.py` (incremental; ledger `tomes/combined/COMBINED_BUILD.json`, 68 sources) | `tomes/combined` |
| Match and round catalog | `scripts/data/prepare.py --store combined` | `data/combined/matches.parquet`, `rounds.parquet`, `prep_report.json` |
| Per-second grid | `scripts/data/make_grid.py --store combined` | `data/combined/grid.parquet` (rewritten 2 Nov) |
| Player status 1 Hz | `scripts/data/extract_status_1hz.py --store combined [--era 2026] [--raw-root E:\csds_final\csds]` | `data/status_1hz/<day>.parquet` |
| Positions 1 Hz | `scripts/data/extract_positions_1hz.py --store combined` | `data/positions_1hz/<day>.parquet`, `_plants/` |
| Position features | `scripts/data/make_pos_features.py --store combined --every 2 [--centre_keys data/benchmark/centre_fit_train_keys.json]` | `cache_pos_2026.parquet` (centres on all matches: test-leaking) or the `_trainctr` variant |
| Every-second positions | `scripts/data/make_pos_features_every1.py` (same flags) | `cache_pos_2026_trainctr_every1`, `data/pos_features_1hz/every1_trainctr/` |
| Snapshot cache | `scripts/model/make_snapshot_cache.py --store combined [--era 2026] [--out-name ...]` | `cache_snapshot_2026.parquet` (XGBoost); `_ext` (sequence, Oct 14-21 additions, writer run not recorded); `_oct23_31` (holdout) |
| Round-state arrays | `scripts/model/build_round_state_map.py --map <map>` | `data/round_state/<map>[_ext].npz` (X N x 10 x 6, split, match_idx, round, second), `.json` |
| Sequence position grid | `scripts/model/build_pos24_maps.py` (stride-2 version saved; `_grid` every-second version not saved) | `data/diagnostics/allmaps_pos24_trainsites[_grid]/<map>.parquet` |
| Split manifest | `scripts/model/split_manifest.py --freeze | --extend | --extend-train | --verify` | `data/combined/split_manifest.json` (see TRAINING.agents.md for versions) |
| Other grids | `build_grid_step.py --step 16`, `build_random_tick_pool.py --seed 0`, `build_random_tick_pool59[_daily].py --seed 0` | tick-rate and random-tick experiments |

Tome naming: `D` = upload day; `D_cNN` = chunk; `D.CHUNKS_DONE` lists finished chunks; `D_to_NN` = range built whole; `YYYY-MM` = a month. Folder list: `data/manifests/tome_folders.txt`. `2026-10-14_to_19` is not in the combined ledger (its per-day chunks are); `2024-02` and `2024-02-all` are not either (only `2024-02-g`).

Two date clocks: the match key gives the upload day; `match_date` in the tables is the processing day for hand uploads (`wp_common.py` comment). Do not mix them.

## Key files in this repo

- `data/benchmark/frozen_test_keys.json`: 3,059 frozen-test keys, uploads 2026-09-01 to 10-13 (43 days).
- `data/benchmark/holdout_22_25_test_keys.json`: holdout B, 653 keys, 2026-10-22 to 25.
- `data/benchmark/split_manifest_a17f219b.json`: the manifest v4-v8 recorded (17,563 entries).
- `data/benchmark/split_manifest.json`: v1-archive manifest (29,885 entries).
- `data/benchmark/centre_fit_train_keys.json`: the 10,577 keys used to fit train-only site centres.
- `data/manifests/`: per-day match counts per table and split, round-state build metadata, tome folder list (see its README).

## Personal data audit (6 Nov 2026)

Scope: 1,493 parquet files on the data root and the S3 mirror; 1,353 value-screened (counts only); frozen, holdout and Oct 26-31 rows excluded by rule. Findings:
- No SteamID64/2/3, email, IPv4, Windows user-path or URL values in screened tables; no name, steam, account, email, chat, IP or path columns.
- Pseudonymous identifiers: `player_id`, `attacker_id`, `player_controller_id`, `player_id_fixed` (per-match), `match_key` (resolves to `E:/csds_final/csds/...`). `deaths.parquet` `player_id` can act as a per-player pseudonym.
- Raw tomes: `basic_player_personal` (33M rows scanned), `basic_player_name`: name and clan tag hold one placeholder value (`redacted`), steam_id holds one-letter codes. `header.client_name` and `header.server_name` are free text, not classifiable. An unlisted tome copy in `G:/pureskill/csds-cs2/evac_2026-10-13/tomes` has `header.game_directory` with `/home/<user>/` paths (904 distinct usernames).
- Not screened: the full E: store (36,895 `player_personal` files; 20-match sample only), some tome tables (`basic_player_inputs`, `_footstep`, `_sound`, grenade, item and weapon tables), git-history contents, export_clean_v4/v5 contents.
- Repo (public): scrubbed on 7 Nov 2026. Author emails, the Windows account name, pod IPs, pod ids and the AWS account number are removed; match keys are pseudonyms (`csds/YYYY/MM/DD/m_<24 hex>/csds`), restorable with `scripts/data/restore_match_keys.py`; `scripts/data/check_repo_privacy.py` reports clean. The Steam64-shaped number is a synthetic base value in `scripts/model/stress_test_winprob.py`. Still named in the repo, on purpose: the author's name, the GitHub login and noreply email, company bucket names, the SageMaker space name and domain resource name, RunPod volume ids, Dr Freeman's surname in docs, the default `sagemaker-user` name, Steam64 range constants used as screening thresholds, the folder name `pureskill_internship` in local paths, non-match uuids (Meridian project id, Claude session ids, a schema-example guid, Office-internal GUIDs), "Robert Freeman" as a bibliography author in `notes/research/csds_docs_facts.md`, and public cloud price lists. Commit ids quoted in older notes refer to the history before the scrub and no longer exist.
Audit scripts: `scratch/session_2026-11-06/` (`value_screen.py`, `tome_value_screen.py`, `pii_audit/`, `pii_refute/`).

## Protected batch and split checks

- `TRAINING_EXCLUDE.json` (RunPod `export_clean_v5/`): 2,848 keys, uploads 2026-10-21 (84), 22 (1,018), 23 (906), 24 (278), 25 (562). Status: lifted 5 Nov by chat decision; written owner confirmation not on file. Inside training populations: sequence (`*_ext.npz`) 84, all train; XGBoost `cache_snapshot_2026` 0.
- `docs/data/EXPORT_REVIEW.md`: export_clean_v2 (57,858 keys) labels 2,372 frozen-test keys (1,891 train); its codes agree with `split_manifest_v2` on 39,226 of 57,858 keys.
