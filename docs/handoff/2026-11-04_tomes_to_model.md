# Handoff: tomes done, model next (2026-11-04)

## Status

- CPU tome work is done. The event-only state table is **not** built yet. The GPU phase has not started.
- Not ready to run training yet. The blockers are at the end of this note.

## Where the data is

All paths are in `s3://pureskillgg-csgo-production-sagemaker-0/`.

- **`adam/tomes/chunks/cNNN/`** (c000 to c067): basic event tomes for 67,224 matches, about 46.8 GB. Each chunk has the header tome, a `DONE.json` marker, and 31 `basic_<channel>.<dates>` folders (`dataframe_*` pages and `keyset_*` files). Chunks 0 to 45 are older-format matches, chunk 46 is July 2026, and 47 onward are Oct 9–25 newer-format matches.
- **`adam/tomes/header_build/`**: header tome for all 67,224 matches. Keys match the admin's list exactly. `max_rounds` is filled for 20,597 matches only (16 for 250, 24 for 20,347). The other 46,627 matches, from 2023-11-23 to 2026-07-10, are null.
- **`adam/tomes/holdout_2026-10-26_31/`**: holdout tomes for 3,463 matches, verified, with `match_keys.txt`.
- **`adam/tomes/harmonized/molotov_state/`**: `c000.parquet` to `c067.parquet` plus `summary.json`. Use these instead of the raw chunk `molotov_state`. Sentinel columns are cast to the admin's types: `Int64` for `was_extinguished_by_smoke`, `extinguisher_id` and `smoke_entity_id`, and `Float64` for `extinguisher_id_fixed` and `was_extinguished_by_thrown_smoke`. The -1 and -2 sentinels are kept.
- **`adam/checks/holdout_dup_check/`**: `summary.json`, `cross_pairs.csv`, `internal_groups.json`.
- **`adam/code/`**: pipeline scripts. `chunk_build.py` is the current version. `chunk_build_v1.py` is the old one. Also `holdout_build.py`, `harmonize_molotov.py`, `run_pipeline.sh`, `dup_check.py`, `header_full.py`, `tome_test.py`.
- **Repo**: branch `tome-build-pipeline`, commit `4f641dd`, has the early scripts and `docs/tome_build/TOME_BUILD_NOTES.md`. The later patches are only in `adam/code/`.

## Data gaps

- **Not in AWS:** 2026-07-11 to 2026-10-08. That covers July 11–31, August, September and October 1–8. The team has some of this locally, including September matches in the frozen list, on the E: drive.
- **Holdout duplicates:** match `csds/2026/10/28/m_050fafc3594bb51ae5ec231b/csds` duplicates two archive matches, `2026/10/17/m_79ee2999…` and `2026/10/17/m_b7a4fb4c…`. The holdout also has 56 internal duplicate groups covering 113 matches.
- **Frozen list vs archive:** the frozen list has 3,059 matches, of which 1,549 are in the archive. The other 1,510 are missing: 449 from September and 1,061 from October. None of the holdout matches are in the frozen list.
- **Unreconciled days:** 2026-10-09 and 2026-10-10 are marked unreconciled in the main chat's handoff. They make up 2,598 matches in our archive.

## Type notes

- `molotov_state` types differ across chunks. Chunks 44 and 45 are `Int64`, chunk 46 is `Float64`, and chunks 47 and up are `Boolean` for some columns. Read from the harmonized files to avoid this.
- Chunk 46 was built with the sentinel cast. Its values match the admin's tome exactly for one checked match.

## Not built yet

- **Event-only state table (10 features).** Build it from the chunk tomes by reading all chunk folders and unioning them at read time. The feature logic is the bake-off's `build_states` and `feature_validation`. Patched copies of both are on the space at `~/pipe/scripts/model/`, but their `TOME` and `SUFFIXES` still point at the admin's tomes, so they need repointing to the chunks.
- **Per-tick features for the 59-input base.** Draw K=64 random ticks per round, with the 13 event inputs at each tick plus 46 loadout features from player status at those ticks. Relevant code: `scripts/data/build_random_tick_pool.py`, `build_random_tick_pool59.py` and `features_status.loadout_features`. The team's draws are in `data/combined/random_tick_pool_s0.parquet`, which is on the laptop, not in AWS.
- **Transformer:** on hold. The team's loader places test rows on the GPU.

## Protocol (from team messages)

- Choose the model on validation only. When there's one final candidate, tell the admin and Freeman, then score it once on the Oct 26–31 holdout.
- Holdout B is contaminated, per the owner's message. Re-pin or record as contaminated: open.
- Loader change is approved, as long as it only stops test rows from loading. Before running the transformer, rerun the 59-input base on validation with the new loader and confirm the loss hasn't changed.
- Freeman suggests evaluating by situation: alive count, economy, time left, and whether the bomb is planted.
- Holdout deduplication policy: open. The proposal is to drop the cross-duplicate and keep one match per internal group.

## Infrastructure and rules

- **Space:** `adam-gpu` (SageMaker domain `d-<domain-id>`, user profile `adam`). It's currently on `ml.c6id.4xlarge` with the CPU image.
- **Storage:** `/home/sagemaker-user` is a 1 TB EBS volume. `/mnt/sagemaker-nvme` is NVMe scratch, wiped on stop.
- **Connecting:** see "Connecting over SSH" below.
- **Idle timer:** 60 minutes. A JupyterLab terminal running work keeps the space up. Detached background jobs don't.
- **Rules:** match data stays inside AWS. Write only under `adam/`. One GPU job at a time. Tell the admin before any job over an hour. Stop the space when done, since storage bills until deletion. Budget is a fixed monthly amount, shared.
- **GPU:** pick the GPU image on Run. Changing the instance type needs the admin's approval.
- **Environment:** `~/venv-dsdk` was built on the CPU image, with pyarrow 25.0.1, `pureskillgg-dsdk` 4.1.1 and `pureskillgg-csgo-dsdk` 3.5.0. It has to be recreated on the GPU image.
- **Not durable on EBS:** the admin's tome copy (43 GB, re-syncable from `data/tomes`), the holdout's extracted files (72 GB, rebuildable from source zips), and temporary staging. The durable copies are all in S3.

## Connecting over SSH

**Prerequisites**
- The space must be running, meaning its JupyterLab app is InService. If it isn't, the proxy fails with "associated app not in InService state". Start it from the SageMaker console: Studio, then JupyterLab, then Run space.
- An AWS profile named `psgg` on the laptop with credentials that can call `sagemaker:StartSession` on the space. The default profile's keys are rejected (`InvalidAccessKeyId`), so use `psgg`. The SSH proxy already uses it.
- The Session Manager plugin installed at `C:\Program Files\Amazon\SessionManagerPlugin\bin\session-manager-plugin.exe`.
- The proxy script `C:\Users\<user>\.ssh\sagemaker_connect.py`. It calls `StartSession` with boto3 and hands the stream to the plugin, which relays to the space's sshd.
- An entry in `C:\Users\<user>\.ssh\config`:

```
Host adam-gpu
  HostName arn:aws:sagemaker:us-east-1:<account-id>:space/d-<domain-id>/adam-gpu
  ProxyCommand python "C:/Users/<user>/.ssh/sagemaker_connect.py" %h psgg
  ForwardAgent yes
  AddKeysToAgent yes
  User sagemaker-user
  StrictHostKeyChecking accept-new
```

**Connect**
- Interactive: `ssh adam-gpu`
- One command, with the host-key fix built in: `pipeline/sagemaker/connect_adam_gpu.sh 'uptime'`
- Quick check that the space is up: `ssh adam-gpu uptime`

**The host-key problem.** The space gets a new SSH host key on every restart, so ssh fails with "REMOTE HOST IDENTIFICATION HAS CHANGED". Only do the fix after an expected restart:
- Manual: back up `~/.ssh/known_hosts`, then run `ssh-keygen -R "arn:aws:sagemaker:us-east-1:<account-id>:space/d-<domain-id>/adam-gpu"`.
- Scripted: `connect_adam_gpu.sh` does the same thing for this one host and logs the new fingerprint to `~/.ssh/adam-gpu-hostkey.log`.

**If it fails**
- If `StartSession` returns an AccessDenied error, send the admin the exact error text. The admin fixed a `sagemaker:AddTags` permission earlier for this kind of failure.
- For a detached command over SSH, use `ssh -n`, and don't put a nested `ssh` inside an SSH command.
- Long jobs must run in a JupyterLab terminal. Detached background jobs don't keep the space up past the 60-minute idle timer.

## Next steps

1. **Decide:** reuse or regenerate the draws; holdout dedupe; holdout B; frozen-test rule; and whether to export to RunPod or use SageMaker training jobs. The RunPod question needs the admin.
2. **Build the event-only state table** from the chunk tomes (CPU), save it to `adam/`, then train the baseline on GPU.
3. **Build per-tick features** for the 59-input base (CPU, long), save to `adam/`.
4. **Protocol:** patch the loader, rerun the 59-input base on validation, pick the candidate on validation, tell the admin, then score once on Oct 26–31.
5. **Wrap up:** stop the space, and commit code and notes to the repo branch.

## Blockers before training

- Event-only state table not built.
- GPU environment not set up or tested.
- Open decisions above, especially the RunPod question and the draws.
