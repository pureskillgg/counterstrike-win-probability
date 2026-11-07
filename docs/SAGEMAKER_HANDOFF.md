# SageMaker handoff

Written 2026-11-02 for a separate chat that handles the SageMaker side only. The main chat keeps running the models. Paste this file into the new chat to start it.

Source of the setup: the admin's setup message (domain, profile, space, data layout, rules). Everything below was checked against the account on 2026-11-02 unless marked as not checked.

## Current state

- Domain `pureskillgg-csgo-production-sagemaker` (`d-<domain-id>`), profile `adam`, space `adam-gpu`: InService. JupyterLab app `default`: InService.
- Image `sagemaker-distribution-gpu`, version alias `4.6.1`.
- Instance `ml.g6.xlarge` (1 x NVIDIA L4, 24 GB; 4 vCPUs; 15 GB RAM; root disk 37 GB; `/home/sagemaker-user` is a 200 GB EBS volume). The admin's message says `ml.g6.2xlarge` (8 vCPUs, 32 GB RAM). The two don't match; see decisions.
- SSH: `ssh adam-gpu` works through the SageMaker proxy. The host key was re-accepted on 2026-11-02, and the old entry is kept in `~/.ssh/known_hosts.old`.
- Python 3.12.15 on the space. `pureskillgg-dsdk` and `pureskillgg-csgo-dsdk` are NOT installed. The `PURESKILLGG_*` environment variables are NOT set. `~/data` does not exist yet.
- Read access from the space, verified: `head-object` and a ranged GET on `csds/2026/10/14/m_partial.zip` in `pureskillgg-csgo-production-dataexchange-csds-final-0`.
- The space's role can list buckets; a listing from this PC's IAM user is denied.

## What is in S3

- Match archive: `s3://pureskillgg-csgo-production-dataexchange-csds-final-0/csds/YYYY/MM/DD/<match>.zip`. One stored zip per match, about 9 to 20 MB. October 14 has 950 zips.
- Working bucket: `s3://pureskillgg-csgo-production-sagemaker-0/`.
  - `data/csds/`: unzipped matches as `data/csds/YYYY/MM/DD/<match>/<channel>`. Checked: `csds/2026/` has months 01 to 07 and 10. Months 08 and 09 are NOT present.
  - `data/tomes/`: present (`tome/` inside it). The admin says 42 GB.
  - `data/win_prob_rows/` and `data/win_prob_train/`: NOT present yet.

## Rules (from the admin)

- GPU bills $1.22 an hour while the space runs. Stop the space when done. It also stops itself after 60 minutes idle.
- One GPU job at a time. Ask before any job longer than an hour. Don't change the instance type without asking.
- Match data stays inside AWS. Nothing downloaded to a laptop, nothing pasted into outside tools, nothing shared. Write only under `adam/` in the working bucket.

## Decisions needed

1. Instance: keep `ml.g6.xlarge`, or switch to `ml.g6.2xlarge` as the admin's spec says.
2. Local copies on the laptop: the raw archive on E: and derived arrays on G: (including about 26 GB of round-state data) are match-derived data. Keep them, or delete them once the space has what it needs.
3. Months 08 and 09 of 2026 are missing from the working bucket's `csds/`. Ask the admin whether they are coming.
4. Stop the space now? Nothing is running on it.

## Reading the data

Install the kit (admin step 5):

```bash
pip install "pureskillgg-dsdk[polars]" pureskillgg-csgo-dsdk
```

Environment, in `~/.bashrc`:

```bash
export PURESKILLGG_TOME_DS_COLLECTION_PATH=/home/sagemaker-user/data
export PURESKILLGG_TOME_COLLECTION_PATH=/home/sagemaker-user/tomes
export PURESKILLGG_TOME_DS_TYPE=csds
export PURESKILLGG_TOME_DEFAULT_HEADER_NAME=header
```

Two ways to read:
- Stream one match from the working bucket with `DsReaderS3` and `GameDsLoader` (the admin's example), without using volume space.
- Sync one day at a time to the volume, then unpack. About 18 GB per 1,000 matches, so the 200 GB volume holds about ten days.

```bash
aws s3 sync s3://pureskillgg-csgo-production-dataexchange-csds-final-0/csds/2026/10/25 ~/data/csds/2026/10/25
python -c "from pureskillgg_dsdk.adx.unzip import extract_adx_zips; print(extract_adx_zips('/home/sagemaker-user/data', delete_zips=True))"
```

Tomes (42 GB) fit on the volume: `aws s3 sync s3://pureskillgg-csgo-production-sagemaker-0/data/tomes ~/tomes`.

All 67,224 matches cannot fit on the volume at once.

## Notes for the main chat

- Our extraction scripts read the layout `<root>/csds/YYYY/MM/DD/<match>/<channel>`. That appears to match the unzipped layout above. Verify before pointing a script at `~/data`.
- Stopping or restarting the space keeps `/home/sagemaker-user`. The root disk and `/mnt/sagemaker-nvme` are wiped.
