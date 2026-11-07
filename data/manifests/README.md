# Data manifests

Per-day match counts for every training input, so you can see which days a model used without opening the data. Each CSV has one row per upload day (taken from the match key `csds/YYYY/MM/DD/...`), with a column per split and a total.

| File | What it covers | Split labels come from |
|---|---|---|
| `xgboost_v3_to_v8_and_xgb83_days.csv` | XGBoost v3 to v8 and the 83-input model (`cache_snapshot_2026`) | the manifest those models recorded (`data/benchmark/split_manifest_a17f219b.json`): 10,577 train, 1,168 val, 3,059 test |
| `round_state_days.csv` | sequence models and the all-maps temporal models (round-state arrays) | the `split` array inside the arrays: 19,486 train, 1,169 validation, 3,061 test |
| `cache_snapshot_2026_days.csv`, `cache_snapshot_2026_ext_days.csv`, `cache_pos_2026_trainctr_days.csv`, `allmaps_pos24_trainsites_grid_days.csv`, `matches_days.csv` | the input tables as they are now | the 57,811-match export's split, for reference only; not the split any model trained on |
| `export_clean_v4_days.csv` | the 57,811-match export | the export's own split |
| `summary.json` | first and last day, match counts per split, for each table above | as listed |
| `tome_folders.txt` | the tome folder names on the data root | |
| `round_state_meta/` | build metadata of each round-state array (scaling window, rows and matches per split, days built) | |

Built by `scratch/session_2026-11-06/day_manifests.py` and the a17f219b recount in this commit's history. Only match keys and split labels were read.
