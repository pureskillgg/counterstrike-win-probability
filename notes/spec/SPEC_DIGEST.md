# CSDS / tome spec digest

Written 2026-10-12 from full reads of: build_basic_tomes.py (114 lines); csds-to-tome.pdf (12 pages); csds-manifest.pdf (6 pages); dsdk docs/tome-data-model.md (326 lines, read in full). Not read by me directly: the dsdk, csgo-dsdk and makenew-pyskill repos and the PureSkill docs pages (Haiku readers covering those are listed at the end once they report). examples/example-1.json and example-2.json, which the manifest PDF cites, are not in G:\ngclab_multi_spectral.

## What the user asked for (status as of this file)
1. Finish October downloads (1-10). 01-08 complete on disk (08 matches the revision stats exactly). 09 in progress (470 of 1,162 at 15:50). 10 not started. The downloader is an external aws process; we did not start it.
2. Move the older CSDS (described as from Google Drive) into the main csds folder. NOT DONE. Blocked on a decision: which 2024-02 copy is canonical, and the merge rule (no overwrite).
3. Set up a main working folder. DONE: G:\pureskill\csds-cs2 (pixi project, tomes, staging, scripts, notes, logs).
4. Build header and basic tomes with dsdk / csgo-dsdk. DONE for 2026 days 01-08 (full build, verified). 2024 waits on the pre-scan. Days 09 and 10 later, into a new root.
5. Profile the data with AI, "consider grill-me". Profiling by Haiku agents done for days 01-07 and one match. grill-me skill NOT found in this account. "lavish" unidentified; asked.
6. Simple non-GPU model for round win probability (xgboost / sklearn). Research done. No model trained yet. Bake-off planned.
7. Record as a new Meridian project. DONE: fa6f6e09-513d-4788-9db5-43123df4ce0f.
8. Checkpoint in Meridian; parallel Haiku subagents; store research in a Zotero folder. Checkpoints and task logs saved. Zotero: NOT possible with connected tools (read-only search only). Fallback: .bib files in notes\research.
9. Keep the tome build open to add 09 and 10. Each snapshot gets its own tome root.
10. Install in this folder with pixi. DONE (.pixi\envs\default inside G:\pureskill\csds-cs2).
11. Statement draft (MAI Lab / PhD) with ADAM TO CONFIRM placeholders: teaching experience, professional goals, personal goals. NOT STARTED. Needs the user's own content.

## Build script facts (build_basic_tomes.py)
- Env vars: PURESKILLGG_TOME_DS_COLLECTION_PATH = folder that holds csds/YYYY/MM/DD/<id>/; PURESKILLGG_TOME_COLLECTION_PATH = output folder, best on a different disk. Optional: PURESKILLGG_TOME_DS_TYPE (default csds), PURESKILLGG_TOME_DEFAULT_HEADER_NAME (default header).
- Flags: --channels a,b,c; --include-per-tick; --max-page-size-mb (default 256).
- Default channels: every channel in the FIRST manifest found, minus header, minus the per-tick set {player_vector, player_status, tick, grenade_vector, world_item_vector}. So player_inputs IS included by default.
- Output: <tome>/tome/csds/header/ and basic_<channel>.<dates>/ (dates = first and last match_date).
- Script has no keys= option. It always walks the whole collection path it is given.

## Tome model facts (dsdk docs/tome-data-model.md)
- Header tome: one row per match. The only tome that searches the disk. Other tomes read its keyset.
- build_basic_tomes: one walk over the matches. Each channel tome is written to a .building staging folder, then renamed to its final name. Each channel tome holds a copy of the header.
- Resume: a complete tome is kept (default behavior_if_complete="pass"), so a rerun reads no match. An interrupted build leaves no partial channel tome; the next call starts over. There is no "continue".
- EMPTY TOMES ARE NOT SUPPORTED ("Empty Tome not supported"). Our basic_player_inputs (0 rows, 0 columns) is therefore an invalid output. Exclude player_inputs from channel lists and remove that folder from the 2026 root.
- Zero rows is different from never looked: a match with no rows still appears in the keyset.
- Pages: zstd by default. Build page size counts Arrow buffer bytes. make_tome counts in-memory pandas size, which runs 2-10x the parquet on disk.
- read_threads default 4. Write to a different disk from the collection (done: G: vs E:).
- Mixed types: int64/float64 vs narrower types are narrowed with checked casts. NarrowingError is raised if a value does not fit. Watch for this on 2024 (old vs compact csds).
- Page dtypes: pyarrow concat with promote_options="permissive"; falls back to pandas with the log line "Page built with pandas".
- Harmless warning seen in pilot: header name does not match the tome_name.start-date,end-date convention.

## Manifest facts (csds-manifest.pdf)
- A CSDS match folder holds the manifest file csds (gzip JSON) plus one parquet per channel, no extension. Path shows the processing day: csds/2026/03/01/<id>/.
- Manifest: 42 channel entries, about 774 column entries, no match rows.
- Match fields: id (folder name; a demo processed twice gets two ids), key (csds/YYYY/MM/DD/<id>/csds; the identity), createdAt (processing time, UTC; the key date), matchDate (played date), platform (steam, faceit, unknown), matchType (e.g. competitive), game (csgo), type (csds), sharecode and demoId (REDACTED in the published data), redacted=true.
- Channel categories: single_event 25, multi_event 7, telemetry 7, player_info 2, header 1.
- "available" is set only on six channels whose events a demo may not carry: bot_takeover, bullet_damage, player_inputs, player_sound, rank_update, round_mvp. available=false means the channel is empty for that reason.
- Column origins: replay (written by the parser), merged (joined from another channel), calculated, calculated-deleted (listed but NOT in the file), replay-redacted, replay-deleted (also not in file), replay-capped, replay_fixed, calculated-redacted. Skip any column whose origin ends in "-deleted".
- Merged player columns: 11 per player involved (*_id_fixed, x/y/z_pos, x/y/z_vel, phi/theta angles, weapon_code, team_code), joined at the nearest tick.

## Tome guide facts (csds-to-tome.pdf)
- Three steps: scan (header: key and match_id), pick (subheader: filter, e.g. one map or Premier only), fill (crawl the subheader's keys; merge channels; tag match_key; copy header in).
- Family tree: header is the root. Subheaders: all.<dates>, has_<channel>, platform_faceit, premier, map_<name>, platform_steam. Data tomes hang off them.
- Page size example: 256 MB gives 9 pages for 6.1M rows; 128 MB about 18 pages; no limit gives 1 page.
- Example numbers in the guide come from a 46,637-match collection read 2026-10-09 (not ours).

## Decisions these facts force on us
- Use an explicit channel list that drops player_inputs (empty). Use the same list for 2024 and 2026 so names match.
- Build each snapshot into its own root. Adding days 09 and 10 means a new root.
- Keep the 2024 pre-scan result as the exclusion list. The lab script has no keys= option, so bad matches must be removed by staging only the good matches.
