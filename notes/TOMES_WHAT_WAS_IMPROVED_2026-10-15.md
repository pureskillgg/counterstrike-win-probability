# Tomes: what was changed, where it is written down, how to use it (2026-10-15)

## One-paragraph answer
The stock dsdk tome builder (4.1.1) restarts from zero when a build stops and cannot add matches to a finished tome. We forked it so it can **resume** a stopped build and **extend** a finished collection with only the new matches, wrote a driver around it, and added a chunked, append-only layer so a day of new matches costs minutes. Day 13 (2,797 matches) went into the store that way this afternoon: merge 428 s, nothing earlier rebuilt. We did **not** make the stock builder itself faster (same per-match reading and narrowing; the gain is that nothing is read twice).

## Where the information is
| What | Where |
|---|---|
| Full specification of the fork (options, on-disk contract, resume and extend rules, crash-safety table, naming) | `G:\pureskill\dsdk-fork\README_FORK.md` |
| The code | `G:\pureskill\dsdk-fork`, branch `incremental-tomes`, commit `e8d0e24` on top of tag `4.1.1` (7 files, +2,329 / -70): `pureskillgg_dsdk/tome/{builder.py, curator.py, writer_fs.py, incremental.py}` and tests `incremental_test.py`, `writer_fs_test.py` |
| Design, the early assessment, and the "Fork implemented" section with the measured numbers | `notes/TOME_INCREMENTAL_BUILD_DESIGN.md` |
| The project driver (fresh / resume / extend in one call, lock file, half-synced matches skipped, `--days`, `--skip-keys`, `--publish-chunks`) | `scripts/data/build_tomes_incremental.py` |
| Chunk layer and equivalence tests | `scripts/data/append_combined.py`, `scripts/data/test_incremental_chunks.py` (8 of 8), `scripts/data/compare_tome_collections.py`, `scripts/data/test_append_equivalence.py` |
| The release run (day 13) | `scripts/run_release_b.ps1`, `logs/release_b.log` |
| In the deck and on the status page | slide "Adding data without restarting" (id `incremental`); status page, "Tooling" |

## What changed
1. **Atomic writes** (always on): manifests and pages are written to a temp file, flushed and renamed, so a crash never leaves a half-written manifest.
2. **`resume=True`**: keeps the staging folder, rebuilds each channel's ledger from the page keysets, and reads only the matches and channels the finished pages lack.
3. **`behavior_if_complete="extend"`**: reads only matches the header lacks and appends new pages to the header and every channel tome; an old page is rewritten only when the new matches widen a column type; header page 0 is rewritten; nothing new on disk means nothing read or written.
4. **`signatures` manifest field** so narrowing can be re-planned over old and new matches exactly.
5. **Names without dates** (`basic_<channel>`), because `{dates}` changes with every extension.
6. **Project layer**: new pages published as chunk collections `<out>_xNNN` (hard links, `BUILD_OK`, listed in `<out>.CHUNKS_DONE`); `append_combined.py` merges only the new chunks into the combined store, with a ledger and recovery.

## Measured
* Fork test suite: 311 tests pass (14 min 25 s).
* Invariant tested: build(all) equals build(first k, killed) then resume, and equals build(A) then extend(B), key for key, row for row, dtype for dtype; an 86-match fixture (`data/mini`) and a 40-match real-data build.
* Day 13 through the project layer: append 428 s, prepare 40 s, grid 35 s, split 20 s, loadout extraction 626 s, model cache 486 s (12 chunks); 20,754 matches in the store afterwards.

## What is NOT in the tomes, and how that part stays iterative
The tomes hold the 36 **event** channels. The per-tick `player_status`, `player_vector` and `grenade_vector` files are read straight from the raw match folders by the one-second extractors, each incremental per match: `extract_status_1hz.py`, `extract_positions_1hz.py`, `extract_blind_events.py` (a day file gets only the matches it lacks). Features built from those tables are recomputed per day and reused when a day's inputs did not change (`make_pos_features.py`). So:
* new matches: incremental at every layer;
* a new or changed feature definition (cone width, distances, flash logic): recomputed from the stored one-second tables in minutes, no raw read;
* a new raw field (for example z-velocity): one re-read of the raw files (about 50 minutes for all matches);
* a new event channel: build just that channel as its own tome with the fork.
Not incremental yet: the player-tensor builder for the graph experiment (`scripts/model/gnn/make_player_tensors.py`, about 12 minutes for all rows; resumable by day with `--from-day`).

## Open
About 2,800 day-13 folders were still syncing earlier; at the last listing the day-13 folder held 2,797 matches, all merged. New uploads go in through `build_tomes_incremental.py --days ... --skip-keys ... --publish-chunks` then `append_combined.py`. Nothing here has been proposed upstream.
