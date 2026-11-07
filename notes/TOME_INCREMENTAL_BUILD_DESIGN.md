# Incremental and resumable tome builds - findings and a design for an upstream contribution

Written 2026-10-14 from a read of `pureskillgg-dsdk` 4.1.1 (MIT, https://github.com/pureskillgg/dsdk), file `pureskillgg_dsdk/tome/builder.py`
and its neighbours (`manifest.py`, `writer_fs.py`, `constants.py`). Line numbers refer to that version. Nothing has been opened upstream
(no issue, no PR); this is a local analysis.

## What the library does today (and why every restart starts from scratch)

1. **A complete header tome is a frozen set of matches.** `_keep_header` (l.232-241) keeps a complete header only with
   `behavior_if_complete="pass"`; `_load_header` (l.253-267) then takes the match list *from the header*, and if the caller passes a
   different `keys` set it raises `ValueError("keys differ from the matches of the complete header tome ... pass
   behavior_if_complete='overwrite' to rebuild it, or another header_tome_name")` (l.261-265). The only ways to add matches are to
   rebuild everything (`overwrite`) or to build a second tome under another name. The module docstring of `build_basic_tomes.py` says it
   plainly: "Complete tomes are kept, so a rerun reads nothing; an interrupted build starts over."
2. **An interrupted build deletes its own progress.** `_walk` begins with `self._remove_staging()` (l.285): the `.building` folder, which
   already holds every finished page, is `rmtree`'d. The pages are written as the walk goes (`_ArrowScribe._write`, l.510-527: page files,
   then the manifest with `isComplete=False`), so the information needed to resume is on disk; the code discards it.
3. **Names embed the data range.** `basic_{channel}.{dates}` (l.57) takes `{dates}` from the header's `match_date` min/max
   (`header_dates`, l.1071-1077), so adding later matches changes every channel tome's name.
4. **Manifest writes are not atomic.** `TomeWriterFs._write_json` (writer_fs.py l.107-109) opens the final path with `"w"`; a crash in the
   middle leaves a truncated manifest.
5. **The header is one page written only at `finish()`** (`max_page_size_mb=None` for the header scribe, l.356), so a crashed run has no
   header rows on disk.

## What is already in the format that makes both features possible

* A tome is `manifest JSON + independent page files`: `pages: [{number, keyset, dataframe, ...}]`, `dataframe_NNNNN` (parquet) and
  `keyset_NNNNN` (parquet, one column of the match keys whose rows are in that page). Pages are immutable once written and numbered
  with `zfill(5)`; appending a page is just "write `dataframe_N+1`, `keyset_N+1`, add one entry to `pages`".
* The per-page **keyset is exactly the resume ledger**: it says which matches a page already holds.
* `_publish` (l.380-408) already has the right publish order (mark the destination partial first, move pages, write the final manifest),
  and `_narrow_written_pages` (l.529-561) already knows how to rewrite old pages when a later match widens a column type.

## Proposal

### A. `resume` (no format change)
* Do not remove `.building` at the start of `_walk` when `resume=True` (default off at first, to stay backward compatible).
* For each channel scribe, reload the staged manifest and its keysets: `done[channel] = union of page keysets`; continue page numbering.
* Walk all matches; for each one read **only the channels that still lack it** (`_read_match(path, channels_needed, with_header=True)`;
  the header row is cheap and the header page is rebuilt from scratch because it is written only at the end).
* Recover the narrowing signatures of the finished pages from their parquet schema + pandas metadata (or, better, store them in the
  manifest as an additive field `signatures`; old readers ignore unknown manifest keys).
* Invariant to test: `build(all)` equals `build(first k matches, kill) -> resume` row for row and dtype for dtype (page boundaries may differ).

### B. `extend` (append new matches to complete tomes)
* New `behavior_if_complete="extend"`: take the existing header keyset, keep only matches not yet in it (this also gives free
  de-duplication), build new pages for them with page numbers after the current maximum.
* Header: read the existing header page, append the new rows, rewrite page 0 (keeps the "one header page" convention) - or, as a later
  step, allow several header pages.
* Channels: new pages only. Recompute the narrowing plan over *old + new* signatures; rewrite an old page only when the plan changes
  its column type (rare; reuse `_narrow_written_pages`). The `_copy_header` sub-header of each channel tome is regenerated with the
  union of keys.
* Naming: allow `tome_name` without `{dates}` (e.g. `basic_{channel}`), or keep the first range and store the live range in the
  manifest; otherwise every extension would rename the tome.
* Publish order for safety (pages are immutable, so the manifest is the commit point): (1) move the new page files into place (orphans
  until listed), (2) write the header, (3) write each channel manifest with a temp file + `os.replace`. Make `_write_json` atomic for
  all manifests.
* Invariant to test: `build(A + B)` equals `build(A)` then `extend(B)` for rows, dtypes, keysets and header; plus an extend whose new
  matches widen an int8/int16 column, which forces the rewrite path.

### C. Smaller independent fixes worth separate PRs
* Atomic manifest write (`os.replace`).
* A friendlier error/option when `keys` differs from a complete header (point to `extend`).
* Documentation of the page/keyset ledger as the resume contract.

## Effort and risk (honest estimate)
* A (resume): about 150-250 lines plus tests; moderate risk because of the narrowing signatures and the multi-channel lockstep (each
  channel's pages end at different matches, so the safe resume point is per channel, not global).
* B (extend): about 250-400 lines plus tests; the delicate part is the narrowing/type-widening rewrite and keeping the sub-header
  copies consistent. The existing `builder_test.py` (30 KB) and `narrowing_test.py` (36 KB) show the test style to follow.
* Both are additive behind new options; default behaviour stays identical, so existing users are unaffected.
* Suggested path: open an issue first to ask the maintainers whether they want `resume`/`extend` and how they would name the options,
  then implement on a fork. **Opening the issue or the PR is publishing to a public repository and needs the user's go-ahead.**

## Implemented locally so far (2026-10-14, project scripts; no library change yet)
* `scripts/data/append_combined.py`: appends new tome sources to an existing combined store using the ledger in `COMBINED_BUILD.json`.
  Per channel it plans `append` (new pages `dataframe_<next>`, keyset union, same casts as `build_combined.py`), `fresh` (a channel the store
  does not have yet), `skip`, or `rebuild` (only when the new sources change that channel's schema: a new column, or a canonical dtype
  change such as `Null -> Float64`; rebuilt from all sources with `build_combined.build_channel`, swapped in at the commit). Safety: a
  `<out>.bak` copy first, orphan pages of an interrupted run removed, rows/keys/pages verified before the ledger (temp file + `os.replace`)
  is rewritten. `build_combined.py` keeps its full-rebuild behaviour (its per-channel loop is now `build_channel()`; same code).
* Dry run on the real store (3 merged sources) with day 09 and day 12 as new sources: 33 channels append, 1 fresh, 3 rebuild
  (`header.bomb_time` Null -> Float64, `player_footstep.kind` and `player_sound.sound` new columns): so schema drift between batches is real,
  and "append, rebuild only the drifting channels" is the right granularity.
* To do: equivalence test (append result == full rebuild: per-channel rows, key sets, schema, sum of row hashes), then the dsdk-level
  `resume` / `extend` prototype in a copy of the installed package (unittest tests; pytest is not installed).

## Prototype assessment (2026-10-14 17:20, after reading builder.py l.1-830 and builder_test.py l.1-280)
A library-level `extend` is larger than the estimate above suggests; a rigorous version is multi-day work and was NOT started. What reading the code showed:
* `{dates}` is part of every channel tome's name (`basic_{channel}.{dates}`), and the dates come from the header's `match_date` range. Extending with later matches changes the range, so either `extend` requires a fixed `tome_name` without `{dates}` (simplest; a documented restriction) or it must rename the tome folders and rewrite `tome`/`key` in the manifests and page entries.
* The staged scribes are built from scratch (`_ArrowScribe` starts with `_page_counter = 0`, empty `_signatures`, empty manifest). Extending needs a scribe that starts from the existing manifest: page numbers after the last page, the old keysets in `_page_keys`, and the old pages' parquet schemas as signatures (only `signature[0]`, the schema, feeds `plan_narrowing`); `_narrow_written_pages` then already rewrites old pages whose type changes. That part is feasible and small.
* The header tome is ONE page written only at `finish()`; extending it means reading page 0, appending the new rows (with the header's own type reconciliation) and rewriting it, and every channel's header copy (`_copy_header`) must be regenerated from the union of keys after the global header is final.
* Publishing: `_publish` marks the destination partial with no pages, then moves pages over. For `extend` the old pages must stay listed until the new manifest replaces them, so the safe order is: new page files under new numbers, narrowed rewrites via temp file + `os.replace`, then one atomic manifest replace. `TomeWriterFs._write_json` is not atomic today (fix independently).
* Tests: `builder_test.py` has a synthetic collection helper (`write_match`, `deaths(...)` with int/float/object drift, empty channels, stored index) that is enough to test `build(A+B) == build(A) + extend(B)`; it imports pytest at module level, which is not installed here, so the tests would be copied into a unittest module or pytest installed in an isolated env (needs the user's go-ahead).
* Suggested order for a real contribution: (1) atomic manifest write; (2) `resume` (per-channel ledger from page keysets); (3) `extend` with fixed names; (4) `{dates}` handling. Open an upstream issue first with this document; nothing is opened without the user's go-ahead.
The project-level equivalents are implemented and tested instead: `run_tome_queue.ps1` + `tome_ledger.py` (build only the matches not in a finished tome), `append_combined.py` (+ `test_append_equivalence.py`, append == full rebuild page by page).

## What this project does in the meantime (no library change)
* Each day is built as chunks of 500 matches, each chunk its own small tome collection with its own `BUILD_OK`
  (`scripts/run_tome_queue.ps1`); a crash, reboot or dropped drive costs at most one chunk, and a rerun skips every finished chunk.
* `scripts/data/build_combined.py` merges all collections (including chunks, via `tomes/<day>.CHUNKS_DONE`) into one store, so new data
  is "build only the new chunks, then merge". The merge itself still rewrites the combined store each time (about 3 GB, minutes);
  making it append-only (reuse the cast pages of unchanged sources when the canonical schema is unchanged) is the next local step.

## Fork implemented (2026-10-15)
The library-level work the assessment above called multi-day was done as a local fork of dsdk 4.1.1: `G:\pureskill\dsdk-fork`, branch `incremental-tomes`
(tag `v4.1.1` is the base; see `README_FORK.md` there). What exists and what was measured:
* **Fork**: atomic manifest and page writes; `resume=True` (keeps the staging folder, rebuilds the per-channel ledger from page keysets, `signatures` manifest field);
  `behavior_if_complete="extend"` (new pages after the old ones, narrowing re-planned over old plus new signatures, header page 0 rewritten, no `{dates}` in names).
  Its own and the upstream tests pass: 311 tests, 14 min 25 s.
* **Driver**: `scripts/data/build_tomes_incremental.py` (runs with `G:\pureskill\envs\dsdk-fork\Scripts\python.exe`): fresh, resume or extend in one call; lock file; matches whose
  files are only half synced are left for the next run; `--days` scans only the named upload days; `--skip-keys` skips matches already in the store;
  `--publish-chunks` publishes the new pages as a chunk collection `<out>_xNNN` (hard links, `BUILD_OK`, listed in `<out>.CHUNKS_DONE`) so that `append_combined.py` merges only the new data.
  `scripts/data/test_incremental_chunks.py`: 8 of 8 pass; `scripts/data/compare_tome_collections.py` compares two collections (rows, key sets, schemas).
* **Real-data check**: a 40-match batch built in one go gives the same collection as the same batch resumed and then extended (rows, keys, schema); an 86-match fixture is in `data/mini`.
* **Day 13 through the project layer** (`scripts/run_release_b.ps1`, log `logs/release_b.log`): merge of the 2,797 finished day-13 matches 428 s, prepare 40 s, grid 35 s, split extension 20 s,
  loadout extraction 626 s, model cache 486 s (12 chunks; 5 chunks was killed twice by the memory guard because two other jobs held about 16 GB of commit). 20,754 matches in the store afterwards,
  17,563 usable (2024: 2,749; 2026: 14,814). The older matches were not rebuilt.
* **Still open**: about 2,800 day-13 folders were still syncing; they go through `build_tomes_incremental.py --days 2026/10/12,2026/10/13 --skip-keys <combined and chunk keys> --publish-chunks`
  then `append_combined.py --sources 2026-10-13-live` (a day of new matches should again cost minutes, not a rebuild). Not pushed or proposed upstream.
