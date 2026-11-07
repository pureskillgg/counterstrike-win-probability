# pureskillgg/dsdk and pureskillgg/makenew-pyskill: digest for basic tomes

Prepared 2026-10-12. Source: default-branch contents read with `gh api` (raw). Read-only: nothing under E:\ or G:\pureskill\csds-cs2\tomes was touched. dsdk pyproject version at read time: 4.1.1.

Legend: FULL = every line read. PARTIAL = only the stated ranges or grep hits. INFERRED = deduced from code or docstrings, not executed. "Verified" = checked against a real file in the repo.

## Summary: five facts for building basic tomes

1. Use `TomeCuratorFs(...).build_basic_tomes(channels)`. One walk over the matches. Output goes to `<tome_root>/tome/<ds_type>/`. Tome names are `basic_{channel}.{dates}`, where dates is `yyyy-mm-dd,yyyy-mm-dd`.
2. Defaults: `max_page_size_mb=256`, `read_threads=4`, `compression="zstd"` (docstring), `behavior_if_complete="pass"`, `behavior_if_partial="overwrite"`.
3. Complete tomes are kept and partial tomes are rebuilt. Interrupted builds leave no partial channel tome.
4. The curator has no library defaults. Four `PURESKILLGG_TOME_*` env vars (or explicit args) are required, or construction raises.
5. Every match must have a `header` channel. Each channel tome is keyed by the manifest's own `key` field, and duplicate keys are dropped (first wins).

---

## 0. Read ledger

| Repo | Path | Bytes | Lines | Status |
|---|---|---|---|---|
| dsdk | README.rst | 4600 | 149 | FULL |
| dsdk | docs/sqs-consumer.md | 2446 | 56 | FULL |
| dsdk | pureskillgg_dsdk/tome/builder.py | 39167 | 1099 | FULL. Module docstring at lines 1-9 |
| dsdk | pureskillgg_dsdk/tome/curator.py | 20492 | 590 | FULL. No module docstring (class docstring only) |
| dsdk | pureskillgg_dsdk/tome/header_tome.py | 4678 | 155 | FULL. No module docstring |
| dsdk | pureskillgg_dsdk/tome/maker.py | 5408 | 154 | FULL. No module docstring |
| dsdk | pureskillgg_dsdk/tome/manifest.py | 3981 | 140 | FULL. No module docstring |
| dsdk | pureskillgg_dsdk/tome/__init__.py | 114 | 3 | FULL (extra: export check) |
| dsdk | pureskillgg_dsdk/tome/factory.py | 107 | 5 | FULL (extra: export check) |
| dsdk | pyproject.toml | 1412 | n/a | PARTIAL: name, version, requires-python (extra) |
| dsdk | CHANGELOG.md | n/a | 152 | PARTIAL: headings, lines 20-50, keyword grep (extra) |
| makenew | README.rst | 9211 | 357 | FULL |
| makenew | sample_data/README.md | 1533 | 27 | FULL (a second README in the repo) |
| makenew | sample_data/csds/2026/10/09/e1013060-d724-4664-9bf7-540fad7fcf6e/csds | 5798 gzip (119098 decompressed) | n/a | Parsed as JSON with node; fields listed (extra) |
| lab | G:\ngclab_multi_spectral\build_basic_tomes.py | 4783 | 114 (CRLF) | FULL, comparison only |

- makenew-pyskill has no `docs/` directory (API returned 404), so "any docs/ files" yields none.
- dsdk `docs/` holds `sqs-consumer.md` and `tome-data-model.md`. The second is excluded as instructed.

---

## 1. dsdk README.rst (FULL)

What it covers: install, usage, module map, docs pointers, development, publishing.

- Scope statement: "It is a library only: every bucket, queue and dataset is an argument you pass in."
- Python: "Requires Python 3.11 or later." Matches pyproject `requires-python = ">=3.11"`.
- Extras: `polars` extra needed for `get_dataframe(library="polars")` and `scan()`. `xgboost` extra for `s3_xgboost` (not tome-related).
- Reading: `DsReaderFs(root_path=..., manifest_key=...)`, `GameDsLoader(reader=...)`, `get_channel({"channel": ..., "columns": [...]})`, `get_channels([...])`. "Leave out `columns` to read every column."
- Module map: `tome` exports `TomeCuratorFs` and `create_tome_curator`. Verified in `tome/__init__.py` and `tome/factory.py`. The tome row says "used by: the makenew-pyskill template's notebooks".
- Development: Git LFS needed (fixtures). Tasks: `make lint`, `make test`, `make watch`, `make format`.
- Publishing secrets (CI only): `PYPI_API_TOKEN`, `GH_USER`, `GH_TOKEN`, `GIT_USER_NAME`, `GIT_USER_EMAIL`, `GPG_PRIVATE_KEY`, `GPG_PASSPHRASE`.
- Env vars: none listed.
- Not verified: the SqsConsumer example against `sqs/consumer.py` (not read).

## 2. dsdk docs/sqs-consumer.md (FULL)

Not on the tome path. Recorded because it is a promised contract for the queue workers.

- `SqsConsumer` is a long-poll, bounded-concurrency consumer. Lifecycle methods: `consume()`, `start()`, `run()`, `stop()`.
- `queue` kwarg: a name is resolved with `get_queue_url`. A value containing `://` is used as a URL. Optional `region_name`. The library creates no queues.
- Handler: `await handler(content, metadata)`.
  - Truthy return: message acked and deleted.
  - Raising `DeleteMessage`: force delete.
  - Falsy return or any exception: message stays. It reappears after the visibility timeout and then follows the queue's redrive policy, which lives in the consuming service's infra.
- `SqsJsonMessageTranslator` parses `Body` into `content` and puts the rest in `metadata`. A translate failure or empty content is routed like a handler exception.
- `CancelledError` propagates. Ordinary `Exception` goes to `error_handler`.
- Logs: "SQS Consumer: Message error" and "Receive error", via `log.exception` (structlog).
- Shutdown: worker tasks are cancelled and awaited before the aiobotocore client closes.
- No Sentry and no `LOG_LEVEL` handling. The host owns logging.
- Doc says it was added in dsdk 3.0 and replaces `loafer`.

## 3. dsdk pureskillgg_dsdk/tome/builder.py (FULL)

Module docstring (lines 1-9), quoted: "Build a header tome and one tome per channel in one walk over the matches." It notes that `make_tome` re-reads every manifest per channel, while `build_basic_tomes_from_fs` "visits each match once ... with pyarrow, and pages each channel's rows into its own tome."

### Public function

`build_basic_tomes_from_fs(channels, /, *, tome_name=DEFAULT_TOME_NAME, header_tome_name="header", keys=None, ds_type="csds", tome_collection_root_path="tomes", ds_collection_root_path="data", max_page_size_mb=256, compression=DEFAULT_PAGE_COMPRESSION, read_threads=4, behavior_if_complete="pass", behavior_if_partial="overwrite", log=None) -> BasicTomes`

- Its docstring says it "takes the curator's paths and ``ds_type`` as keywords." The defaults `"tomes"` and `"data"` are a trap if called directly (INFERRED). Use the curator.

### Constants (values as written)

- `DEFAULT_TOME_NAME = "basic_{channel}.{dates}"`
- `DEFAULT_MAX_PAGE_SIZE_MB = 256`, `DEFAULT_READ_THREADS = 4`
- `HEADER_KEY_COLUMN = "key"`
- `BEHAVIORS_IF_COMPLETE = ("pass", "overwrite", "fail")`
- `BEHAVIORS_IF_PARTIAL = ("overwrite", "pass", "fail")`
- `STAGING_FOLDER = ".building"`
- `_PARQUET = "application/x-parquet"`, `_COMPACT_EVERY = 256`, `_STATUS_EVERY = 1000`

### Return value

`BasicTomes` (frozen dataclass):
- `header`: TomeLoader, one row per match.
- `tomes`: dict of channel name to TomeLoader. A channel no match has gets no entry.
- `dates`: `"yyyy-mm-dd,yyyy-mm-dd"` from header `match_date` (first and last day), or None.

### Validation (ValueError unless noted)

- `channels` that is a str or dict, or empty: rejected.
- `tome_name` must contain `{channel}`. Only `{channel}` and `{dates}` are allowed.
- `{dates}` used but the header has no `match_date`: rejected.
- Tome name equal to the header name: rejected.
- Behavior value not in the allowed set: rejected.
- `read_threads < 0`: rejected. `max_page_size_mb <= 0` (None allowed): rejected.

### Which tomes get built (`_Build.run`, `_keep_header`, `_build_tome`)

- Header is kept only when complete AND `behavior_if_complete == "pass"`. Complete with "overwrite" rebuilds (rescans the collection). Complete with "fail" raises `Tome already exists <header>`.
- Partial header: "A partial header is built again unless "fail"."
- Channel tome, missing: built. Complete: per `behavior_if_complete`. Partial: per `behavior_if_partial`.
- If the header is rebuilt, a "fail" for an existing channel tome is raised after the walk (docstring Notes).
- With the header kept, its keyset is the match list. If `keys=` is given and differs, ValueError: "pass behavior_if_complete='overwrite' to rebuild it, or another header_tome_name."

### The walk (`_walk`, `_read_match`)

- `_walk` first removes staging, so an interrupted build starts over.
- Each match: `DsReaderFs(root_path=ds_root, manifest_key=path).read_manifest()`. Uses `manifest["channels"]` entries with `channel`, `contentType`, `key`.
- Header build: match key = `manifest.get("key")`, falling back to the path with `os.sep` replaced by `/`. The code comment says "The header's key is the one the manifest declares, as in create_header_tome". The header row gets `key` and `match_id` (`manifest["id"]`).
- Missing `header` channel while building the header: `Exception("Channel header not found in replay.")`.
- Channel rows: a channel absent from a match is skipped for that match. A `contentType` other than `application/x-parquet`: `Exception("Unsupported content type ...")`.
- Channel rows get a `match_key` column (docstring; `MATCH_KEY_COLUMN` is in constants.py, not read).
- Parquet path: `os.path.join(ds_root, os.path.normpath(entry["key"]))`.
- Duplicates: first occurrence kept. Logs "Duplicate match keys dropped" with count. Docstring: "A key listed twice, or declared by two manifests, is built once".
- Order: header order. `read_threads` read-ahead via `_ordered_map`. 0 or 1 reads sequentially.
- `_header_paths`: with no `keys=`, uses the glob in header_tome.py. ValueError "No matches to build a header from in <root>" if empty.

### Staging and publish (`_publish`, `_remove_staging`)

- Staging: `<tome_root>/tome/<ds_type>/.building/<header_name>/<channel>` (INFERRED from `_staging_name` and `_staging_path`; docstring gives `tome/<ds_type>/.building`).
- Docstring: "Tomes are written to a staging folder ... and moved to their names at the end, once the header's dates are known."
- `_remove_staging` deletes `.building/<header_name>`, then `.building` if empty.
- `_publish`: a channel with zero keys is skipped (info log "Channel in no matches: no tome"). Otherwise:
  1. Write destination manifest with `isComplete=False` and `pages=[]`.
  2. `os.replace` each page's dataframe and keyset file to its final key.
  3. Write the copied header.
  4. Write the final manifest.
- Quote from the code comment: "an interrupted publish leaves a partial tome, which the next build builds again, never a complete tome whose pages are half old and half new."
- Copied header per channel tome: `create_subheader_tome_from_fs(name, src_tome_name=<header>, selector=df["key"].isin(keys) & ~df["key"].duplicated(), is_copied_header=True, preserve_src_id=True, ds_type=..., compression=...)`.
- INFERRED: two concurrent builds with the same tome root and header name share staging and can delete each other's files.

### Names and keys

- Channel tome name: `tome_name.format(channel=c, dates=dates)`. The dates string contains a comma (legal on Windows; awkward in shells).
- Dates: `header["match_date"]`, `str[:10]`, then string min and max. Relies on ISO-like dates (INFERRED). The sample README says `match_date` is cut to the minute, e.g. `2026-10-09T23:00...`.
- Final tome manifest key: `tome/<ds_type>/<name>/tome`.
- Header key: the manifest's own `key` field (e.g. `csds/2026/10/09/<id>/csds`). Verified in the sample manifest.
- Keys from `keys=`: backslashes replaced with `/` by `normalize_key`.

### Dtypes (docstring, code, CHANGELOG 4.0.0)

- Quote: "a column is ``Int64`` when any match's file declares it so."
- `realize` concatenates with `promote_options="permissive"`, then checks that pandas dtypes match what `pd.concat` would give. On mismatch the page falls back to pandas (`pandas_page`), logged "Page built with pandas".
- Dictionary (category) columns are decoded to values (e.g. compact `place_name`) before joining.
- pandas index columns are dropped.
- Mixed wide and narrow int types: narrowed to the tome's type. Values that do not fit raise `NarrowingError`. Pages are rewritten at the end (`_narrow_written_pages`). Narrowing internals are in narrowing.py (not read).
- Flags 0/1 are narrowed to bool; a missing value stays missing (CHANGELOG 4.0.0).
- Header tome is one page (`max_page_size_mb=None` for the header).

### Memory and pages

- Page is cut when Arrow buffers exceed `max_page_size_mb`, checked after every match.
- Quote: "Every channel fills a page at once, so memory peaks near this times the number of channels."
- Page file keys: `dataframe_00000`, `keyset_00000` (INFERRED from manifest.py).

### Env vars, logs, warnings

- No env vars read in builder.py.
- Logs: "Build Basic Tomes Update" every 1000 matches; "Build Basic Tomes Done" (`matches`, `tomes_written`, `pages_built_with_pandas`); "Page narrowed"; "Page built with pandas".
- Quote on disk: "Reading from a hard disk, write the tomes to a different disk."
- `Exception("Couldn't read in parquet file <channel>")` on a pyarrow ValueError.
- `Exception("Empty Tome not supported")` is guarded by `_publish` (INFERRED: unreachable from the build path).

## 4. dsdk pureskillgg_dsdk/tome/curator.py (FULL; no module docstring)

Class `TomeCuratorFs` docstring: "Simple API to manage tomes." Parameter docs list env fallbacks for each argument.

### Constructor

`TomeCuratorFs(*, default_header_name=None, ds_type=None, tome_collection_root_path=None, ds_collection_root_path=None, log=None)`

- Each None falls back to an env var via `get_env_option`, resolved eagerly in `__init__`.
- Env names (verified against the lab script):
  - `PURESKILLGG_TOME_DEFAULT_HEADER_NAME`
  - `PURESKILLGG_TOME_DS_TYPE`
  - `PURESKILLGG_TOME_COLLECTION_PATH` (tome root)
  - `PURESKILLGG_TOME_DS_COLLECTION_PATH` (CSDS root)
- Missing value raises `Exception("Missing option {name}, pass in or set {KEY} in environment")`.
- The library has no defaults of its own for these. "csds" and "header" are not defaults here.
- `get_env_option` checks only for None, so an env var set to "" returns "".

### Methods

- `create_header_tome(self, tome_name=None, /, *, path_depth=None, compression="zstd") -> TomeLoader`. `path_depth` warns: "the keyword path_depth is deprecated." Docstring: "Search is recursive."
- `create_subheader_tome(self, tome_name, selector=lambda df: [True]*len(df), /, *, src_tome_name=None, compression="zstd") -> TomeLoader`.
  - BUG (verified by reading, not run): does not pass `ds_type` to `create_subheader_tome_from_fs`. That function defaults to `"csds"`, so the source header is read from `tome/csds/...` even when `self._ds_type` differs. No effect when `ds_type` is `csds`, the only type in the repo.
- `build_basic_tomes(self, channels, /, *, tome_name="basic_{channel}.{dates}", header_tome_name=None, keys=None, max_page_size_mb=256, compression="zstd", read_threads=4, behavior_if_complete="pass", behavior_if_partial="overwrite") -> BasicTomes`. Forwards to `build_basic_tomes_from_fs` with the curator's paths and `ds_type`.
- `get_dataframe(self, tome_name, *, columns=None, library="pandas", widen=False)`. `library` is "pandas" or "polars" (polars needs the extra). `columns` is read "in this order". `widen`: "Every integer column as a 64-bit integer and every float column as float64; bool columns stay bool."
- `scan(self, tome_name, *, widen=False) -> pl.LazyFrame` (polars extra).
- `get_keyset(self, tome_name) -> list`, `get_manifest(self, tome_name) -> dict`, `iterate_pages(self, tome_name)`.
- `get_loader(self, tome_name) -> TomeLoader`: key `tome/<ds_type>/<tome_name>/tome`.
- `get_header_loader(self)`: uses default header name.
- `get_random_match(self, subheader_name=None, /) -> GameDsLoader`: `random.choice` over the keyset. Raises IndexError on an empty tome (INFERRED).
- `get_match_by_index(self, index=0, subheader_name=None, /) -> GameDsLoader`: reads `df_header["key"][index]` and opens it with `DsReaderFs(root_path=ds_collection_root_path, ...)`. Relies on a RangeIndex (docstring: "A pandas frame has one ``RangeIndex``").
- `make_tome(self, tome_name, /, *, header_tome_name=None, ds_reading_instructions=None, max_page_size_mb=None, max_page_row_count=None, limit_check_frequency=100, compression="zstd", **kwargs) -> TomeMaker`.
  - `max_page_size_mb` default None. Docstring: "By default the scribe will not check." So `make_tome` has no page cap. `build_basic_tomes` caps at 256.
  - `compression`: "gzip" writes "the format dsdk 3.2.2 and earlier used", `None` writes uncompressed.
  - Passing `**kwargs` through to `TomeMaker`.
- Module function `get_env_option(name, value)`.

### Warnings and naming

- Tome names are checked by `warn_if_invalid_tome_name` (constants.py, not read).
- No existence checks in the curator's header and subheader methods (INFERRED: they call functions that always write).

## 5. dsdk pureskillgg_dsdk/tome/header_tome.py (FULL; no module docstring)

- `default_tome_name() -> "header"`.
- `create_header_tome_from_fs(tome_name=None, /, *, ds_type="csds", tome_collection_root_path="tomes", ds_collection_root_path="data", path_depth=None, update_frequency=0, compression=DEFAULT_PAGE_COMPRESSION, log=None) -> TomeLoader`.
  - `path_depth` is accepted and unused (pylint disabled).
  - Scans every manifest from the glob. Per match: `get_channel({"channel": "header"})`, then `df["key"] = manifest["key"]` and `df["match_id"] = manifest["id"]`, then `scribe.concat(df, manifest["key"])`.
  - Logs "Create Header Update" every `update_frequency` matches when non-zero.
  - Writes `tome/<ds_type>/<name>/tome`.
  - INFERRED: no existence check here, so re-running overwrites the header. `behavior_if_*` options exist only in `build_basic_tomes`.
  - INFERRED: `manifest["key"]` has no fallback, unlike the builder's `manifest.get("key") or path`.
- `create_subheader_tome_from_fs(name, /, src_tome_name=None, selector=lambda df: [True]*len(df), *, tome_collection_root_path="tomes", ds_type="csds", is_copied_header=False, preserve_src_id=False, compression=DEFAULT_PAGE_COMPRESSION, log=None) -> TomeLoader`.
  - Selector is applied as `df.loc[selector]`. The new rows are keyed by `list(df["key"])`.
  - New manifest `dsType` is copied from the source manifest. The source read uses the `ds_type` argument (see the curator bug above).
- `get_manifest_key_paths_from_glob(ds_root_path, ds_type) -> list`.
  - Pattern `<root>/*/**/<ds_type>` with `recursive=True`. Directories excluded. Returns sorted, deduplicated relative keys.
  - Uses `os.path.normpath`, so on Windows the keys contain backslashes (INFERRED). The builder handles this by taking the manifest's `key` for header rows.
  - INFERRED: a manifest directly under the root, with no folder above it, is not matched by `*/**`.
- `fetch_ds_loader_from_fs(root_path, manifest_key, log) -> GameDsLoader` (DsReaderFs).

## 6. dsdk pureskillgg_dsdk/tome/maker.py (FULL; no module docstring)

`TomeMaker(*, header_loader, scribe, ds_reading_instructions, ds_type, tome_loader, header_copier, ds_collection_root_path, behavior_if_complete="pass", behavior_if_partial="continue", print_status_frequency=100, reader_class=DsReaderFs, log=None)`.

- `iterate()` is a generator yielding `(data, key)` per remaining key. The caller calls `concat(df)`, which appends to the current key's tome. Returns immediately when nothing is left to do.
- `_load()` reads the full header dataframe first, then picks an action:
  - Tome missing: start new (copies header, full keyset).
  - Complete tome: "pass" means no keys. "overwrite" means rebuild all and copy the header. "continue" is treated as pass. "fail" raises `Tome already exists <name>`.
  - Partial tome: "continue" (the default) resumes. Keys are the header keyset minus keys already in the tome, and no header copy is made. "overwrite" rebuilds. "pass" means no keys. "fail" raises.
- Divergence: `make_tome` defaults partial tomes to resume ("continue"). `build_basic_tomes` overwrites them.
- Per-key read: `reader_class(root_path=ds_collection_root_path, manifest_key=key, log=...)` then `GameDsLoader.get_channels(ds_reading_instructions)`. `None` reads everything per curator docstring (INFERRED for `get_channels(None)`).
- Progress log "Tome Maker Update:" every `print_status_frequency` keys.

## 7. dsdk pureskillgg_dsdk/tome/manifest.py (FULL; no module docstring)

`TomeManifest(*, tome_name, ds_type, is_header=False, header_tome_name=None, src_id=None, is_copied_header=False, log=None)`.

- Manifest fields: `id` (new uuid4 each time), `key` (`make_key(["tome", ds_type, tome_name, "header" if copied else "", "tome"])`), `tome`, `sourceId` (from `src_id`), `type` "tome", `isHeader`, `headerTomeName`, `dsType`, `createdAt` (UTC ISO), `isComplete` False, `pages` [].
- `preserve_src_id` writes `sourceId`, not `id`.
- `end_page(n)` returns `{"number", "keyset": {"keyset": "keyset_00000", "key", "contentType": "application/x-parquet"}, "dataframe": {"dataframe": "dataframe_00000", "key", "contentType"}, "createdAt", "timings"}`. Keys include a `header` path segment for copied headers.
- `finish()`: `isComplete=True`, sums timings across pages (seconds, minutes, hours, days), sets `completedAt`.
- INFERRED: `start_page()` must precede `end_page()`, which uses the start time.

## 8. dsdk pureskillgg_dsdk/tome/__init__.py and factory.py (FULL; extra)

- `__init__.py` exports `TomeCuratorFs`, `create_tome_curator`, `NarrowingError`.
- `create_tome_curator(**kwargs)` returns `TomeCuratorFs(**kwargs)`.
- The lab's `from pureskillgg_dsdk.tome import TomeCuratorFs` resolves.

## 9. dsdk pyproject.toml and CHANGELOG.md (PARTIAL; extra)

- pyproject: `name = "pureskillgg-dsdk"`, `version = "4.1.1"`, `requires-python = ">=3.11"`.
- CHANGELOG headings (in order): Unreleased (line 8), 4.0.2 (line 22, CI and pin changes only), 4.0.0 (line 31), 3.3.1 (line 44, adds `build_basic_tomes`), 3.3.0, 3.2.2, 3.2.1, 3.2.0, 3.1.0, 3.0.1, 3.0.0, 2.0.0.
- There is no 4.1.x section even though pyproject says 4.1.1. Lines 8-19 of Unreleased were not read.
- 4.0.0 adds `NarrowingError`, the `widen` argument, mixed-type narrowing, flag narrowing to bool, and dictionary decoding in `build_basic_tomes`.

## 10. makenew-pyskill README.rst (FULL)

- Template and tutorial for a PySkill. Tutorial "runs on four Counter-Strike 2 matches that come with this repo, in `sample_data`."
- Requirements: Python v3.14 (setup shows 3.14.6 and uv 0.11.21). Stricter than dsdk's 3.11.
- Setup: `uv sync`, then `uv run jupyter notebook notebooks`. Tutorial notebook: `notebooks/tutorial/1 - Setup.ipynb`.
- Dev: `make lint`, `make test`, `make watch`.
- Secrets: `PYPI_API_TOKEN`. Optional: `GH_USER`, `GH_TOKEN`, `GIT_USER_NAME`, `GIT_USER_EMAIL`, `GPG_PRIVATE_KEY`, `GPG_PASSPHRASE`.
- `makenew.sh` (not read) strips the boilerplate and deletes itself.
- External docs (docs.pureskill.gg/datascience) not fetched.
- No env vars for tome building.

## 11. makenew-pyskill sample_data/README.md (FULL)

- Four CS2 matches from 2026-10-09, from the AWS Data Exchange "PureSkill.gg Competitive CS2 Gameplay" data set.
- Matches: `6fddf4fa-...` de_anubis (FACEIT); `83d933c3-...` de_ancient (Valve, Premier); `e1013060-...` de_dust2 (FACEIT); `f9b6c00e-...` de_mirage (Valve, Premier).
- "Each match is 43 objects: 42 channels plus the `csds` index object that lists them."
- "Keep the `csds/yyyy/mm/dd/<match>/` folder layout: each match's index names its files by that path."
- "For these copies, the header channel's `match_date` was also cut to the minute, to match the index object's `matchDate`."
- License CC BY-NC-SA 4.0: no commercial use, share-alike. "Anything you publish from it must carry the text "Data provided by PureSkill.gg.""

## 12. Sample CSDS manifest (extra; parsed with node)

Path: `sample_data/csds/2026/10/09/e1013060-d724-4664-9bf7-540fad7fcf6e/csds`. Gzip (magic `1f 8b`). Decompresses to 119,098 bytes. Only this one of the four manifests was checked.

- Top-level keys: id, type, jobId, game, platform, demoId, sharecode, matchType, matchDate, createdAt, metadata, context, timings, channels, redacted, key.
- Top-level `key`: `csds/2026/10/09/e1013060-d724-4664-9bf7-540fad7fcf6e/csds`.
- `channels`: 42 entries, all `application/x-parquet`. Includes `header` and the five per-tick channels. Entry keys: channel, columns, contentType, key, category, events, redacted.
- Default lab selection for this match: 42 minus header minus 5 per-tick = 36 channels.
- `metadata.channel` is "replay". That is metadata, not a channel entry.

## 13. Lab script comparison: G:\ngclab_multi_spectral\build_basic_tomes.py

What the script does: PEP 723 header `pureskillgg-dsdk>=4.0.2`. Requires `PURESKILLGG_TOME_DS_COLLECTION_PATH` and `PURESKILLGG_TOME_COLLECTION_PATH`. Optional `PURESKILLGG_TOME_DS_TYPE` and `PURESKILLGG_TOME_DEFAULT_HEADER_NAME`. Args: `--channels`, `--include-per-tick`, `--max-page-size-mb` (default 256). Calls `TomeCuratorFs(...)` then `build_basic_tomes(channels, max_page_size_mb=...)`.

### Checked and consistent

- Env var names match `get_env_option` exactly.
- Constructor keyword names and the `build_basic_tomes` call shape match.
- `result.dates` exists on `BasicTomes`.
- Output listing at `<tome_root>/tome/<ds_type>/` matches the staging and loader layout. `.building` is filtered by the script.
- `PER_TICK` set matches the docstring list.
- Manifest handling (gzip or plain JSON; `channels` and `channel`) matches the real sample manifest and the builder's `manifest["channels"]` and `entry["channel"]`.
- The `>=4.0.2` pin is satisfiable: the API exists since 3.3.1, narrowing and dictionary decoding in 4.0.0, and 4.0.2 changes only CI and pins. Current version is 4.1.1.

### Divergences and contradictions

1. Doc pointer is wrong. The docstring says `csds-to-tome/csds-to-tome.md` is "beside this file." Neither the folder nor the `.md` exists. The only file is `G:\ngclab_multi_spectral\csds-to-tome.pdf`.
2. Defaults claim. The docstring says `PURESKILLGG_TOME_DS_TYPE` and `PURESKILLGG_TOME_DEFAULT_HEADER_NAME` "default to csds and header." The library does not. `TomeCuratorFs` raises if neither the argument nor the env var is set. The lab works only because it passes both explicitly.
3. Channel list comes from the first match only, as the script's own docstring says. The builder builds exactly the list it receives. A channel present in later matches but absent from the first is silently not built (INFERRED impact).
4. Discovery mismatch. The lab's `first_manifest` uses `os.walk` (any depth, including directly under the root). The dsdk header glob `<root>/*/**/csds` needs at least one folder above the manifest. A flat layout passes the lab's channel discovery and then fails with "No matches to build a header from in <root>" (INFERRED from glob semantics, not executed).
5. Memory. Default 256 MB pages with the builder's rule "memory peaks near this times the number of channels": 36 channels (sample default) at 256 MB is roughly 9 GB of Arrow buffers at peak (INFERRED from docstring). The lab has no memory warning.
6. Rerun claim. "Complete tomes are kept, so a rerun reads nothing" holds only when the header and every requested channel tome are complete. Any missing channel tome makes the walk read every match (INFERRED from `run` and `_walk`).
7. Partial-tome default differs by API. `make_tome` resumes partial tomes and has no page cap. `build_basic_tomes` overwrites partial tomes and caps pages at 256 MB. Matters only to anyone mixing the two.
8. Concurrent runs with the same tome root and header name share `.building` and can delete each other's staging (INFERRED).

No signature mismatch with the lab's calls was found.

---

## 14. Not read

- `docs/tome-data-model.md` (18880 bytes): excluded by instruction (handled elsewhere).
- makenew-pyskill `docs/`: does not exist (API 404).
- dsdk tome modules outside scope: `constants.py` (`make_key`, `MATCH_KEY_COLUMN`, `DEFAULT_PAGE_COMPRESSION` = "zstd" per docstring only, `warn_if_invalid_tome_name`, `filter_ds_reader_logs`), `narrowing.py`, `page_frames.py`, `scribe.py`, `loader.py`, `reader_fs.py`, `writer_fs.py`, `header_copier_fs.py`. Exact values of constants and tome-name warning text are taken from docstrings only.
- dsdk outside scope: `ds_io/` (including `DsReaderFs.read_manifest` and gzip handling), `sqs/`, `adx/`, `ds_models/`, `conftest.py`, all `*_test.py`, `pylintrc`, `Makefile`, `.github/`, `uv.lock`.
- dsdk CHANGELOG: only headings, lines 20-50 and a keyword grep. Unreleased lines 8-19 not read.
- dsdk pyproject: three lines only.
- SqsConsumer example in README: not checked against code.
- makenew-pyskill: `pureskillgg_makenew_pyskill/`, `notebooks/`, `tests/`, `makenew.sh`, `pyproject.toml`, `uv.lock`, CHANGELOG, `.github/`.
- makenew sample data: three of four manifests and all channel parquet files not read.
- External docs (docs.pureskill.gg/datascience, CSDS spec, PII removal) not fetched (outside the listed sources).
- Lab folder: other items (`csds-to-tome.pdf`, `csds-manifest.pdf`, `envs/`, `experiments/`, `inputs/`, `models/`, `.bag` files) not read.
- Narrowing paths in builder.py are read at call level only; `narrowing.py` not read.
