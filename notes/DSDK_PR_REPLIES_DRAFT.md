# Draft replies for the DSDK incremental-tomes PR (not posted; review before sending)

Branch: `incremental-tomes` in the fork. Changes are in the working tree, not committed.

## 1. `builder.py` (header can omit keys after a stopped extension)

The window is real and by design. Channels are committed first and the header last, because the header is the list of matches.
A stop between the two leaves matches in the channels that the header does not list yet. The next extension reads the header's list,
adds those matches to the header only (the channels already hold them and are skipped), and ends with the same collection as an uninterrupted
build. New test: `test_extend_stopped_after_the_channels_and_before_the_header_is_repaired`.

## 2. `builder.py` (channels published after the header commit)

Fixed. New channels are now published before the header commit, as the docstring says. Their header copy is taken from the staged
header (the committed one does not yet list the new matches). New test: `test_extend_publishes_a_new_channel_before_the_header`.

## 3. `builder.py` (header page replaced non-atomically)

Two files cannot be replaced atomically together, and the header page keeps its names because tools read `header/keyset_00000`.
The rows are replaced first and the keyset (the match list) last, so a stop in between leaves the old list: the next run adds the matches
back and rewrites the rows. A stop never leaves a listed match without rows. New test:
`test_extend_stopped_between_the_header_files_is_repaired_by_the_next_run`. A stronger fix (a new header page number with one manifest
replacement) would break the tools that read `header/keyset_00000`; that is a decision for the maintainers.

## 4. `incremental.py` (orphan cleanup misses `rewrite_*` staging files)

Fixed. The page-file pattern now includes the `rewrite_` prefix, so a staged rewrite left by a stopped build is removed after the next
commit. The existing unlisted-files test now also covers `rewrite_dataframe_00003` and `rewrite_keyset_00003`.

## 5. CI and tests

No CI has run on the PR, as noted. The full local suite was run with `pytest pureskillgg_dsdk` (log: `logs/dsdk_fork_full_tests.log` in the
project). Result: see that log before posting.

## 6. `README_FORK.md`

Moved to `docs/incremental-tomes.md`. The fork's notes now sit with the other docs in `docs/`.
