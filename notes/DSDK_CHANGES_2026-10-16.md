# DSDK fork changes (incremental tomes)

Date: 2026-10-16. Repository: `G:\pureskill\dsdk-fork`, branch `incremental-tomes`, commit `e8d0e24`, on top of tag `4.1.1` (`779a56c`).
Upstream: `pureskillgg/dsdk` (origin). The commit message says the fork is local and not pushed anywhere. No pull request exists yet.

## What the commit does

Tome builds can now be resumed after an interruption, and a finished collection can be extended with newly uploaded matches, without rebuilding the old pages. The feature is off by default.

- `writer_fs.py`: manifests and pages are written all-or-nothing (temporary file, fsync, atomic replace).
- `builder.py` and `curator.py`:
  - `resume=True` keeps the `.building` staging folder and reads only the matches and channels it lacks.
  - `behavior_if_complete="extend"` appends matches that appeared since a complete build. It writes new pages, replans the narrowing over old and new signatures, and regenerates the header.
  - Manifests carry additive `signatures` and `resume` fields.
- `incremental.py` (new): ledger reading from manifests and keysets, and removal of orphaned pages.
- `README_FORK.md` (new): the on-disk contract and the crash-safety argument.
- Tests: `incremental_test.py` (1,031 lines) and `writer_fs_test.py` (142 lines).

Diff size: 7 files, 2,329 insertions, 70 deletions.

## Why it mattered

The CS2 tome is built in day-sized chunks as new uploads arrive. Before this change, an interrupted chunk meant a full rebuild, and new days could not be added without redoing old ones. The extend path adds new matches to a finished collection. The release B run appended the new day in 428 seconds (`notes`: STATUS 2026-10-15), without rebuilding the earlier days.

## Open before a pull request

- Decide the target: push the branch to a fork under your GitHub account and open the PR against `pureskillgg/dsdk`, or keep it local.
- Run the test suite on the branch.
- The commit's co-author line names Sonnet 5.5. New commits in this work carry Haiku 5.5.
