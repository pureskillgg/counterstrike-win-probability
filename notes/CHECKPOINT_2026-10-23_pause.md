# Checkpoint 2026-10-23: paused before a battery swap

Background jobs stopped: the controlled-ablation run (diagnose_why_models_lag.py) and the chunk rebuild for 16 and 17 October (extend_tomes_chunks.py). Nothing else was started.

## State on disk

- **Combined store (tomes/combined):** restored from tomes/combined.bak. The ledger COMBINED_BUILD.json is byte-identical to the backup: 25 sources, last commit 2026-10-15 15:04. The 14–21 October sources are NOT in the store yet. The append that failed left no commit.
- **Partial append (tomes/combined.partial_2026-10-23, 21 GB):** the failed append's output. Not needed. Keep until the restored store is verified, then delete to free space.
- **Tome chunks 2026-10-14 to 19:** built, BUILD_OK, except the three quarantined below.
- **Quarantined (tomes/_bad_pages_2026-10-23):** 2026-10-16_c03 (old), 2026-10-17_c01, 2026-10-17_c03. Each has an unreadable page in basic_player_inputs or basic_player_sound ("Invalid thrift protocol error" or "Data corruption detected"). Not model inputs, but the append cannot read them.
- **Rebuilt 16 October:** 2026-10-16_c03 (new, 226 matches) BUILD_OK. Its first attempt failed (exit 1), and the second succeeded. Log: logs/extend_2026-10-16_c03.log.
- **17 October, unfinished:** 2026-10-17_c06 is partial (no BUILD_OK). 2026-10-17_c07 is not built. A stale lock tomes/2026-10-17.LOCK is left behind by the stopped run.
- **Built and checked:** 2026-10-20_c01, 2026-10-20_c02, 2026-10-21_c01 (BUILD_OK). 2026-10-17_c02, c04, c05 (BUILD_OK).
- **Code changes, untested:** split_manifest.py has a new --extend-train mode (train-only for new matches; the frozen test and validation are untouched). It needs its selftest. The ablation script diagnose_why_models_lag.py now has the round-weight column; its results are not in yet.

## Resume (in order)

1. Confirm no extend or ablation process is running. Then delete tomes/2026-10-17.LOCK.
2. Delete the partial tomes/2026-10-17_c06 (no BUILD_OK). Rebuild 17 October with `extend_tomes_chunks.py --days 2026-10-17` (E: reader, serial, no GPU job alongside).
3. Full-read every page of all 14–21 October chunks, and confirm none is unreadable. Earlier, a full read found 3 corrupt pages out of 6,295.
4. Append with `append_combined.py --out tomes/combined --sources <the BUILD_OK chunk names>`, dry-run first. The append keeps its own backup.
5. Continue the runbook: prepare.py, make_grid.py, the train-only split extension (after its selftest), positions and status for 14–21 October, round-state arrays, then the GRU and THGNN retrains.
6. Rerun the ablation (diagnose_why_models_lag.py) on CPU.

## Owner decisions still open

- 2024 February: whether the position models should include it. The baseline cannot, because cache_snapshot_2026 has no 2024 rows and its builder has not been found.
- The 2026-10-08 to 10-19 batch (about 78% of 2026 matches): keep, cap or reweight.
- Whether 2024 holdout matches stay held out.
