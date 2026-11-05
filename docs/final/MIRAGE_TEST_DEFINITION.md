# de_mirage test: the rule the export follows

Status: NEEDS OWNER CONFIRMATION. Written on 5 Nov 2026 from the export index files and the split manifests. The owner has not confirmed the rule, the set or the counts below. No split file was changed to write this.

Decision on 5 Nov 2026 (delegated to the assistant by the user): the split of record for the export is split_manifest_v2, which is what the export already uses. The frozen de_mirage count (687 or 1,033) is still open for the owner. Note that a Meridian pin on the split manifest says the 1,033 count was accepted by the owner, which this repo note could not find.

## Sources

- Export: `export_clean_v2` on the space (`README.json`, `match_split.npy`, `round_map.npy`, `round_match.npy`, `match_keys.json`). Copies were made with scp. `match_keys.json` (sha256 e560a05f...) and `round_match.npy` (sha256 a07007bb...) were checked against the space; `match_split.npy` (57,939 bytes) and `round_map.npy` (1,058,818 bytes) match the space listing sizes.
- Split manifests: `data/combined/split_manifest_v2.json` (current), `data/combined/split_manifest_v1_archive.json` (the frozen v1 assignment, 29,885 keys).
- Counts were made with `scratchpad/loadout_fix/mirage_count/count_mirage.py` and `label_nonexport.py`. They are counts only. No score was computed on any test set.

## The rule the export follows

1. The export's split is `split_manifest_v2`. The export README says so in its `split` field: "frozen v1 + appended 80/10/10 by map, seed 20261104".
2. Export split counts, matches: train 47,501, val 4,633, test 5,677, total 57,811. The `match_split.npy` codes are 0 = train, 1 = val, 2 = test; the code-to-label pairs agree with `split_manifest_v2.json` on all 57,811 export keys.
3. Map of each match: the map of its rounds (every round agrees with its match's map).
4. The v1 frozen de_mirage test is not in the export. Of the v1 manifest keys that are in the export, de_mirage matches are train (5,795) or val (420), and none is test (0).
5. The export's de_mirage test matches are the v2-appended ones: 859 matches, none from v1.
6. 1,531 v1 test keys (of 4,264 v1 test keys) are not in the export. The v2 manifest has 1,537 test keys not in the export. The map of these keys is not in the export files, so whether any of them is de_mirage is not established here.

## Counts (de_mirage, matches in the export)

| Group | Train | Val | Test |
|---|---|---|---|
| All export matches | 12,661 | 1,278 | 859 |
| v1 frozen (in export) | 5,795 | 420 | 0 |
| v2 appended (in export) | 6,866 | 858 | 859 |

Total de_mirage matches in the export: 14,798.

## Figures in the request that the files do not confirm

- "v1 frozen de_mirage test of 1,033, excluded from all splits": not reproduced. The export has 0 v1 de_mirage test matches, which is consistent with exclusion, but 1,033 has no source in the files read here. The internal baseline record (`data/combined/internal_baseline_de_mirage_test.json`, split "test (frozen manifest)") gives 687 matches and 865,974 rows. The decision register (`notes/decision_register_2026-10-27/audit_review_2026-10-27.md`) also names 687 matches for the de_mirage frozen test.
- "v2 appended de_mirage test of 860": the export count is 859.
- "owner accepted 1,033 as the frozen test": no owner record of that was found in `notes/` or `HANDOFF.md`. The decision register recommends retiring the 687-match de_mirage test for decisions (its D5), and keeping it for descriptive reporting only.
- "export README cites v2 for splits": confirmed (point 1 above).

## What the owner needs to decide

1. Which set is the de_mirage frozen test: 687 matches (baseline and decision register), 1,033 (request), or some other count. Give the source of the number.
2. Whether any of the 1,531 v1 test keys absent from the export are de_mirage, and whether all v1 de_mirage test matches are excluded from every split. A map lookup for those keys is needed to answer this.
3. Whether the appended test count is 859 (export) or 860 (request).
4. Whether this rule stands as the record. Until then, treat this file as a description of the export, not as a decision.
