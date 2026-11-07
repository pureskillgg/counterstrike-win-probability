# 2024-02 CSDS copy comparison

Read-only comparison of the three 2024-02 copies on E:. Nothing under E: was written, moved, renamed or deleted, and no packages were installed.

Copies compared (day 01 unless noted):
- A: `E:\csds_extracted\csds\2024\02\01`
- B: `E:\csds_recovered_20261011\csds\2024\02\01`
- C: `E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02\01`
- C days 02 to 25: `E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02\NN` (summary counts only)

Method: a metadata inventory (names and sizes) of every file in each copy, then set comparison. Full hashes were not computed. The first and last 1 MB were SHA-256 hashed only for a 37-file A-vs-B sample. The seven repaired files, all under 1.2 MB, were compared byte-for-byte. Scripts are in `G:\pureskill\csds-cs2\scripts\spec\`.

## 1. Match folders and overlaps

| Copy | Match folders | Files | Bytes | Empty match folders |
|---|---:|---:|---:|---:|
| A | 227 | 7,029 | 9,446,610,442 | 0 |
| B | 227 | 7,029 | 9,446,610,442 | 0 |
| C | 230 | 107 | 137,655,909 | 219 |

- Only in A: 0 folders.
- Only in C: 3 folders: `m_cc14b2b9d30731d13d73b9ca`, `m_2f33614d3d33c32d8318f1f4`, `m_d89dbf28514d7682ff045917`.
- In both A and B: 227. In both A and C: 227. In all three: 227.
- **A and B are identical by folder set, file names and file sizes** (all 7,029 files). The 37-file byte sample (7 repaired, 10 largest, 20 random with fixed seed) matched on first and last 1 MB. Full byte identity is not proven, because that would need full hashing of multi-GB files.

## 2. A vs C in the 227 shared folders

| Status | Files |
|---|---:|
| Same name and size | 7 |
| Same name, different size | 0 |
| Missing in C (present in A) | 7,022 |
| Missing in A (present in C) | 7 |
| **Total differing or missing on one side** | **7,029** |

- The 7 same-size files are the repaired files in section 3.
- The 7 files missing in A are all in `m_bb0ad43bd38ae7c458c92d68`: `round_end`, `round_mvp`, `round_start`, `round_state`, `tick`, `weapon_action`, `weapon_fire`.
- The 7,022 files missing in C are: all 6,789 files in the 219 empty C folders (219 x 31), the 30 non-repaired files in each of the 7 repaired folders (210), and the 23 files in `m_bb0ad43b` (23).
- Files in C outside the shared folders: 93, all in the three C-only folders (31 each).
- Full per-file list (7,036 paths, with status): `G:\pureskill\csds-cs2\notes\spec\A_vs_C_file_rows_day01.csv`.

## 3. The seven repaired files (from the handoff)

| Match folder / file | Size A | Size B | Size C | Sizes match | Bytes identical A, B, C |
|---|---:|---:|---:|:---:|:---:|
| `m_9805aa61-.../bomb_action` | 17,488 | 17,488 | 17,488 | yes | yes |
| `m_c40f829f-.../bomb_action` | 18,913 | 18,913 | 18,913 | yes | yes |
| `m_3203b994-.../other_death` | 21,767 | 21,767 | 21,767 | yes | yes |
| `m_856031b4-.../tick` | 1,131,487 | 1,131,487 | 1,131,487 | yes | yes |
| `m_7dbe5dcf-.../player_footstep` | 9,455 | 9,455 | 9,455 | yes | yes |
| `m_59e98355-.../weapon_fire` | 161,273 | 161,273 | 161,273 | yes | yes |
| `m_1547a130-.../player_hurt` | 66,811 | 66,811 | 66,811 | yes | yes |

All seven exist at the same path in A and B. A's copies are byte-identical to the C copies that the handoff says match the archive CRC, so A's copies are equally good.

## 4. Empty match folders in C (day 01)

- **219 of 230** C match folders are empty (95.2%). Every one of them is populated in A with 31 files.
- Five examples: `m_94fc9d1cc34dccf22b794d50`, `m_cea98c8b60b1c4e50a9aa290`, `m_7091cd950390b7f8e8dc280d`, `m_2b108c38d6ef5452395901ae`, `m_dae629431e92d3caf6c9b740`.
- Full list: `G:\pureskill\csds-cs2\notes\spec\C_day01_empty_match_folders.txt`.

## 5. Name completeness

- The union of file names across A is 31. 226 of A's 227 folders have all 31.
- `m_bb0ad43b` has only 23 files in A, missing 8 names: `player_vector`, `round_end`, `round_mvp`, `round_start`, `round_state`, `tick`, `weapon_action`, `weapon_fire`.
- The three C-only folders each contain exactly the 31 names, so they look like complete matches.
- After adding C's 7 files, `m_bb0ad43b` has 30 names. **`player_vector` is missing from all three copies** (A, B and C).
- The salvage index `E:\csds_rar5_tail_salvage_20261011.tsv` lists the seven C-only `m_bb0ad43b` files but no `m_bb0ad43b/player_vector` entry. The index has only 3 `player_vector` lines under 2024/02/01 in total, so it looks partial. The absence there is not proof the file is missing from the source archive.

## 6. Days 02 to 25 (summary counts only, copy C)

Total across 24 days: **4,210 match folders and 130,449 files**. Day 01 adds 107 files, so C holds 130,556 files across days 01 to 25.

| Day | Folders | Files | Day | Folders | Files |
|---|---:|---:|---|---:|---:|
| 02 | 188 | 5,828 | 14 | 180 | 5,580 |
| 03 | 158 | 4,898 | 15 | 116 | 3,566 |
| 04 | 155 | 4,805 | 16 | 173 | 5,365 |
| 05 | 122 | 3,782 | 17 | 110 | 3,410 |
| 06 | 129 | 3,969 | 18 | 147 | 4,557 |
| 07 | 4 | 124 | 19 | 207 | 6,417 |
| 08 to 11 | 0 | 0 | 20 | 264 | 8,184 |
| 12 | 1,004 | 31,123 | 21 | 246 | 7,625 |
| 13 | 219 | 6,789 | 22 | 209 | 6,478 |
|  |  |  | 23 | 157 | 4,867 |
|  |  |  | 24 | 224 | 6,944 |
|  |  |  | 25 | 198 | 6,138 |

Not examined: the handoff's eight unresolved files sit on days 18 to 24. They are outside this day-01 comparison.

## 7. Recommended non-overwriting merge plan

Nothing has been copied. The proposed manifest is `G:\pureskill\csds-cs2\notes\spec\merge_manifest_day01.csv` (7,136 rows).

1. **Base: copy A.** It holds all 227 shared match folders and 7,029 files (9.45 GB). B is identical by name and size, so it is not needed as a source. Keep it as a cross-check copy. Leave B and C untouched.
2. **Add from C: 100 files, 136.2 MB.**
   - `m_bb0ad43b`: add the 7 files A lacks (`round_end`, `round_mvp`, `round_start`, `round_state`, `tick`, `weapon_action`, `weapon_fire`).
   - The three C-only folders (`m_cc14b2b9`, `m_2f33614d`, `m_d89dbf28`): add all 93 files.
3. **Do not copy the 7 repaired files from C.** A's copies are byte-identical. The 7 C duplicates (1.43 MB) are marked `skip_duplicate` in the manifest.
4. **Empty C folders (219): copy nothing.** A already holds their full content. Do not recreate them as empty folders in the merged tree.
5. **Mechanics.** Write into a new, empty target directory on G:, never E:. Use copy-if-absent: stop on any existing destination and never overwrite. Verify each copy by size against the manifest. Byte-level spot checks can use first and last 1 MB SHA-256, so multi-GB files are not fully hashed.
6. **Open item: `m_bb0ad43b/player_vector`** is missing from A, B and C. Before treating it as lost, list `E:\csds.rar` read-only and look for that path. Do not fabricate the file.

## Key numbers

- **7,029 files / 9.45 GB** in A, identical by name and size to B. This is the base.
- **100 files / 136.2 MB** to add from C (7 for `m_bb0ad43b`, 93 in three C-only folders).
- **219 of 230** C day-01 match folders are empty. A covers all of them.
