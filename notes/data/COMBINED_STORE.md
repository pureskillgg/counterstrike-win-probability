# Combined dataset: one store for 2024 and 2026, with an audit trail

Current numbers for any build are in `data\<store>\prep_report.json` (written by `prepare.py`); this note explains how the store is made and what each rule is for. Written 2026-10-14.

## What "one folder" means

`tomes\combined\tome\csds\` has the layout of a dsdk tome: `header\dataframe_*` and, for each channel, `basic_<channel>.COMBINED\dataframe_*` plus `keyset_00000`. It is made by `scripts\data\build_combined.py` from every finished collection:

| Collection | Content | Notes |
|---|---|---|
| `2024-02-g` | 4,435 matches, 2023-12-28 to 2024-02-25, 24 channels | vetted; the older `2024-02` and `2024-02-all` are superseded |
| `2026-09` | September 2026 upload folders (3,023 matches on E: at 02:50) | needs `BUILD_OK` |
| `2026-10-01_to_08` | 1,752 matches | vetted |
| `2026-10-09`, `-10`, `-11`, `-12` | one collection per upload day | need `BUILD_OK`; days 10 and 12 only after the S3 sync finishes |

The build changes dtypes only (no row is dropped or edited). Rules: `string`/`large_string` become String; an int or float flag next to a bool becomes Boolean after checking that every value is 0 or 1 (the one exception is `molotov_state.was_extinguished_by_smoke`, which 2024 stores as -1 / 1 and which is mapped -1 to false, 1 to true); several int widths become Int64; int next to float becomes Float64; a column missing from one era is filled with nulls. Any other combination stops the build. `round_end` gains `player_count_norm`: 2024 logs `player_count` at twice the 2026 scale (20 for a full 5v5, 8 for wingman, 19 or 18 with a missing player), so the normalised column is `player_count // 2` for 2024 and the raw value for 2026.

Why a new store and not "copy the 2024 files into the 2026 folder": the two eras differ in 85 shared columns' types, in 14 channels the 2024 data lacks, and in 12 header fields (the schema comparison is `notes\spec\schema_2024_vs_2026.md`). Appending pages blindly makes polars concatenation fail or silently cast bool to int.

## Catalog and rounds (`scripts\data\prepare.py`)

**Scope.** A match is in scope when it is not wingman (`header.is_wingman`; for 2024, which has no flag, the modal normalised player count is 5 or less), not a hostage map (`cs_*`) and its map is one of de_ancient, de_anubis, de_cache, de_dust2, de_inferno, de_mirage, de_nuke. These seven are the 5v5 maps with real volume in 2026 (1,561 of 1,603 non-wingman matches in days 01 to 08; the next map has 21). Everything else carries a `scope_reason` (wingman, hostage, map_out_of_pool, map_unknown) so it can be counted and examined.

**Round filters**, first failing rule recorded in `rounds.round_status`: normalised player count 10; roster 5 T and 5 CT; no disconnect or bot takeover; a round_start row; a freeze-end row; every death in the live window has a resolvable side; a resolved label; at least one undecided second. 2024 `player_info` lacks one player (no team code) in about 14% of rounds although `round_end` counts ten; with 5 on one side and 4 on the other the missing player must be on the 4-player side (a side cannot have six), so those rounds are kept and that player's deaths take that side.

**Effective end.** The grid stops before the decisive event: the bomb outcome, or the death that wipes a side; the recorded `round_end` tick otherwise. In 2026 the two agree (median gap 0 ticks). In 2024 `round_end` is logged a median of 428 ticks (6.7 s) after the decisive event, so the recorded end would add about 6.7 seconds of decided states per round.

**Duplicates.** Two keys. The exact event signature (round_end, player_death, player_info). The tolerant key (map, first round, round count, header final scores, per-round winner and death count, at least 16 rounds) catches copies whose ticks or player ids were renumbered. Tried first without the death counts it produced false positives on short stomp matches (same winners, different games), which is why the death counts are in the key. Keeper of a group = the member with most usable rounds, ties to the lowest key.

## Labels: four signals and a vote

| Signal | Definition |
|---|---|
| recorded | `round_end.winner_team_code` |
| bomb | CT if a defuse event, T if an explosion, inside the round's live window |
| score | the side whose freeze-end score rose between round r and r+1 (side-attached scores; undefined where the fields swap sides, rounds 12, 27, 33, 39 ...; they do not swap between 24 and 25 or 30 and 31) |
| elimination | a side with all five dead; a wiped T side only counts if the bomb was not already planted |

Final label: the bomb outcome if there is one; otherwise the side with at least two of {recorded, score, elimination} (recorded does not vote on suspect rounds); a lone recorded signal is accepted on non-suspect rounds; otherwise the round is dropped. Nothing is flipped because of its round number.

**What the audit found** (on 2024 + 2026 days 01 to 08):
- 2026 (both game builds 10896 and 10924): the recorded winner agrees with bomb, score and elimination in 100% of rounds that have the signal (6,935 / 32,252 / 25,911). No relabelling.
- 2024 build 9842: 100% with bomb and score, 99.7% with elimination. No relabelling.
- 2024 build 9961 (2,478 of 4,435 matches): the recorded winner is inverted on the first round of each half-period, rounds 13, 25, 28 (2024 has no rounds beyond 30, so later overtime halves are untested). All 1,995 usable rounds of that kind differ from the recorded winner (13: 1,799; 25: 105; 28: 91); no other round of that build differs. Those rounds are relabelled by the bomb outcome (450) or by score and elimination agreeing (1,545); 172 rounds with fewer than two signals are dropped.
- Earlier notes (D5, D9, A6) described the same defect and estimated 2,040 affected labels; the vote reproduces it and gives a per-round audit trail (`rounds.label_source`, `rec_suspect`, `final_differs_from_recorded`).

## What is deliberately not here

No score lead anywhere in the model inputs (the freeze-end score difference is kept in the grid as `score_lead_diag` for ablations only). No player identity, no rank (the header averages are per match, identical for both sides, zero for most 2026 matches). No money or buy-choice scoring.
