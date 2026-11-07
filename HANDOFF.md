# CS2 round win-probability project: handoff

Written 2026-10-13 so another session or model can continue without this conversation. Update the "Status" section when anything changes. Everything here is either VERIFIED (checked in code or data) or marked UNVERIFIED.

## Ground rules
- Never read or edit `.env` or `meridian.toml` (project rule). Credentials come from the Windows user environment: `ZOTERO_USER_ID` (numeric) and `ZOTERO_API_KEY` (write access).
- Do not write under `E:`. It is a slow USB hard drive that holds raw match files. All outputs go to `G:\pureskill\csds-cs2`.
- Do not commit or push anything unless the user asks.
- Subagents: Haiku only, as the user requested.
- Meridian project: `fa6f6e09-513d-4788-9db5-43123df4ce0f`.

## Where things are
| What | Path |
|---|---|
| Raw match files (per-tick, slow) | `E:\csds_final\csds\2026\10\<DD>\<match id>\<channel>`, plus the 2024 tail copy at `E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02` |
| 2026 tomes, days 01 to 08 (canonical) | `tomes\2026-10-01_to_08` (header + 35 channel tomes; empty `player_inputs` removed) |
| 2024 tomes, canonical | `tomes\2024-02-g` (4,435 matches, 24 channels; verified row-for-row equal to `2024-02-all`) |
| 2024 tomes, superseded | `tomes\2024-02` (partial, 4,208 matches), `tomes\2024-02-all` (E-sourced, comparison only), `tomes\pilot-2026-10-01_to_08` |
| 1 Hz in-round state sample (150 matches, days 01 to 07) | `data\state_sample\players_1hz.parquet`, `data\state_sample\rounds.parquet` (validated) |
| Baseline model | `models\xgb_primary_seed0.json` and `_meta.json` |
| Model card | `notes\model\MODEL_CARD.md` |
| Bake-off script and results | `scripts\model\bakeoff.py`, `notes\model\bakeoff_results.md`, `logs\bakeoff.log` |
| Specs and schemas | `notes\spec\` (SPEC_DIGEST.md, feature_spec.md, csds_dictionary.csv, schema_2024_vs_2026.md) |
| EDA reports | `notes\eda\` (flow.md, combat.md, players.md, batch_effect_2026-10-08.md, priors_by_map_round.md, economy_snapshot_prototype.md, state_sample_extraction.md; ablation reports pending) |
| Research notes | `notes\research\` (literature, GBDT checks, tabular foundation models, CS2 mechanics, feature feasibility, industry, model alternatives) |
| Zotero collection | "CS2 CSDS Round Win Probability" (online library; 68 items filed, 33 more from later research) |
| Pixi environment | `.pixi\envs\default` (python 3.12, xgboost 3.4.2, lightgbm 4.7.0, scikit-learn 1.9.1, pandas 3.0.6, pyarrow 25, duckdb 1.5.6, polars 2.0) |

## Status (as of 2026-10-13)
- **Tomes:** done for 2026 days 01 to 08 and for 2024.
- **Days 09 and 10:** on disk but unresolved. They hold 2,301 and 2,758 match folders, against about 1,162 each in the revision stats, so they may contain duplicates or be incomplete. Do not build from them until the counts are reconciled.
- **Baseline model:** saved. Test log loss 0.4366, Brier 0.1464, AUC 0.870 (VERIFIED on the reference run). It uses 262 of 362 trees, which the prediction must set with `iteration_range=(0, 262)`.
- **Provenance problem:** `bakeoff.py` was edited after the run that produced the saved results. A reference rerun was interrupted by a session restart. The saved model and results may not match the current script. Re-run the bake-off to confirm before relying on the numbers.
- **1 Hz state sample:** done and validated. Alive counts match the kill events on 99.995% of snapshots. One round (`m_7392c5c6d537a57c2e735e98`, round 16) is flagged as an anomaly.
- **Ablations (health, equipment, weapons, utility; positions and map control):** running as a Haiku workflow. Their reports will appear as `notes\eda\ablation_*.md`. Until they exist, the question of which features matter is open.
- **Research:** done for the literature, the GBDT checks, the CS2 mechanics and the feature feasibility. Liquipedia blocks automated fetches (HTTP 403). Valve and Steam are reachable.
- **Zotero:** the filer is written and tested (37 tests pass), but its changes are uncommitted in the Meridian worktree `desktop-zotero`. Papers are filed as records only. PDFs are not attached (Meridian proposal `a6d590e7-3c57-4a0e-acda-f4828b706e52`).

## Cautions for anyone continuing
1. **Halves are 12 rounds (MR12),** not 24. The phase flips after round 12.
2. **Score fields:** `round_state.t_score` and `ct_score` are mostly side-attached, but 15 to 20% of matches on days 01 to 02 are team-attached (see the Meridian note "round_state score fields"). The bake-off's `ct_score_minus_t_score` feature can have the wrong sign for those. Derive the score lead from round winners and the swap rule instead.
3. **Player IDs are per-match.** Cross-match player features are invalid. `player_id_fixed` as an alias is not fully verified.
4. **Sentinels:** older matches use 65535 for attacker and assister IDs. The `round_start` tick of 66 in round 1 is a placeholder in 39% of matches. Raw `player_id = 0` appears as a placeholder.
5. **Leakage:** `rank_update` is post-match, so exclude it. `bomb_defused`, `bomb_exploded` and round-end fields must not feed a prediction at an earlier tick.
6. **Dead columns:** `item_equip.def_index` is 0 everywhere. Money is not in the tomes; it exists only in the raw `player_status` on `E:`.
7. **Redacted fields:** player names, Steam IDs, chat text, demo IDs and sharecodes are redacted. Demo IDs cannot be used to check duplicates.
8. **Batch effect:** 2026-10-08 is about 72% of clean rounds and has gaps in `bullet_damage` and `player_sound`. Its metadata (match type, platform) is unknown for 93% of its matches. A held-out-batch test is still needed.
9. **Post-round logging:** 2 to 4% of weapon, grenade and molotov rows fall after the round's end. Use an explicit tick cutoff.
10. **Duplicates:** the bake-off's current dedupe removes 47 duplicate matches (the baseline agent once counted 51 removed matches). Verify the count when rerunning.

## How to reproduce the main pieces
Run from `G:\pureskill\csds-cs2` with `pixi run python -I -X utf8 <script>`.
- Bake-off: `scripts\model\bakeoff.py`. Takes about 20 minutes. Writes `notes\model\bakeoff_results.md` and `logs\bakeoff.log`.
- 1 Hz sample: `scripts\eda\extract_state_1hz.py`, then `scripts\eda\verify_state_1hz.py`. Takes about 1 minute.
- Zotero filing: from the Meridian worktree `desktop-zotero`, run `pixi run python -m meridian.zotero_filer file --collection "CS2 CSDS Round Win Probability" --bib <files> [--dry-run]`. Re-running is safe; duplicates are skipped.

## Open decisions for the user
- Delete the superseded tomes (`tomes\2024-02`, `tomes\2024-02-all`, `tomes\pilot-2026-10-01_to_08`). Only after the user confirms.
- Reconcile days 09 and 10 before any build.
- Whether to attach the PDFs to Zotero (proposal `a6d590e7`).
- The user's statement draft is theirs, not part of this project.
