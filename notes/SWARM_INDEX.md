# Swarm index

Built 2026-10-13 from the four files matching `notes\research\swarm_*.md` and `notes\eda\swarm_*.md`. Figures are copied from those reports; the status column says what the report itself marks as verified.

## Index

| File | Question | Key number | Verified or unverified | Changes the plan? |
|---|---|---|---|---|
| `notes\research\swarm_R1_new_papers.md` | Are there 2024-2026 CS2/CS:GO round win-probability papers not already in the 25-entry note? | No full-text round-level WP paper verified. Parshakov 2026: within-map intra-class correlation about 0.074 on first-half round scores. UFSCar 2026 pre-match match-level holdout ROC-AUC 0.683. | Mixed. Parshakov read from PDF (VERIFIED). Most WP papers are abstract-only or blocked (UNVERIFIED). UFSCar is metadata page only. | Yes, for evaluation. Use map- or match-grouped CV and clustered intervals. Do not cite the Xu and Moka p-values, because the target is computed from the test matches. The ISRITI 92% accuracy needs a leakage check. |
| `notes\research\swarm_R2_industry.md` | Do industry or product pages disclose a CS2 round win-probability model class, features or validation numbers? | No vendor gives defined metrics. Best open match-level baseline (lc-leonardo, LightGBM): ROC-AUC 0.72, Brier 0.231, about 5-8% Brier skill over a coin flip (approximate). | Mostly VERIFIED page reads, but summaries are model-extracted and paraphrased. Many sources blocked (HLTV, Bayes primary, Valve GSI). Benchmark arithmetic VERIFIED by script. | Marginal. No vendor benchmark to beat. Leetify's four economy groups per side is a feature idea. |
| `notes\research\swarm_R3_attribution.md` | How can win-probability change be attributed to individual players, and can player IDs support cross-match attribution? | Exact Shapley over 10 slots needs 1,024 coalitions per state. No global player key in the tomes: steam_id has 19 distinct values in 2026 and is a slot label, not a SteamID. | Method formulas VERIFIED from PDFs. ID probes VERIFIED. Literature from abstracts UNVERIFIED. Feasibility reasoning UNVERIFIED. | Yes. Season-level ratings and MVP are out unless a key is found. The absent-player value function forces either a masked-slot model or a team-aggregate counterfactual. |
| `notes\eda\swarm_E2_map_profile.md` | What are the per-map CT win rate, plant rate, round length and kill rates in the 2026 and 2024 tomes? | CT win 52.04% over 31,065 rounds (2026); 50.72% over 72,782 rounds (2024). | VERIFIED: code run, independent recount, side-code checks. Causes of the 2024 round-13 defect and the unknown-attacker deaths UNVERIFIED. | Yes. The 2024 round-13 side-code defect affects labels. The 2026-10-08 batch split affects validation design. |

## Three findings most likely to change the model

1. 2024 round 13 side code disagrees with the bomb outcome in 434 of 605 rows (matches dated 12-25 Feb 2024). Cause unresolved (E2). Repair or exclude these rounds before using 2024 labels or side features.
2. The 2026-10-08 batch shifts CT rates by map (de_cache 54.2% vs 47.1% on other days; de_dust2 49.6% vs 53.9%) (E2). Validation should be batch-aware, and map-side priors may be batch artifacts. The link to `notes\eda\batch_effect_2026-10-08.md` was not checked.
3. Exact Shapley needs values for absent players, and the tomes have no cross-match player key (R3). This rules out player-identity features across matches and sets the model design.

## Three findings most likely to be wrong

1. R3 says slot IDs are sufficient for every within-match method. Its own counts show 1,216 of 1,751 matches with exactly 10 fixed IDs, 285 death rows disagreeing with the mapping and 69 unmapped. Per-round slot identity is not verified, and the halftime side-swap reading is UNVERIFIED.
2. E2 reports a kill-rate era gap (6.61 per round in 2024 vs 7.14 in 2026). The 2024 tome has 4.8% unknown-attacker deaths vs 0.2% in 2026, and the cause is UNVERIFIED. The gap may be a data artifact.
3. R2's Brier-skill figures assume balanced outcomes. The lc-leonardo base rate is unknown, and the table mixes per-game-state (Xenopoulos) and per-match rows. Treat the skill values as approximate.

## Not indexed

- R4 (evaluation practice): no `swarm_R4_*.md` file exists. Results are recorded in a Meridian note only.
- E4 (day stability): output is `scripts\swarm\out\e4_day_stability_run1.txt`, not a `swarm_*.md` file.
- E3 (trades): still running when the agents reported.
- One fixed rerun was reported as pending, with no agent named.
