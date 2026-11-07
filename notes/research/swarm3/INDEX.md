# Swarm 3 index: neural options vs XGBoost for CS2 round win probability

Date: 2026-10-13. Condensed from the synthesis output (task w8r1nb09a). Tags: OPENED-V = opened, numbers re-read by a verifier; OPENED = opened, not re-checked; SNIPPET = search text only; BLOCKED = 403/429; REPO-UNVERIFIED = repo summary only.

## Bottom line
- We have not trained a neural model. No source shows a neural net reliably beating tuned gradient boosting on a temporal split of a table our size with a probability metric.
- Closest CS:GO evidence: ESTA (arXiv 2209.09861, OPENED-V): an sklearn MLP beat untuned boosted baselines by 0.009 to 0.025 log loss on all seven maps. The split is random by state, not grouped by match, and ESTA does not use score as a feature.
- Our trees already span only 0.001 (XGB 0.4366, LightGBM 0.4376, HistGB 0.4369). Logistic regression on all features is 0.0173 worse, so interactions matter.
- Suggested adoption threshold for a neural model: 0.005 log loss better on a clean grouped split, with a paired interval excluding zero, and Brier and ECE not worse.
- Before any neural comparison, re-baseline XGBoost. Dedupe, the round-13 side-code defect (2024) and the batch effect can each move 0.4366.

## Neural evidence
| Source | Data | Result | Status |
|---|---|---|---|
| ESTA, arXiv 2209.09861 | CS:GO, 1,558 demos, one state per round, 7 maps | MLP lowest log loss on all 7 maps (Dust2 0.415, Mirage 0.430, Inferno 0.442, Nuke 0.416, Overpass 0.438, Vertigo 0.424, Ancient 0.431). Boosted baselines untuned | OPENED-V |
| ESTA, same paper | Same | Set Transformer worst on all 7 maps; Deep Sets ties or beats boosting on 2 maps | OPENED-V |
| Xenopoulos et al., arXiv 2109.12990 | CS:GO, game win at round start | 2-layer NN 0.736, XGBoost 0.739, LR 0.793 (weighted test LL) | OPENED-V |
| Xenopoulos and Silva, arXiv 2207.14124 | CS:GO, 14.3M one-per-second states | State MLP 0.4351; GAT+State 0.4276; GCN alone 0.6111. No boosted baseline, one split | OPENED-V |
| Leathart and Polaczuk, arXiv 2002.02644 | CS:GO, GRU over round sequences | NLL 0.3830 uncalibrated, 0.3793 with temporal calibration. No tree baseline | OPENED |
| McDonald and Ramachandran, ICSTA 2026 | LoL post-game stats (not live) | XGBoost ROC AUC 0.9660 beat TabNet 0.9625 and MLP 0.9475 | OPENED-V |
| Thesis on Dota 2 snapshot transformer (2025) | Dota 2 | Claims accuracy and calibration, no numbers | BLOCKED |

Tabular benchmarks: TabReD (temporal splits, 106K to 605K rows) ranks XGBoost 2.9, LightGBM 3.1, MLP 5.0 (OPENED-V). TabM, RealMLP and ModernNCA are about 1% or tied with GBDT, mostly on non-temporal benchmarks. TabPFN-2.5 and TabICL are GPU models with ceilings below our 230K rows and have licence limits.

## Cost and CPU inference
- LoL 21,910 rows: XGBoost 0.45 s, MLP 20.5 s, TabNet 46.7 s (hardware not stated).
- Published CPU batch-1 timings (other tasks): XGBoost 1.9 ms vs MLP 1.5 ms per query (2026 distillation paper). Harness overhead dominates, so CPU inference is not the constraint.

## Standout round signals
| Signal | Effect | Source | Status |
|---|---|---|---|
| Buy type (CS:GO) | Round win: full 59%, hero half 48%, half 34%, low 27%, eco 3% | 2109.12990 Table 1 | OPENED-V; confounded by score |
| Equipment value | Largest XGBoost gain importance | 2011.01324 Fig. 7 | OPENED-V; ranks only |
| Second round after lost pistol (CS:GO) | T side: eco 27%, half 63%; CT side: eco 6%, half 91% | 2109.12990 Table 5 | OPENED; not verifier-checked |
| Opening kill (CS2 2026) | Winners won 64.8 to 72.6% of rounds | scope.gg blog | OPENED summary; company blog, no n |
| First pistol win vs match win (CS:GO majors) | OR 2.95 (95% CI 2.13 to 4.07) | Yalciner and Kilci 2023 | SNIPPET only |
| Positions alone (GCN) | 0.6111 vs state 0.4351; adding to state gives no gain | 2207.14124 | OPENED-V |

## Pistol and buy-type definitions
- CS:GO buy thresholds (2109.12990): eco equipment 0 to 3k and spend 0 to 2k; full = equipment plus spend of at least 20k per team. Not checked for CS2.
- CS:GO MR15: pistols on rounds 1 and 16. CS2 MR12: pistols on rounds 1 and 13 (SNIPPET; our repo confirms the side flip at round 13).
- No source defines anti-eco.

## Data pitfalls to check against our repo
1. Duplicate matches across splits: 46 groups (97 matches) among channel-kept matches; 51 matches removed; the claim that none crosses the batch boundary is unverified.
2. Round-13 side-code defect (2024 tome): cause unverified.
3. Side convention: awpy 2 = T, 3 = CT. Keep one convention and check against the final score.
4. Halftime and overtime: a side is not a team identity.
5. Effective sample size: adjacent states are highly correlated (2011.01324). Use match-level bootstrap; standard bootstrap undercovers.
6. Batch effect: 2026-10-08 has 1,133 matches with almost all metadata unknown.
7. Leakage: rank_update rows sit in each match's last round (VERIFIED earlier). Neural models make leakage easier to hide.

## Ranked experiments
1. Re-baseline XGBoost on a deduplicated, match-grouped split with 5 seeds and a paired match-clustered bootstrap. Prerequisite. CPU minutes.
2. Same-feature MLP vs tuned XGBoost on the same folds (2 to 3 layers, dropout, early stopping). Adopt only under the 0.005 rule above. Minutes per run on the RTX 3080; CPU workable.
3. Per-map effect: one-hot map in XGBoost and per-map models. Published game-level effect is -0.009 (2109.12990); our per-map CT rates span 48 to 52%.
4. Set or graph model over 10 player slots plus state vector. Only if experiment 2 shows the flat MLP helps. Needs per-player features.
5. GRU over 1 Hz round sequences. Needs a demo re-parse (about 10 times our rows). Last.

Not recommended now: TabPFN-2.5 and TabICL (GPU memory, row ceiling, licence).

## Caveats
- Log-loss values come from different games, labels, splits and data sizes. None is directly comparable to our 0.4366.
- The "20% test-loss reduction" in 2207.14124 is relative to an older model; the in-table gain is 1.7%.
- Non-peer-reviewed sources: scope.gg, vlr.gg, CANA, Abios, Bayes Esports (unreachable).
- Repo GitHub leads (a CS2 live 1 Hz repo; a graph-temporal transformer) are REPO-UNVERIFIED; their demos are private.
- Full raw output: task w8r1nb09a output file (search and verification JSON).
