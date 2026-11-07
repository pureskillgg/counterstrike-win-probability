# R1: new CS2 / CS:GO round win-probability literature, 2024-2026

Compiled 2026-10-13 for Meridian project fa6f6e09-513d-4788-9db5-43123df4ce0f.
Scope: round (or match) win-probability and round-outcome papers on CS2 and CS:GO, dated 2024-2026, not already in `notes\research\model_choice_literature.md` (25 entries). Pre-2024 items are listed separately as precedents.

## Status legend

- **VERIFIED-PDF**: PDF pages read with the Read tool (path or URL given).
- **VERIFIED-PAGE**: a web page fetched with WebFetch and the content read (URL given).
- **VERIFIED-ABSTRACT**: abstract read from a search-index record (paper_search, OpenAlex or Crossref). Body not read.
- **UNVERIFIED**: metadata only, or the publisher page was blocked or empty.

## Bottom line

1. No full-text CS2 round-level win-probability paper from 2024-2026 was verified. The one round-level CS:GO hit from 2024 (Hirota, GCCE) is abstract-only.
2. Two CS2 match-level pre-match predictors from 2025-2026 were found. The 2026 thesis (UFSCar) gives an honest baseline: ROC-AUC 0.683 on a temporal holdout.
3. The only quantitative round-dependence number found is an intra-class correlation of about 0.074 for first-half round scores (Parshakov 2026), which bears on how round-level evaluation should be grouped.
4. None of the opened papers reports an effect size for team economy, team health or bomb state.

## Search log

- **arXiv (paper_search):** 5 queries. "Counter-Strike 2 round win probability" and "Counter-Strike round outcome prediction economy bomb" returned off-topic results. "Counter-Strike" sorted by date returned 30 hits (output too large to parse; the CS2 items it showed were non-WP). The real-time-WP query returned off-topic results.
- **OpenAlex (paper_search):** 8 queries. Useful for abstracts of Hirota 2024, Thai 2025, Kar 2026 (title record only) and the ISRITI 2025 and Frontiers 2026 CS items.
- **Crossref (paper_search):** 4 queries. Surfaced Hirota 2024, Kar 2026, Thai 2025, Sinap 2023, Huang 2022 (metadata only).
- **Semantic Scholar:** HTTP 429 on 3 calls. No citation counts or TL;DRs obtained.
- **WebSearch:** 1 query on the Hirota title. No hit for the paper itself. Returned a summary of Xia, Salinas and Morstatter (IEEE CoG 2025) on 2022 pro Dust 2 round and death prediction, which matches note entry #11 ("Precision under fire"). Treated as the same paper, still UNVERIFIED on body text.
- **WebFetch / publisher access:** IEEE Xplore returned HTTP 418 for abstract pages (10760945, 11393205), and blank pages for the full-text redirects (10760945, 11114397, 11393205). Springer redirected to an IdP login (10.1007/978-981-96-9983-4_5). UFSCar repository metadata page read (VERIFIED-PAGE). arXiv 2409.05052 and 2610.05251 PDFs read via Read (VERIFIED-PDF).

## The six papers opened

| # | Paper | Year | Venue / ID | Unit and target | Features (economy / health / bomb) | Model | Split | Metrics | Effect sizes | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Hirota, "Predicting Win Conditions of Counter-Strike: Global Offensive for Analyzing Round Progression" | 2024 | GCCE 2024; https://doi.org/10.1109/gcce62371.2024.10760945 | Round. Predicts the win condition (how the round is won) at the end of each round, not the match | Player coordinates during frames within rounds (per abstract). Economy, health and bomb not named in abstract | Set Transformer | Not stated | Not stated | None in abstract | VERIFIED-ABSTRACT (OpenAlex W4406265769, read via paper_search). Body blocked (IEEE HTTP 418). |
| 2 | Kar, Chakrabarty, Sarkar, Kole, "Counter-Strike Winner Prediction Using Machine Learning" | 2026 | Springer LNNS; https://doi.org/10.1007/978-981-96-9983-4_5 | Unknown (round or match) | Unknown | Unknown | Unknown | Unknown | None | UNVERIFIED (title and authors only; Springer IdP redirect; OpenAlex W7160195820 has no abstract) |
| 3 | Thai and Scirea, "Trajectory Analysis and Prediction in Counter-Strike with LSTM Models" | 2025 | IEEE CoG 2025; https://doi.org/10.1109/cog64752.2025.11114397 | Player trajectory regression (not win probability) | Player positions, ESTA dataset | LSTM | Not stated | Test MAE 14.44, MSE 477.24 | None | VERIFIED-ABSTRACT (OpenAlex W4413321661). Body blocked. Not WP; adjacent. |
| 4 | "Predicting CS2 Match Outcomes Using Machine Learning" (ISRITI) | 2025 | IEEE ISRITI; https://doi.org/10.1109/isriti68345.2025.11393205 | Match outcome (pre-match or in-match not stated) | Teams, players, historical match data, maps, win rate, "in-game metrics" (not specified). Data from CSStats | Logistic regression, SVM, RF, KNN; LR best | Not stated | "Average accuracy 92%"; precision, recall, F1 and ROC-AUC mentioned without values | None | VERIFIED-ABSTRACT (OpenAlex W7130726549). Body blocked (IEEE HTTP 418). **Leakage risk:** if "in-game metrics" are from the same match, 92% is inflated. UNVERIFIED. |
| 5 | Herdy Rocha, "Predicao de resultados de partidas de profissionais de Counter-Strike 2 com aprendizado de maquina" (UFSCar undergraduate thesis) | 2026 (issued 2026-07-01) | https://repositorio.ufscar.br/items/05d81da4-11fa-4a99-a98c-2935fbead599 | Pre-match win probability for one team (match level) | 11 pre-match difference features: recent statistics, performance indicators, pre-match strength. No in-round economy, health or bomb | L2 logistic regression, C = 0.3, no class balancing; chosen from 58 configurations | Temporal: older matches for training and selection; 2026_s1 window held out | Holdout ROC-AUC 0.6828; accuracy 0.6293 | None | VERIFIED-PAGE (metadata page). PDF not read. |
| 6 | Xu and Moka, "Rating Players of Counter-Strike: Global Offensive Based on Plus/Minus value" | 2024 (arXiv v1, 8 Sep 2024) | arXiv:2409.05052; https://arxiv.org/abs/2409.05052 | Match level: team point difference (ResultDiff) as response, win/loss as a logistic variant. Player-level coefficient is the output | Player attendance design matrix (5 per team, +1/-1/0). No economy, health or bomb | Ridge, elastic-net linear; elastic-net logistic; hierarchical Bayesian (Stan) | Section IV.A: random 80/20 split. Section II/III text also says "trained on 2018-2022" and tested against 2023 (inconsistent) | Pearson test of predicted vs true Plus/Minus on test set: ridge p = 0.57; Bayesian p = 0.0332; logistic p = 2.09e-5; elastic logistic p = 2.2e-16 | None | VERIFIED-PDF (Read, pp. 1-6). Data: HLTV big-event matches 2018-2023 (github patchoulixu/hltv-data). |

Notes on row 6: the Plus/Minus target is computed from the same matches used to fit the model, so the test correlation is not independent evidence (inference from the paper's own definition, Section II). Players with fewer than 50 matches are dropped. The paper does not model round-level win probability.

Notes on row 1: the abstract says "predictive probabilities of win conditions". This is a probability of an end-of-round outcome type, not a round win probability. Check the body before citing it as WP.

## Adjacent new papers (not WP, or match-level with abstract only)

| Paper | Year | ID | What it is | Numbers | Status |
|---|---|---|---|---|---|
| Parshakov, "What a Shorter Contest Really Costs: Serial Dependence, Effective Trials, and Evidence from a Rule Change in Professional Esports" | 2026-10-04 | arXiv:2610.05251; https://arxiv.org/abs/2610.05251 | Not WP. Discrimination (ability slope over residual SD) in pro CS maps. Panel of 34,453 professional matches and 72,576 maps. Sept 2023 MR15 to MR12 change | Benchmark 0.948; post-switch 0.648 (95% CI 0.52-0.81); full panel 0.901 (0.82-0.99) (abstract). In-game economy gives within-map serial dependence: intra-class correlation about 0.074 for first-half round scores; a 15-round half is worth 7.35 independent draws, a 12-round half 6.60. Full buy in 73.3% of non-pistol rounds after 2019 vs 67.4% before. | VERIFIED-PDF (Read, pp. 1-3) + VERIFIED-ABSTRACT. Remaining sections not read. |
| Frontiers in Psychology, "Predicting competitive outcomes in professional e-sports from 60-second pre-match voice acoustics" | 2026-09-07 | https://doi.org/10.3389/fpsyg.2026.1819513 | Match level, pre-match team voice audio, 89 pro CS:GO matches. Frame-level features aggregated to match score. Stratified group 5-fold CV; bootstrap CIs | Voice plus ranking, decision tree: AUC 77.3% (95% CI 65.8-86.9), accuracy 78.6%. Voice only: DT AUC 67.8%; RF 64.0%. Ranking-only models not above chance | VERIFIED-ABSTRACT (OpenAlex W7211871117). Small sample; single-team, retrospective (authors' own limits). |
| "Impact of Map Architecture on Competitive Balance in CS2" | 2025 | https://doi.org/10.20431/2454-6380.1102005 | Descriptive. CT and T win rates on six maps, 2024-2025 pro matches; entry-kill correlation with round wins | Qualitative: compact maps favour CT; multi-path maps favour T | VERIFIED-ABSTRACT (OpenAlex W4417025008). Relevant for side-prior features. |
| "Differentiating playing roles in professional CS2 by performance indicators" | 2026 | https://doi.org/10.58962/hsr.1395 | Per-round indicators, role classification (random forest 77-80% role recovery). Not WP | Macro-F1 0.78-0.82 | VERIFIED-ABSTRACT (OpenAlex W7204440738). |
| "Round Outcome Prediction in VALORANT Using Tactical Features from Video Analysis" | 2025 | CoG 2025, https://doi.org/10.1109/cog64752.2025.11114177; arXiv:2510.17199 | Valorant round outcome from minimap video (TimeSformer). Not CS. Round-level FPS | About 81% accuracy with tactical event labels (preliminary) | VERIFIED-ABSTRACT (arXiv search record). |
| "Account Consistency from Gameplay Traces: Same-Player Verification in CS2" | 2026 | arXiv:2608.24893 | Not WP. Economy/buy is one fingerprint family | ROC AUC 0.926 and 0.956 | VERIFIED-ABSTRACT. |
| CS:GO world-model papers: LeCuration (arXiv:2610.09285, 2026-10-07; qualitative, CS:GO) and WorldCast (arXiv:2610.12412, 2026-10-08; CS2 generative) | 2026 | arXiv | Not WP | None | VERIFIED-ABSTRACT. |

## Pre-2024 precedents (out of window, not in note)

- Sinap, "Makine Ogrenmesi Teknikleri ile CS:GO Raunt Sonuclarinin Tahminlenmesi", JISTA 2023 (doi 10.38016/jista.1235031). CS:GO round outcome, seven classifiers, random forest 88% accuracy. Published 2023-08-14. VERIFIED-ABSTRACT (Turkish). Excluded by date.
- Huang, Wang, Xu, "Predicting Round Result in CS:GO Using ML", ICSP 2022 (doi 10.1109/icsp54964.2022.9778597). Metadata only. Excluded by date.
- Yalciner and Kilci, "Effect of Pistol Round and First Kill on Match Outcome", Aloma 2023 (doi 10.51698/aloma.2023.41.2.21-25). Match level. Excluded by date.
- Temporal probability calibration (arXiv:2002.02644, 2020) uses a CS:GO round-sequence dataset. Snippet only, UNVERIFIED. Excluded by date but relevant to calibration.

## Already in the existing note (matched by searches; not new)

- Note #11 "Precision under fire": WebSearch summary of Xia, Salinas, Morstatter (CoG 2025) matches this entry. Body still UNVERIFIED.
- Note #7 (Valorant Seq2Seq) and note #8 (Valorant 60.61%): OpenAlex lists the 60.61% paper under David Wang (W4410426464), matching the note's authorship conflict.
- Note #23 (Peng 2024 eSports): OpenAlex W4393436108 matches. Still low-trust.
- ESTA (note #3), Xenopoulos 2020 and 2021 (notes #1, #2), and "Enabling Real-Time Prediction of In-game Deaths" (note #22) reappeared. Already in note.

## Effect sizes: team economy, team health, bomb state

- None of the six opened papers reports a quantified effect size for team economy, team health or bomb state.
- Parshakov (2026) gives economy-related structure only: the serial-dependence intra-class correlation (about 0.074) and the buy-rate change (73.3% vs 67.4%). These are not win-probability effects.
- The Plus/Minus paper (2024) has no economy, health or bomb variables.

## Methodological implications for the swarm

- Round outcomes within a map are correlated (intra-class correlation about 0.074 on first-half scores, Parshakov 2026). Evaluate round-level WP with map-grouped or match-grouped CV, and treat rounds as clustered when computing uncertainty.
- The Plus/Minus evaluation (2024) uses a target computed from the same matches it is tested on. Do not cite its p-values as out-of-sample evidence.
- The ISRITI 2025 92% accuracy needs a leakage check (are "in-game metrics" post-match?).
- The UFSCar 2026 holdout (ROC-AUC 0.683) is a realistic pre-match CS2 match baseline, not a round-level target.

## What could not be verified

- Full text of Hirota 2024, Kar 2026, Thai 2025 and ISRITI 2025 (IEEE HTTP 418, Springer IdP redirect, blank pages). Access through an institutional route or the authors' pages would be needed.
- Semantic Scholar rate-limited: no citation counts.
- Whether the Hirota "win condition" target is a round win probability in the sense used by the existing note.
- The UFSCar PDF (TCC-Gabriel-CS2.pdf) was linked but not read; the dataset size and date range are unknown.
- The Parshakov paper beyond page 3 (model equations for the discrimination estimator and the rho estimate) was not read.

## Sources (URLs as used)

- https://doi.org/10.1109/gcce62371.2024.10760945 (Hirota 2024; OpenAlex W4406265769 for the abstract)
- https://doi.org/10.1007/978-981-96-9983-4_5 (Kar 2026; OpenAlex W7160195820)
- https://doi.org/10.1109/cog64752.2025.11114397 (Thai and Scirea 2025; OpenAlex W4413321661)
- https://doi.org/10.1109/isriti68345.2025.11393205 (ISRITI 2025; OpenAlex W7130726549)
- https://repositorio.ufscar.br/items/05d81da4-11fa-4a99-a98c-2935fbead599 (UFSCar 2026, VERIFIED-PAGE)
- https://arxiv.org/abs/2409.05052 (Xu and Moka 2024, VERIFIED-PDF)
- https://arxiv.org/abs/2610.05251 (Parshakov 2026, VERIFIED-PDF pp. 1-3)
- https://doi.org/10.1109/cog64752.2025.11114177 and https://arxiv.org/abs/2510.17199 (Valorant round outcome)
- https://doi.org/10.3389/fpsyg.2026.1819513 (voice acoustics, 2026)
- https://doi.org/10.20431/2454-6380.1102005 (CS2 map architecture, 2025)
- https://doi.org/10.38016/jista.1235031 (Sinap 2023, excluded by date)
- https://doi.org/10.1109/icsp54964.2022.9778597 (Huang 2022, excluded by date)
