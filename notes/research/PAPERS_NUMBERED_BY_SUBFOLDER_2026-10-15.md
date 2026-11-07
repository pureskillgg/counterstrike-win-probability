# Papers to read, numbered in reading order, then grouped by your Zotero subfolders (2026-10-15)

Same papers as `PAPERS_DEEP_TEACHER_STUDENT_2026-10-15.md` (Haiku-only swarm; 43 of 45 candidates had their page opened and title/authors/year checked).
The subfolders are the nine that exist now under "CS2 CSDS Round Win Probability" in your Zotero (screenshot, 2026-10-15). Numbers are stable: Part 1 and Part 2 give the same
number to the same paper. Figures quoted in the older note come from a paper's body as read by a Haiku agent, so **confirm any number against the PDF before quoting it.**
Nothing is filed: Zotero cannot be written to from here without a key (see `zotero_options.md`). Already read by you, not repeated: 2011.01324, OSF u9j5g_v1, 2407.00956.

Subfolders: 1 CS2 and esports win probability · 2 Sports win probability and critiques · 3 Calibration, scoring and evaluation · 4 Boosted trees, deep learning and tabular ...
· 5 Tabular foundation models and licences · 6 Interpretable and custom architectures · 7 CS2 CSDS data, spec and tooling · 8 Inspecting rounds and uncertainty ·
9 Serving, testing and reliability.

## Part 1. One list, in the order I suggest reading

| # | Paper | Subfolder |
|---|---|---|
| 1 | Xenopoulos & Silva, *Graph Neural Networks to Predict Sports Outcomes*, arXiv 2207.14124 (2022) | 1 |
| 2 | Xenopoulos & Silva, *ESTA: An Esports Trajectory and Action Dataset*, arXiv 2209.09861 (2022) | 1 |
| 28 | 42 Analytics, *Evaluating Player Actions in Professional Counter Strike using Temporal Heterogeneous Graph Neural Networks*, conference paper ID 20251422 ("Other Sports" track; PDF, 18 pages; read in full 2026-10-15) | 1 (also 6) |
| 3 | Leathart & Polaczuk, *Temporal Probability Calibration*, arXiv 2002.02644 (2020) (also fits 3) | 1 |
| 4 | Xia, Salinas, Morstatter, *Precision Under Fire*, IEEE CoG (2025) | 1 |
| 5 | Hodge et al., *Win Prediction in Multi-Player Esports: Live Professional Match Prediction*, IEEE ToG (2019) | 1 |
| 6 | Lopez-Paz, Bottou, Schoelkopf, Vapnik, *Unifying distillation and privileged information*, arXiv 1511.03643 (2016) | 6 (closest fit) |
| 7 | Karlsson et al., *Using Time-Series Privileged Information for Provably Efficient Learning of Prediction Models*, arXiv 2110.14993 (2022) | 6 (closest fit) |
| 8 | Liu et al., *Improving Text-based Early Prediction by Distillation from Privileged Time-Series Text*, arXiv 2301.10887 (2023) | 6 (closest fit) |
| 9 | Provodin et al., *Rethinking Knowledge Transfer in Learning Using Privileged Information*, arXiv 2408.14319 (2024) | 6 (closest fit) |
| 10 | Yan et al., *Learning with privileged and sensitive information: a gradient-boosting approach*, PMC10679679 (2023) | 4 |
| 11 | Tanna et al., *Pocket Foundation Models: Distilling TFMs into CPU-Ready Gradient-Boosted Trees*, arXiv 2605.18654 (2026) | 5 |
| 12 | Tanna et al., *Distilling Tabular Foundation Models for Structured Health Data*, arXiv 2605.18702 (2026) | 5 |
| 13 | Lee, Apostolopulos, Markov, *Practical Knowledge Distillation: Using DNNs to Beat DNNs*, arXiv 2302.12360 (2023) | 4 |
| 14 | Fakoor et al., *FAST-DAD*, arXiv 2006.14284 (2020) | 4 |
| 15 | Qin et al., *GEAR*, arXiv 2608.18849 (2026) | 4 |
| 16 | Frosst & Hinton, *Distilling a Neural Network Into a Soft Decision Tree*, arXiv 1711.09784 (2017) | 6 |
| 17 | TabPFN-2.5 report, arXiv 2511.08667 (tree-distillation engine; weights carry a non-commercial licence) | 5 |
| 18 | McTavish et al. (2026), single-tree student (arXiv id not recorded in the swarm output; look up before filing) | 6 |
| 19 | Kim et al. (2025), arXiv 2508.20224: teacher calibration error predicts student accuracy | 3 |
| 20 | Yang & Song (2021), arXiv 2111.01684: calibrate the teacher first | 3 |
| 21 | Mueller, Kornblith, Hinton (2019), arXiv 1906.02629: a label-smoothed teacher is a worse teacher | 3 |
| 22 | Kim et al. (2026), arXiv 2602.12687 (calibration when distilling) | 3 |
| 23 | Vapnik & Izmailov (2015), JMLR 16: learning using privileged information (background) | 6 (closest fit) |
| 24 | Wang (2019), arXiv 1903.03694 (background and theory) | 6 (closest fit) |
| 25 | NFL player-tracking win probability, arXiv 1906.01760 | 2 |
| 26 | Football axial transformer, arXiv 2511.18730 | 2 |
| 27 | 3M esports distillation, arXiv 2406.09076 (event detection, not win probability; low priority) | 1 |

## Part 2. By your subfolders (same numbers)

**1 CS2 and esports win probability:** 1, 2, 3, 4, 5, 27, 28
**2 Sports win probability and critiques:** 25, 26
**3 Calibration, scoring and evaluation:** 19, 20, 21, 22 (and 3, cross-listed)
**4 Boosted trees, deep learning and tabular ...:** 10, 13, 14, 15
**5 Tabular foundation models and licences:** 11, 12, 17
**6 Interpretable and custom architectures:** 6, 7, 8, 9, 16, 18, 23, 24, 28 (graph architecture) (6-9, 23, 24 are teacher-student / privileged-information papers: no existing subfolder is a clean fit; a tenth "Teacher-student and privileged information" would be, if you want it)
**7 CS2 CSDS data, spec and tooling:** none from this list
**8 Inspecting rounds and uncertainty:** none from this list
**9 Serving, testing and reliability:** none from this list

The NBA recurrent / mixture-density versus random-forest paper (gaps under half a point of accuracy) is in the swarm output without an identifier (it belongs in 2): see
`PAPERS_DEEP_TEACHER_STUDENT_2026-10-15.raw.txt` before adding it.

## Why this order
1-2 are the only published deep per-second CS:GO models (our direct comparison); 3 is the calibration-over-time reference; 6-9 are the teacher-student templates (the teacher sees more
than the live service has); 11-12 are the only papers that distil into CPU-ready trees and report calibration; 19-22 matter only if we build a teacher. Our own paired result:
a deep teacher on the same aggregate inputs is 0.0012 to 0.0020 worse than XGBoost and distilled students are level at best (`exp_deep_distill*.json`); the untested route is per-player
positions, as in papers 1 and 2.

## Notes on paper 28 (read in full, 2026-10-15)
* **Task and data:** per-snapshot P(defenders win) for CS2 rounds; 114 S-tier professional matches on **Inferno only** (Oct 2023 to Sep 2024, HLTV demos, awpy, 4 snapshots per second), about 90,000 windows of 20 snapshots (5 s, 2.5 s offset).
* **Model:** a heterogeneous graph per snapshot: a hand-built **map graph** (nodes of about smoke size plus contact points, edges = walkable paths, built with a former professional player) and **player nodes** (position, velocity, view direction, weapons, health, in-match stats and previous-year player ratings: more than 160 features) with player-to-map and self-loop edges; graph-level features (time, bomb, team totals). Spatial part: heterogeneous GAT layers (2) then map-to-map GAT layers (3); temporal part: two GraphGRU layers over the 20 snapshots; dense head.
* **Reported results (their Table 3.1, one split, no intervals, split protocol not stated):** log loss 0.4587, AUC 0.8549, F1 0.7463, accuracy above 76%, ECE 0.0307; logistic regression 0.4688 / 0.7564; XGBoost 0.5353 / 0.7913 (worse than their logistic regression, so that baseline looks untuned); constant 0.693.
* **Explanations:** Shapley values of ridge surrogates fitted to feature changes between snapshots, split over actions: win-probability-added style attributions.
* **Caveats for us:** not comparable to our numbers (pro matches, one map, 5 s windows, player-rating features that identify who is playing, overlapping windows with an unstated split); the gain over their logistic regression is 0.0101 log loss, over a weak XGBoost 0.077. It is the clearest published case for a graph plus temporal model on CS2, with positions as the main new information.
