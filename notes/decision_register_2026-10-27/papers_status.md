# Papers: what was read, and what was tested (27 Oct 2026)

Reading list: `notes/research/PAPERS_NUMBERED_BY_SUBFOLDER_2026-10-15.md` (27 numbered papers). Status below comes from our notes and pins. "Tested" means a pin or a result file reports a run that uses the paper's method. "Read" means the notes say the paper was read, not just listed.

## The 27 numbered papers

| # | Paper (ID) | Read status | Tested? | Where |
|---|---|---|---|---|
| 1 | Xenopoulos and Silva 2022, GNN to predict sports outcomes (arXiv 2207.14124) | Read; headline claims refuted | Yes: attention idea (T3). Fails | 32f55bbe, 0242913c |
| 2 | Xenopoulos and Silva 2022, ESTA (arXiv 2209.09861) | Read (Table 1, split) | Yes: plain MLP, fails. Set-encoder idea: fails once distances are in | b5a0071d, 27589820, 5ba17087 |
| 3 | Leathart and Polaczuk 2020, temporal probability calibration (arXiv 2002.02644) | Listed, not read in full | No | none |
| 4 | Xia, Salinas, Morstatter 2025, Precision Under Fire | Listed; notes say not read in full | No | none |
| 5 | Hodge et al. 2019, live professional match prediction | Listed | No | none |
| 6 | Lopez-Paz et al. 2016, distillation and privileged information (arXiv 1511.03643) | Listed | Related only: teacher-student runs, no pin | exp_deep_distill*.json (unpinned) |
| 7 | Karlsson et al. 2022, privileged time series (arXiv 2110.14993) | Listed | Related only, as for 6 | exp_deep_distill*.json (unpinned) |
| 8 | Liu et al. 2023, privileged text (arXiv 2301.10887) | Listed | No | none |
| 9 | Provodin et al. 2024, privileged information (arXiv 2408.14319) | Listed | No | none |
| 10 | Yan et al. 2023, privileged gradient boosting (PMC10679679) | Listed | No | none |
| 11 | Tanna et al. 2026, Pocket Foundation Models (arXiv 2605.18654) | Listed | No | none |
| 12 | Tanna et al. 2026, distilling tabular foundation models (arXiv 2605.18702) | Listed | No | none |
| 13 | Lee, Apostolopulos, Markov 2023, practical distillation (arXiv 2302.12360) | Listed | No | none |
| 14 | Fakoor et al. 2020, FAST-DAD (arXiv 2006.14284) | Listed | No | none |
| 15 | Qin et al. 2026, GEAR (arXiv 2608.18849) | Listed | No | none |
| 16 | Frosst and Hinton 2017, soft decision tree (arXiv 1711.09784) | Listed | No | none |
| 17 | TabPFN-2.5 report (arXiv 2511.08667) | Listed | Partly: TabPFN v2 was tested zero-shot and fine-tuned. The 2.5 report itself was not tested | a79d4206, exp_foundation.json, exp_finetune*.json (unpinned) |
| 18 | McTavish et al. 2026, single-tree student | Listed; arXiv ID not recorded | No | none |
| 19 | Kim et al. 2025, teacher calibration (arXiv 2508.20224) | Listed | No | none |
| 20 | Yang and Song 2021, calibrate the teacher (arXiv 2111.01684) | Listed | No | none |
| 21 | Mueller, Kornblith, Hinton 2019, label smoothing (arXiv 1906.02629) | Listed | No | none |
| 22 | Kim et al. 2026, calibration when distilling (arXiv 2602.12687) | Listed | No | none |
| 23 | Vapnik and Izmailov 2015, learning using privileged information (JMLR 16) | Background | No | none |
| 24 | Wang 2019, background and theory (arXiv 1903.03694) | Background | No | none |
| 25 | NFL player tracking win probability (arXiv 1906.01760) | Listed | No | none |
| 26 | Football axial transformer (arXiv 2511.18730) | Listed | No | none |
| 27 | 3M esports distillation (arXiv 2406.09076) | Listed; event detection, low priority | No | none |
| 28 | Szmida and Toka, THGNN, "Evaluating Player Actions in Professional Counter Strike using Temporal Heterogeneous Graph Neural Networks" (42 Analytics, conference ID 20251422) | Read in full (18-page PDF) | Yes: THGNN-lite, not adopted | 4edc74c3, 6787d3b7. 2025 by paper ID, 2026 by the listing (open) |

Totals from the numbered list: 4 tested directly (1, 2, 17 partly, 28), 2 related only (6, 7), 21 not tested.

## Sources used that are not on the numbered list

| Source | Read status | Tested? | Where |
|---|---|---|---|
| Xenopoulos, Doraiswamy, Silva 2020, Valuing Player Actions in CS:GO (arXiv 2011.01324) | Read. Team bombsite distance (verified) and a navigation-mesh graph distance (Section III) | Yes, as the distance features | ee61ba3a, 428961a1, d71bc655, 7b4f7a87 |
| Xenopoulos, Coelho, Silva 2021, team economic decisions (arXiv 2109.12990) | Read | No | none |
| Lundberg, Erion, Lee 2018, TreeSHAP (arXiv 1802.03888) | Confirmed today | Used for explanations (not a test) | slide 6; winprob.py |
| TabICLv2, Qu et al. 2026 (arXiv 2602.11139) | Confirmed today | Yes, zero-shot | a79d4206, exp_foundation.json |
| Bai et al. 2018, temporal convolution (arXiv 1803.01271) | Abstract only | No | none |
| Huang, Lu, Pipping, Wyner 2026, defuse race working draft | Not found online. Our notes cite a working draft v13.1 | Used: the contested-round rule and the 250 units per second speed | 28cfd049, 370f5ec6. Unconfirmed |
| Chahrour 2023, word2vec position embeddings | Not found as a paper. Repo kamalchah/CounterStrikeWinProb exists | Yes: a replication | c95b21e0, 5ba17087, ab0fa0c8, f748c56f, a70d2a77, 3aed07a1. Author name unconfirmed |
| Henry1145 repo, CS2_Win_Probability_Model (de_inferno, no licence) | Read for method; code not copied | No: comparator only | none |
| Mikolov et al., word2vec (gensim) | Method | Yes, in the Chahrour replication | as above |
