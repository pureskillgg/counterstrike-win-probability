# Deep and teacher-student methods for round win probability: what exists (2026-10-15)

Found by a Haiku-only search swarm (6 angles, 45 candidates) and a second Haiku pass that opened each paper's page (43 verified, 2 dropped). Full raw output:
`PAPERS_DEEP_TEACHER_STUDENT_2026-10-15.raw.txt`. "Verified" means a page was opened and title/authors/year match; several numbers below come from a paper's body as read
by a Haiku agent, not from the abstract: **confirm any figure you quote against the PDF.** Already read by you (not repeated): 2011.01324, OSF u9j5g_v1, 2407.00956.

## Short answer
* **Deep models for per-second CS round win probability: yes, published.** Xenopoulos & Silva (2207.14124) and ESTA (2209.09861), both on pro CS:GO.
* **Teacher-student for CS/esports win probability: none found.** Closest: privileged-information distillation in other fields, and distilling neural / foundation models into GBDT.
* **Our own result** (`exp_deep_distill.json`, same held-out matches, paired): an MLP teacher on the same 59 aggregate inputs is 0.0020 worse than XGBoost [0.0017, 0.0024];
  a student XGBoost trained on the teacher's out-of-fold probabilities is 0.0002 to 0.0020 worse than the hard-label XGBoost (best at 75% hard label). The inputs are low-dimensional
  aggregates, where trees are hard to beat. The papers' deep gains use **per-player positions**, which our 1 Hz table does not carry: that is the untested route.

## Read first (closest to our task)
| Paper | Why |
|---|---|
| Xenopoulos & Silva, *Graph Neural Networks to Predict Sports Outcomes* (arXiv 2207.14124, 2022) | Per-second CS:GO P(CT wins); each living player a node (position, HP, armor, equipment, grenades, bombsite distances), GAT branch + state branch; 14.3 M states, 70/10/20 split, one model per map. Reported log loss 0.4276 vs 0.4351 for the state-only model (-0.0075). The nearest comparison to our model. |
| Xenopoulos & Silva, *ESTA: An Esports Trajectory and Action Dataset* (2209.09861, 2022) | Benchmarks LightGBM / XGBoost / MLP on vector states and Deep Sets / Set Transformer on set states, one random state per round, 7 maps. Reported: MLP lowest log loss on every map (e.g. Dust2 0.415 vs XGBoost 0.437). Not per-second, so not directly comparable to ours. Public data. |
| Leathart & Polaczuk, *Temporal Probability Calibration* (2002.02644, 2020) | GRU on a 12,362-game CS:GO corpus; calibrating a win probability over round / score index: NLL 0.3830 to 0.3793, ECE 0.0316 to 0.0235. Check whether it is the same corpus as ours. |
| Hodge et al., *Win Prediction in Multi-Player Esports: Live Professional Match Prediction* (IEEE ToG, 2019) | Generic live-prediction precedent (Dota 2); not deep. |
| Xia, Salinas, Morstatter, *Precision Under Fire* (IEEE CoG 2025) | Round-result prediction in CS:GO from health, position and economy; XGBoost best of the models tried. Supports the model family. |

## Teacher-student and privileged information (the natural fit: the teacher sees more than the live service has)
| Paper | Why |
|---|---|
| Lopez-Paz, Bottou, Schoelkopf, Vapnik, *Unifying distillation and privileged information* (1511.03643, 2016) | The template: a teacher with richer inputs, distilled into a student that sees only the deployable inputs. |
| Karlsson et al., *Using Time-Series Privileged Information for Provably Efficient Learning of Prediction Models* (2110.14993, 2022) | Training-only information recorded between the prediction time and the outcome; the deployed model sees baseline features only. Closest structure to "teacher sees later play, student sees the live state". Linear theory. |
| Liu et al., *Improving Text-based Early Prediction by Distillation from Privileged Time-Series Text* (2301.10887, 2023) | Same temporal design (teacher sees post-prediction data), applied to text. |
| Yan et al., *Learning with privileged and sensitive information: a gradient-boosting approach* (PMC10679679, 2023) | Privileged features inside a boosted-tree framework (not distillation): maps onto an XGBoost student. |
| Provodin et al., *Rethinking Knowledge Transfer in Learning Using Privileged Information* (2408.14319, 2024) | Cautionary: many claimed LUPI gains come from artefacts or model-design changes. Compare against a strong student trained alone (we do). |
| Vapnik & Izmailov 2015 (JMLR 16); Wang 2019 (1903.03694) | Background and theory. |

## Distilling neural / foundation models into trees (what a student XGBoost keeps)
| Paper | Why |
|---|---|
| Tanna et al., *Pocket Foundation Models: Distilling TFMs into CPU-Ready Gradient-Boosted Trees* (2605.18654, 2026) | TabICLv2 teacher, XGBoost/CatBoost students on 153 datasets: about 96.5% of teacher AUC at ~1.9 ms per row on CPU. Finds label leakage in in-context teachers (soft targets collapse to one-hot); stratified out-of-fold teacher labels fix it (we use match-grouped out-of-fold). |
| Tanna et al., *Distilling Tabular Foundation Models for Structured Health Data* (2605.18702, 2026) | Tree students keep 98.8 to 100.7% of teacher AUC; the distilled LightGBM had lower ECE and Brier than the default one. Only paper found that reports calibration of a distilled tree. |
| Lee, Apostolopulos, Markov, *Practical Knowledge Distillation: Using DNNs to Beat DNNs* (2302.12360, 2023) | DNN ensemble to a single GBDT in a production real-time system. |
| Qin et al., *GEAR* (2608.18849, 2026); Fakoor et al., *FAST-DAD* (2006.14284, 2020) | Augmented distillation: the teacher labels extra generated states. Tree students gain less than MLP students. |
| Frosst & Hinton 2017 (1711.09784); TabPFN-2.5 report (2511.08667, tree distillation engine, closed); McTavish et al. 2026 (single-tree student) | Background; TabPFN-2.5 weights carry a non-commercial licence. |

## Calibration when distilling
Kim et al. 2025 (2508.20224): teacher calibration error predicts student accuracy; Yang & Song 2021 (2111.01684): calibrate the teacher first; Mueller, Kornblith, Hinton 2019
(1906.02629): a label-smoothed teacher is a worse teacher; Kim et al. 2026 (2602.12687). None is tabular or sports.

## Gaps the swarm reported
No CS2 teacher-student paper; no deep per-second CS2 round win probability paper for 2023-2026 (a Transformer checkpoint "CS-NET v4" exists on Hugging Face, no paper or metrics);
one esports distillation paper (3M, 2406.09076) is event detection. Sports: deep in-game models exist (NBA recurrent/MDN vs random forest on three inputs: gaps under half a point of accuracy;
NFL tracking 1906.01760; football axial transformer 2511.18730) but none reports calibration against trees.

## What this suggests we try next (not done)
1. A per-player (set / graph) model that uses positions, as in 2207.14124 and 2209.09861: needs 1 Hz positions for the training matches (raw `player_vector`).
2. Privileged-information teacher: teacher sees positions and the following seconds; student XGBoost sees only the loadout snapshot (the live service's inputs); soft-label mix and out-of-fold teachers as in `exp_deep_distill.py`.
3. Report calibration (ECE with intervals), not only accuracy, for every teacher and student.
