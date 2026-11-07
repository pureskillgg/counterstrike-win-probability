# Win-probability features and methods: CS and esports papers (Topic B)

Compiled 2026-10-13 for Meridian project fa6f6e09-513d-4788-9db5-43123df4ce0f.
Starting list: notes\research\model_choice_literature.md (25 papers). BibTeX: notes\research\winprob_features_methods.bib.

## Status legend

- **V** = verified in the PDF text or rendered pages (page numbers given, URL in the table header).
- **V~** = verified, but the value was read off a rendered table or bar chart, so treat it as approximate.
- **U** = unverified: search snippet, a WebFetch summary, or the earlier notes file. Not opened in this run.

## Method and gaps

- WebFetch returned compressed PDF bytes for every arXiv PDF. The binaries were saved locally and read page by page with the Read tool.
- Six papers were read in full: Xenopoulos 2020 (pp. 1-8), Xenopoulos 2021 (pp. 1-6), ESTA (pp. 1-12), Choi 2023 (pp. 1-6), Honor of Kings (pp. 1-9), LoL real-time 2023 (pp. 1-8).
- The 2025 CoG paper (Xia, Salinas, Morstatter, "Precision under fire") is the most CS2-relevant item, but no full text was reachable. Only the ISI listing page was fetched (U). It is listed below, not counted among the six.
- Not covered by this task: how companies (betting, analytics, broadcast) build their win-probability models. No company source was read.

---

## Paper 1. Xenopoulos, Doraiswamy, Silva (2020). Valuing Player Actions in CS:GO

URL: https://arxiv.org/abs/2011.01324 (PDF read: https://arxiv.org/pdf/2011.01324). IEEE 2020 per page footer; venue name not on the pages read.

| Item | Finding | Status |
|---|---|---|
| Feature set | Section III-C lists 8 attributes: Map; Ticks Since Start; Equipment Value (round start, per team); Players Remaining; HP Remaining; Bomb Planted; Bomb Plant Site; Team Bombsite Distance (min player distance to each site, per side). Fig. 7 shows the 14 model inputs after per-team expansion: T and CT equipment value, HP remaining, players remaining, distance to A and to B; plus Bombsite, Bomb planted, Ticks since start, Map. | V p.3, p.7 |
| Model class | Logistic regression (sklearn, no regularisation, SAGA), CatBoost, XGBoost (hist, 100 estimators); map-average baseline. Grid search over XGBoost max_depth {6,8,10,12} and min_child_weight {1,3,5,7}; CatBoost depth and l2_leaf_reg on grids of the same shape. | V p.6 (Table I) |
| Label | Y_i = 1 if CT wins round i, else 0. Each game state G_{i,t} is one sample. | V p.5 |
| Sample unit | A state is recorded whenever a footstep, damage or bomb event occurs. Samples are event-driven, not per tick. | V p.6 |
| Split | Chronological. Train: 55M states from matches 23 Oct 2016 to 31 May 2019. Test: 18M states from 1 Jun to 22 Dec 2019. 4,682 LAN matches from HLTV. Tuning used a 2:1 train/validation split; how validation was drawn is not stated. | V pp.5-6 |
| Metrics (test, Table II) | LR: log loss 0.5539, Brier 0.1912, AUC 0.7743. CatBoost: 0.5443, 0.1875, 0.7851. XGBoost: 0.5353, 0.1842, 0.7913. Map average: 0.6917, 0.2493, 0.5303. Calibration: 100 equal-width bins, Fig. 5 (XGBoost closest to the diagonal). | V p.6 |
| Feature importance / SHAP | XGBoost total gain, normalised to sum 100 (Fig. 7, p.7). Approximate values: T equipment value about 16; CT equipment value about 14; CT HP remaining about 11; T HP remaining about 11; Map about 10; T remaining about 9; CT remaining about 8; the four distance-to-site features 3-4 each; Ticks since start about 3; Bomb planted about 0. Text: equipment value is the lead determinant, HP remaining next, Map ranks fifth. No SHAP. | V~ p.7 |
| Time in round | One feature (Ticks Since Start) in a per-state classifier. No sequence model. Performance by round time (Fig. 6, p.7): log loss falls from about 0.6 at round start to about 0.2 near 110 s; AUC and accuracy rise. | V~ p.7 |

## Paper 2. Xenopoulos, Coelho, Silva (2021). Optimal Team Economic Decisions in CS

URL: https://arxiv.org/abs/2109.12990 (PDF read: https://arxiv.org/pdf/2109.12990). Accepted to IJCAI-21 AISA workshop (page 1 footer).

| Item | Finding | Status |
|---|---|---|
| Feature set | Round-start vector R_{g,i} (p.3): each side's current score; score difference; each side's starting equipment value and money; both sides' buy decisions in round i. Map: per-map models, or one-hot map feature (pp.3-4). The LR baseline uses team scores only (Table 2 caption, p.4). | V pp.3-4 |
| Model class | LR (baseline); XGBoost tuned with Hyperopt (ranges in Table 3, p.4: learning rate 0.05-0.31, max depth 3-10, min child weight 1-8, colsample 0.3-0.8; early stopping patience 10); 2-layer NN (dense 16-128, dropout 0.1-0.6, lr 1e-3 to 1e-5, random search). | V p.4 |
| Label | Game-level outcome Y_g with 3 values: CT game win, T game win, draw. Predicts the game from the round-start state, not the round. Multiclass log loss (Eq. 1, p.3). | V p.3 |
| Split | Chronological. Train: Apr-Sep 2020 (3,308 demofiles). Validation: Oct-Nov 2020 (1,108). Test: Dec 2020 to Apr 2021 (2,122). 6,538 HLTV demofiles in total. | V p.4 |
| Metrics (Table 2) | Weighted log loss: LR 0.793; XGBoost 0.739; NN 0.736; one-hot-map XGBoost 0.730; one-hot-map NN 0.732. The paper calls the XGBoost-NN gap minimal. | V p.4 |
| Feature importance / SHAP | None reported in pp.1-6. | V pp.1-6 |
| Time in round | No in-round time. Progress is only implicit through the score. | V p.3 |

Note: the LR baseline sees fewer features than the other models, so the 0.793 vs 0.739 gap mixes model class with feature set.

## Paper 3. Xenopoulos and Silva (2022). ESTA: An Esports Trajectory and Action Dataset

URL: https://arxiv.org/abs/2209.09861 (PDF read: https://arxiv.org/pdf/2209.09861). Preprint, under review.

| Item | Finding | Status |
|---|---|---|
| Feature set | Vector state (p.7): time since last game phase change; active fires and smokes; bomb planted (A, B, None); defuse kits on alive players; starting equipment value per side; alive players per side; current equipment value; starting equipment; HP; armour; helmets; grenades remaining; players in bomb zone; flag for bomb in a T player's inventory. Set state (p.8): per-player location, velocity, view direction, HP, equipment, flags (alive, blinded, in bomb zone), plus the global features. | V pp.7-8 |
| Model class | Vector: XGBoost, LightGBM, MLP (sklearn defaults, features scaled to [0,1]). Set: Deep Sets, Set Transformer (Adam lr 1e-3, batch 32, hidden 128, 100 epochs, 10 early-stopping rounds). Boosted trees use package defaults plus 10 early-stopping rounds (p.8). | V p.8 |
| Label | Y_i = 1 if CT wins the round that the sampled state belongs to. | V p.7 |
| Sample unit | For each parsed demo, one game state per round is sampled at random (p.8). Frames are recorded at 2 per second, about 188 per round; average round 94 s (p.7). | V pp.7-8 |
| Split | Random 70/10/20 over the sampled states. Not grouped by match: states from one match can fall in train and test. Maps modelled separately (7 maps x 5 models = 35). Demos: 1,558 LAN, Jan 2021-May 2022 (p.5). | V pp.5, 8 |
| Metrics (Table 1, p.9) | Log loss (LL) and ECE with 10 bins; mean and SE over 10 runs. MLP has the lowest LL on all seven maps (e.g. Dust2 MLP 0.415 vs XGBoost 0.437, LightGBM 0.433; Inferno MLP 0.442 vs XGBoost 0.453). Best ECE differs by map. Text: no model is best across all maps and metrics. | V~ p.9 |
| Feature importance / SHAP | No table. Text (p.9): the features with highest importance in the gradient-boosted models include each team's HP and whether the bomb is planted. No ablation. | V p.9 (qualitative) |
| Time in round | Time since last phase change, not clock time. No sequence model; one sampled state per round. | V p.7 |

## Paper 4. Choi, Kim, Lee (2023). Rethinking Evaluation Metric for Probability Estimation Models Using Esports Data (LoL)

URL: https://arxiv.org/abs/2309.06248 (PDF read: https://arxiv.org/pdf/2309.06248). DOI 10.1109/trustcom60117.2023.00374 per OpenAlex (U, from earlier notes).

| Item | Finding | Status |
|---|---|---|
| Feature set | 14 indicators (pp.4-5): gold difference for each role player (5); experience difference for each role player (5); dragons killed per team (2); towers destroyed per team (2). | V pp.4-5 |
| Model class | Logistic regression only, in pp.1-6 (Table II). No other model is named. | V p.5 |
| Label | Match outcome at each time point (binary). | V p.5 |
| Time in round | Not a round study. Time handled by a separate dataset and model per time point: 5, 10, 15 min. | V p.5 |
| Split | Match-level. Per time point: 100,000 matches, 60,000 train and 40,000 test. | V p.5 |
| Metrics (LR, Table II) | 5 min: accuracy 65.56%, Brier 0.2159, ECE 0.0058, Balance 0.0043. 10 min: 73.55%, 0.1755, 0.0068, -0.0016. 15 min: 79.84%, 0.1385, 0.0078, -0.0039. Optimal models on simulated Beta(0.5,0.5), Beta(1,1), Beta(2,2) reach 68.9-81.9% accuracy, so accuracy cannot measure probability quality (p.5). ECE uses M = 10 bins (p.3). | V p.5 |
| Feature importance / SHAP | None in pp.1-6. | V pp.1-6 |

Relevance: the paper argues that accuracy and Brier cannot judge calibration in esports. It is a metric paper, not a CS feature study.

## Paper 5. Yang et al. (2021). Interpretable Real-Time Win Prediction for Honor of Kings (MOBA)

URL: https://arxiv.org/abs/2008.06313 (PDF read: https://arxiv.org/pdf/2008.06313). arXiv v3, 16 Apr 2021. Later journal version per OpenAlex (U).

| Item | Finding | Status |
|---|---|---|
| Feature set | 184,362 games (June 2019, top 1% players). Frames every 30 s: 5,253,661 frames, 96 features per frame, 11 feature types. Six groups: Gold (difference), Kill (difference), Tower (difference), Wild Resource, Soldier (distance to the rival crystal), Heroes (hero IDs). Game time is also an input (pp.3-5). | V pp.3-5 |
| Model class | TSSTN, two-stage. Spatial stage: LR for Gold, Kill, Tower; feed-forward nets for Wild Resource, Soldier, Heroes. Each gives a win-score in [-1,1]. Temporal stage: six softmax weights per time point; F^t = sum_i w_i^t s_i (Eqs. 1-3, p.5). Contribution = w_i^t x s_i (p.5). Trained jointly with cross-entropy (p.6). | V pp.4-6 |
| Label | Win side of each frame (binary). | V p.4 |
| Split | Random at game level: 10,000 test games; the other 174,362 split 90/10 train/validation (p.6). | V p.6 |
| Metrics (Table II, accuracy %) | At 0 / 5 / 10 / 15 / 20 min. Heuristic (gold difference): 50.0 / 69.2 / 78.4 / 73.0 / 63.0. FC NN: 54.7 / 69.3 / 79.3 / 75.0 / 70.0. TSSTN: 54.6 / 69.2 / 78.5 / 73.4 / 67.8. LR: 53.9 / 63.8 / 78.2 / 70.5 / 56.6. LSTM: 54.3 / 69.4 / 78.4 / 76.3 / 66.5. Text (p.7): TSSTN is a little below FC and LSTM because interpretability costs some coupling between groups. | V~ p.6-7 |
| Human comparison | Top 0.1% experts on 200 test games per time point (Fig. 4, p.7): experts are below both models at 0, 17.5 and 20 min. | V~ p.7 |
| Feature importance / SHAP | Model-internal contributions (weight x win-score), not post hoc. Example at 15 min: Gold contributes most (0.205), then Soldier (0.105) (p.5). Fig. 6 (p.8): Gold importance is high from about 2 min and falls after about 12 min; Heroes is about zero after 0 min (p.7). No faithfulness test or user study in pp.1-9. | V~ pp.5-8 |
| Time in round | Game minutes as an input to each spatial model; time-variant softmax weights in the temporal stage. | V pp.5-6 |

## Paper 6. Junior and Campelo (2023). League of Legends: Real-Time Result Prediction

URL: https://arxiv.org/abs/2309.02449 (PDF read: https://arxiv.org/pdf/2309.02449). Venue not stated on the pages read.

| Item | Finding | Status |
|---|---|---|
| Feature set | About 40 variables (Table VI, p.5) for each team: champion kills, first blood, dragon kills by type, herald, baron, tower and inhibitor kills, total gold, minions and jungle minions killed, average player level. Feature selection by Pearson correlation with blue-team victory (Tables II-V, pp.4-5). | V pp.4-5 |
| Model class | LR, random forest, naive Bayes, sklearn gradient boosting, XGBoost, LightGBM, NN, bagging (p.5-6). Random search on RF and LightGBM (Table VIII, p.7). | V pp.5-7 |
| Label | Blue-team victory (binary). | V p.5 |
| Time in round | Not a round study. Snapshots at 20, 40, 60, 80% of the match's total duration ("percentage elapsed time", p.4). The total duration is not known during a live match, so the live claim is not tested. | V p.4 |
| Split | 64,556 ranked matches, Iron to Diamond, 12 Jan to 18 May 2023 (p.4). Matches under 5 min dropped (p.4). Evaluation by cross-validation; number of folds and grouping not stated in pp.4-7. | V pp.4-7 |
| Metrics | Accuracy, precision, recall, F1, AUC-ROC (Table VII, p.7). Random-search accuracy (Table VIII, V~): LightGBM 0.619 / 0.704 / 0.779 / 0.855 at 20 / 40 / 60 / 80%; RF 0.618 / 0.703 / 0.780 / 0.854. Abstract: LightGBM 81.62% at 60-80%. Text (p.7): LR best at 20%; LightGBM best at 40, 60, 80%. | V p.1, V~ pp.7-8 |
| Feature importance / SHAP | Random-forest impurity importance (Figs. 2-5, p.6). Top feature: redTotalGold at 20% and 60%; blueTotalGold at 40%; blueChampionKill at 80%. Importances come from the RF, not from LightGBM (the best model). No SHAP. | V p.6 |

---

## Paper 7 (not read in full). Xia, Salinas, Morstatter (2025). Precision under fire: analysis and predictive modeling in CS:GO

URL: https://www.isi.edu/results/publications/31260/precision-under-fire-analysis-and-predictive-modeling-in-counter-strike-global-offensive (ISI listing, fetched via WebFetch). IEEE Conference on Games (CoG) 2025, pp. 1-8.

- U: pro Dust 2 matches from 2022. Two tasks: round outcome and player death. Features named as health, spatial position and economy. Models: LR, decision tree, XGBoost; XGBoost best. The listing's "View Paper" link points to the homepage, and a WebSearch found no PDF. Full text is needed before any claim from it is used.

---

## What matters most, with the evidence

| Feature group | Evidence (source, page, value) | Consistent across sources? | Buildable from event-only tomes? |
|---|---|---|---|
| Economy: team equipment value (CS); gold (MOBA analogue) | CS: Xenopoulos 2020, Fig. 7 p.7 (T and CT equipment value rank 1 and 2, about 16 and 14 of 100 gain). CS: 2021 uses equipment and money but reports no importance. HoK: Gold is the most accurate single group late in the game, and the largest contributor in the 15-min example (p.5, p.7). LoL: total gold top in RF at 20%, 40%, 60% (p.6). | Yes for gold-type economy across games. In CS, only one source gives importance numbers. | No (equipment, money, buy type). |
| Health and alive count | CS: HP remaining ranks 3-4 in Xenopoulos 2020 (about 11 each); players remaining ranks 6-7 (about 8-9 each). ESTA names team HP among the top GBM features (p.9, no numbers). LoL: kill counts top at 80% (p.6). | Yes in CS, but one quantitative source and one qualitative. | Alive count: likely yes, from death events. HP: only if the damage events carry health. Verify. |
| Map | Xenopoulos 2020: Map ranks 5th (about 10). Map alone gives LL 0.692 vs 0.535 for the full model (Table II). Xenopoulos 2021: one-hot map 0.730 vs 0.739 without map (small gain). | Moderate. Helps, but small once team state is in. | Yes (map name from demo header). |
| Bomb state (planted, site) | Xenopoulos 2020: Bomb planted about 0 gain; distances to sites 3-4 each. ESTA p.9: bomb planted named as a top GBM feature (qualitative). | No. The two CS sources disagree. | Yes (bomb events), if the tomes carry them. |
| Position and distance | Xenopoulos 2020: T and CT distance-to-A/B features rank 8-12 (about 3-4 each, about 15 combined). ESTA set models use positions but do not beat vector models on LL (p.9-10). HoK: Soldier (distance to crystal) grows in importance late (p.7). | Weak. Not established. | No (no per-tick positions). |
| Time in round | Xenopoulos 2020: Ticks since start about 3 gain, but log loss falls from about 0.6 to 0.2 across the round (Fig. 6). HoK: time-varying weights (p.8). LoL: accuracy rises with elapsed fraction (p.7), though confounded by the known final length. | Yes that performance depends on time. Time as a single feature gets low gain. | Yes (ticks, phase events). |
| Kills and objectives (in CS: death and kill events) | No CS paper here uses kill events as WP inputs; Xenopoulos 2020 uses players remaining instead. LoL: champion kills and dragons matter (Tables II-V, Figs 2-5). HoK: Kill group contributes less than Gold. | Not tested in CS within these papers. | Yes (death events). |

Reading of the table: economy and HP or alive counts are the groups with the most support in CS. Position, bomb state and map are weaker or conflicting.

## Features we cannot build from event-only tomes

Using the Xenopoulos 2020 gain shares (read off Fig. 7, approximate), with the tomes as stated (no per-tick positions, money or equipment):

- **Not buildable (about 45 of 100 of the XGBoost gain):** T and CT equipment value (about 30 combined; the two largest features); T and CT distance to bombsite A and B (about 15 combined). Also not in the tomes: money, loss bonus, buy type (Xenopoulos 2021 inputs); defuse kits, armour, helmets, grenades, smoke and fire counts, positions and velocity (ESTA inputs).
- **Conditional (about 22 of 100):** T and CT HP remaining. Buildable only if damage events carry the victim's health. Verify against the tome schema.
- **Buildable from event-level data (about 33 of 100):** map name; players remaining per side (from death events); bomb planted and bomb site (from bomb events); ticks since round start.

Implication from the literature: the event-only model would lack the two strongest CS features in the only CS paper that reports importance.

## Model-choice evidence (what is established on XGBoost)

- Xenopoulos 2020 (Table II, p.6): XGBoost beats LR on all three metrics, but LR is close (LL 0.554 vs 0.535; Brier 0.191 vs 0.184). The authors chose boosted ensembles for their calibration (p.6).
- Xenopoulos 2021 (Table 2, p.4): LR 0.793 vs XGBoost 0.739. The LR baseline uses fewer features, so this gap is not a clean model-class comparison.
- ESTA (Table 1, p.9): boosted trees were chosen for ease of use, strong performance and built-in importance (p.8). Yet the untuned MLP had the lowest LL on all seven maps. Boosted trees ran at defaults, so this is not a tuned comparison.
- LoL 2023 (p.7): LR is best at 20% of the match; LightGBM is best from 40% on.
- Choi 2023 (p.5): the paper uses LR and argues that the metric matters more than the model choice.

Assessment (our reading, not a claim by any source): no source shows a clear winner. Boosted trees are the common default for tabular event data, with calibration and importance output as the stated reasons. A logistic regression baseline is close on small feature sets, and the early-game results in the LoL paper favour it.

## Caveats

- ESTA's split is random over states, not grouped by match, so its log-loss values may be optimistic. The size of the effect is not measured in the paper (U).
- Xenopoulos 2020: the validation split method is not stated (p.6).
- LoL 2023: percentage elapsed time uses the match's final length; RF importances come from a model other than the best one; correlation selection may run before cross-validation (not stated).
- HoK: the interpretability claim is by construction; no faithfulness or user test was found in pp.1-9.
- Every value read from a figure or rendered table is approximate (V~).

## Open items

- Full text of Xia, Salinas, Morstatter (CoG 2025). This is the most relevant CS2-era source and is still missing.
- ESTA results on pp.10-12 beyond Table 1 (Fig. 4 only checked).
- Xenopoulos 2020 bold selected hyperparameters in Table I (read from image; not confirmed).
