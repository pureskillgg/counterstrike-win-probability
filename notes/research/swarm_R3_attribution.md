# R3: Explainability and attribution of win-probability change to individual players

Date: 2026-10-13. Swarm task R3. Working folder: G:\pureskill\csds-cs2.
Scripts run (pixi, read-only on G:): scripts\swarm\r3_schema_probe.py, r3_id_probe.py, r3_steam_probe.py, r3_steam_shape_probe.py.
Nothing was written under E:. No existing report, bakeoff.py or models\ file was edited.

## Evidence key

- VERIFIED (PDF): the PDF was saved by the WebFetch tool and read page by page with the Read tool (page numbers given).
- VERIFIED (fetch): the page was opened with WebFetch. The text is a model summary of the page, so formulas that the page does not print are marked as paraphrase.
- VERIFIED (probe): code was run on the G: tomes (script named).
- UNVERIFIED (abstract): only the abstract returned by paper_search / OpenAlex was read. The full text was not opened.
- UNVERIFIED (analysis): my reasoning, with no source.

## 1. Bottom line

1. The win-probability credit methods in the literature fall into three families: (a) per-event change in win probability (WPA, Xenopoulos 2020 for CS:GO), (b) probability-based action values (VAEP), and (c) Shapley allocation over a coalition game (sports lineup and feature-level work). Shapley is the only one of the three that is explicitly fair across players in a game. It is also the only one that needs a counterfactual value function v(S) for coalitions that were never observed.
2. Per-match attribution does not need cross-match player IDs. Every method above can run within one match if each event carries a stable slot ID. On the 2026 tome, slot IDs are stable within a match: raw-to-fixed mapping is one-to-one in 1,750 of 1,751 matches (VERIFIED, probe).
3. Cross-match attribution (player ratings, season MVP, pooling lineups across seasons) needs a global player key. The G: tomes have none that I could find. The only candidate, steam_id in basic_player_personal, is a single alphabetic character with 19 distinct values in 2026 and 10 in 2024 (VERIFIED, probe). It is a slot label, not a SteamID.
4. Shapley on the 10 players of a round costs 2^10 = 1,024 coalition evaluations per state (exact), or a permutation-sampling estimate (VERIFIED formula source: survey section 3). A win-probability model that accepts an arbitrary subset of alive or present players is required. I did not find a CS paper that does this.
5. Esports precedent: Xenopoulos 2020 (CS:GO) is the only esports paper I opened that assigns win-probability change to individual players (VERIFIED, PDF pp.5-6). I found no CS or LoL paper that computes player-level Shapley over a round win probability (search coverage: arXiv and OpenAlex keyword searches; Semantic Scholar was rate-limited).

## 2. Method summary

| Method | Formula (short) | Key assumptions | Works with within-match slot IDs? | Source and status |
|---|---|---|---|---|
| Shapley value | phi_i = (1/n) sum over S not containing i of [v(S+i) - v(S)] / C(n-1,\|S\|) | v(S) defined for every coalition; four axioms (null, efficiency, symmetry, linearity) characterize the value | Yes, within a match. Cross-match pooling needs a global key | arXiv 2202.05594, Def. 7, Eq. 1, Thm. 1 (VERIFIED, PDF pp.2-3) |
| Permutation-sampling Shapley | average of marginal contributions over k random orderings | Uniform permutation sampling; error bounds from CLT or Hoeffding (Castro et al. 2009 as cited) | Yes | arXiv 2202.05594, sec. 3.1, Alg. 1 (VERIFIED, PDF p.4) |
| Feature-level Shapley on win-loss model (MVP-Shapley) | phi_i(x) = (1/2pq) sum over S of [f(x_{S+i})_win - f(x_S)_win] / C(2pq-1,\|S\|) | LightGBM win-loss model f; features padded to p players per team; paired home and away rows; TreeSHAP exact values | Yes per game (slot-ordered features). Season MVP needs player identity across games (sec. 3.4.2) | arXiv 2506.04602, Eqs. 1, 3, 4, 5, 6 (VERIFIED, PDF pp.2-3) |
| Lineup Shapley (generalized) | Shapley with v(S) = P(lineup S wins), from logistic regression on Dean's factors | Lineup-level win model; sparse lineup coverage; bootstrap CIs | Needs cross-season player IDs (NBA). Not usable on CS G: tomes as-is | DOI 10.1007/s10479-022-04653-z (UNVERIFIED, abstract only; Springer full text not reachable) |
| Lineup or team Shapley with convex optimisation | Shapley on team matchups; lineups as teams, possessions as games | Team outcome data sufficient; per-possession outcomes | Team-level | Rowe, ojs.zefr.org/index.php/intplanet/article/view/48 (UNVERIFIED, abstract only) |
| Restricted Shapley (football) | Shapley with coalitions restricted to passing-network subsets; cohesion from xGA | Passing network defines admissible coalitions | Yes, within a match | arXiv 2603.11016 (UNVERIFIED, abstract only) |
| WPA (per event) | WPA_e = WE(after e) - WE(before e), credited to batter (+) and pitcher (-) | Calibrated win-expectancy model; additive over events; no interaction credit | Yes (actor per event) | FanGraphs library, misc/wpa (VERIFIED, fetch; formula in words only) |
| Leverage index | LI = sum over outcomes of p_k times \|dWE_k\|, divided by the average potential WE swing; average LI = 1.0 | Outcome distribution of the next event (base-out or state model) | Yes. Weight only, not a credit | FanGraphs library, misc/li (VERIFIED, fetch; formula in words only) |
| CS:GO action value (Xenopoulos 2020) | V(a_t) = Yhat_{t+1} - Yhat_t for damage events; sign flipped for the receiving side; player WPA = sum of V per round | Calibrated round-win model Yhat (boosted trees chosen for calibration); only damage events are actions | Yes (PlayerName per event; within-match identity) | arXiv 2011.01324, Eq. 5, sec. III-B, V (VERIFIED, PDF pp.5-6) |
| VAEP | V(a_i) = dP_scores(a_i) - dP_concedes(a_i); rating(p) = (90/m) sum of V(a_i) | Scoring and conceding models over the next k=10 actions; features from last three actions; CatBoost for calibration | Yes (actor per action); cross-match rating needs a global key | arXiv 1802.07127, Def. 1, Eqs. 1-4, secs. 3.1-3.2, 4.1-4.2 (VERIFIED, PDF pp.2-4) |
| Expected possession value (EPV) | decomposed into subcomponents; calibrated models per component | Tracking data needed | Tracking required | arXiv 2011.09426 (UNVERIFIED, abstract only) |
| nflWAR | EP by multinomial logit; WP by GAM on EP; player WAR by multilevel models | Public play-by-play; on-field player lists per play for extension | Yes if on-field lists per play exist | DOI 10.1515/jqas-2018-0010 (UNVERIFIED, abstract only) |

## 3. Method notes

### 3.1 Shapley value (general definition)

- Formula (VERIFIED, arXiv 2202.05594 sec. 2.1 Def. 7 and Eq. 1; PDF pp.2-3): phi_i = (1/|Pi(N)|) sum over permutations pi of [v(P_i^pi union {i}) - v(P_i^pi)], where P_i^pi is the set of players before i in pi. Equivalent subset form (MVP-Shapley Eq. 1, VERIFIED): phi_i = (1/n) sum over S subset of N\{i} of [v(S union {i}) - v(S)] / C(n-1,|S|).
- Axioms and uniqueness (VERIFIED, same source, Thm. 1 citing Shapley 1953): null player, efficiency, symmetry, linearity together characterize the Shapley value.
- Cost (VERIFIED, same source sec. 3): exact computation needs 2^n characteristic-function evaluations. Monte Carlo permutation sampling (Castro et al. 2009) and KernelSHAP-style weighting (Lundberg et al. 2017) are the standard approximations.
- Assumptions needed (UNVERIFIED analysis): v(S) must be defined for coalitions S that did not occur. In a CS round, S is a set of alive or present players. The value of a player then depends on how the model behaves off the observed support. This is a real identifiability issue, not a detail.

### 3.2 Shapley in sports and esports

- MVP-Shapley (arXiv 2506.04602, VERIFIED, PDF pp.2-3): v(S) = f(x_S)_win, where x_S keeps the feature indices of coalition S. The "players" are feature blocks in a 2pq-dimensional vector of paired home and away rows (Sec. 3.1). Player contribution = sum of feature-level Shapley values over the player's home features minus the away features (Eq. 5). The season MVP needs the same player across games (Sec. 3.4.2, "player p_i who has participated in T games"). Unstated in the pages read: how absent features are filled in when a coalition excludes them (baseline or path-dependent TreeSHAP). This matters for the value of v(S).
- Generalized lineup Shapley (Metulini and Gnecco; DOI above, UNVERIFIED, abstract only): v(S) is the probability that a lineup wins, estimated by logistic regression. Bootstrap CIs. NBA 2004/05 to 2020/21. Cross-season identities are needed.
- Rowe (2026, UNVERIFIED, abstract only): Shapley on team outcomes; lineups as teams and possessions as games.
- PRSV (Cefis et al. 2026, arXiv 2603.11016, UNVERIFIED, abstract only): coalitions restricted to tactically admissible subsets; cohesion from an action-based xGA.
- Shapley for conversions across channels (Zhao et al. 2018, arXiv 1804.05327, UNVERIFIED, abstract only): a simplified Shapley formula for advertising credit. A possible precedent for the Shapley-with-subsets computation.
- Joint Shapley values (Harris et al. 2021, arXiv 2107.11357, UNVERIFIED, abstract only): interaction-aware credit for feature sets, relevant if lineup synergy matters.

### 3.3 WPA

- Definition (VERIFIED, fetch of library.fangraphs.com/misc/wpa; the page prints the definition in words, not a formula): WPA of an event is the change in win expectancy across that event. Credit is symmetric: the batter gains what the pitcher loses. Pitchers receive all defensive credit or debit. A player's total is the sum of event values.
- Assumptions (UNVERIFIED analysis): the win-expectancy model is calibrated; events are the unit; credit is not shared among the players involved beyond the pitcher-batter split.
- CS adaptation (Xenopoulos 2020, see 3.5): round-level Yhat plays the role of WE.

### 3.4 Leverage index

- Definition (VERIFIED, fetch of library.fangraphs.com/misc/li; formula given in prose): LI is the probability-weighted absolute change in win expectancy over possible outcomes, divided by the average potential swing. Average LI = 1.0. LI depends on the state (inning, score, outs, runners). The exact published formula (Tango et al., The Book) was not opened. UNVERIFIED as a symbolic formula.
- Use: a weight for the pivotal states, not a credit allocation. In CS terms, the analogue would weight round states by how much the round outcome could still move. No CS paper using LI was found.

### 3.5 CS:GO action value (Xenopoulos, Doraiswamy, Silva 2020)

- Eq. 5 (VERIFIED, arXiv 2011.01324 PDF p.5): V(a_{i,t}) = Yhat_{i,t+1} - Yhat_{i,t}, where Yhat = P(CT wins round i | game state). Sign is flipped for the receiver of the action. Player WPA is the sum of V over the player's actions, normalised per round. Only damage events are actions.
- Model (VERIFIED, PDF p.6, Table II): logistic regression, CatBoost and XGBoost. Test log loss: 0.5539, 0.5443, 0.5353 (XGBoost best). Brier: 0.1912, 0.1875, 0.1842. Map-average baseline: log loss 0.6917. Training 55M states (2016-10-23 to 2019-05-31), test 18M (2019-06-01 to 2019-12-22). Calibration plot Fig. 5.
- Data model (VERIFIED, PDF pp.3-4 Fig. 1): events carry PlayerName (a string) inside the round object. Identity is therefore within-match by name, which suffices for per-match credit.
- Assumptions (VERIFIED from text + UNVERIFIED analysis): Yhat is calibrated; action credit is local; the choice of damage events ignores kills, flashes, economy decisions. The paper's WPA is the closest published per-player CS credit method.

### 3.6 VAEP (Decroos, Bransen, Van Haaren, Davis 2019)

- Definition (VERIFIED, arXiv 1802.07127 PDF pp.2-4, Eqs. 1-4): dP_scores(a_i, x) = P_scores(S_i, x) - P_scores(S_{i-1}, x) and the same for conceding. V(a_i, x) = dP_scores(a_i, x) - dP_concedes(a_i, x) (Def. 1). Player rating = (90/m) sum of V over the player's actions in the window (Eq. 4).
- Models (VERIFIED, PDF pp.3-4): k = 10 actions look-ahead (Sec. 4.1); features from the last three actions (Sec. 4.2); CatBoost used for the probability models, with the calibration justification deferred to Sec. 5.6.3 (VERIFIED, text reference only). Data: Wyscout, 2012/13 to 2017/18.
- Assumptions (VERIFIED from text): the probability models are the entire valuation; the state is an action window (approximate Markov); calibration matters. There is no coalition: credit is additive per actor.
- Identity: actor per action (SPADL attribute "Player", PDF p.2). Within-match suffices for per-match value. Ratings across matches need a global key.

### 3.7 Other sports win-probability models

- Lock and Nettleton 2014 (DOI 10.1515/jqas-2013-0100, UNVERIFIED abstract): random forest WP before each NFL play. Used to evaluate plays.
- Yurko et al. 2019 nflWAR (DOI 10.1515/jqas-2018-0010, UNVERIFIED abstract): EP by multinomial logit, WP by GAM, player WAR by multilevel models. The abstract says the framework extends to any position "if researchers have data specifying the players on the field during each play". That is the same requirement as the slot-level data here.
- Yurko et al. 2020 (DOI 10.1515/jqas-2019-0056, UNVERIFIED abstract): continuous-time within-play valuation from tracking data.
- Brill, Yurko, Wyner 2025 (DOI 10.1515/jqas-2024-0130, UNVERIFIED abstract): in simulation, dependence in observational play-by-play data inflates bias and variance and lowers effective sample size; intervals need to be wide. Directly relevant to the assumptions here: rounds within a match share players, map and team, so cluster by match_key.
- STATS X (Shimizu and Yamanobe 2026, arXiv 2602.19513, UNVERIFIED abstract): latent contribution index from a time-evolving dominance process on minute-level box scores. Not Shapley.

### 3.8 Counterfactual and Shapley credit in multi-agent ML (same math, different value function)

- COMA (Foerster et al. 2017, arXiv 1705.08926, UNVERIFIED abstract): counterfactual baseline that marginalises out one agent's action with others fixed.
- Shapley Q-value (Wang et al. 2019, arXiv 1907.05707, UNVERIFIED abstract): distributes a global reward by Shapley value of agent contributions.
- STAS (arXiv 2304.07520, UNVERIFIED abstract): Shapley over temporal and spatial credit with Monte Carlo sampling.
- CCPO (arXiv 2603.21563, UNVERIFIED abstract): leave-one-out counterfactual removal of an agent.
- Survey of Shapley in ML (arXiv 2202.05594, VERIFIED, sec. 2-3): see 3.1.

## 4. Event-data check on G: tomes (per-player ID question)

Probes were read-only and printed counts and shapes only. No steam IDs or player names are reproduced here.

| Check | Result | Status |
|---|---|---|
| Header 2026 (1,752 rows) | Has match_id, key, map_name, unique_steamids (10 in the three rows printed); no per-player key | VERIFIED (probe 1) |
| basic_player_death 2026 (253,480 rows) | player_id, attacker_id, assister_id (in-match slot integers; 65535 seen with no assister); player_id_fixed (1-10 in most matches); position and velocity at death; match_key | VERIFIED (probe 1) |
| basic_player_info 2026 (347,243 rows) | player_id (raw), player_controller_id, player_id_fixed, team_code, wins, rank, rank_type | VERIFIED (probe 1, 2) |
| Raw slot vs fixed slot | One-to-one within match in 1,750 of 1,751 matches | VERIFIED (probe 2) |
| Fixed slot count per match | 1,216 of 1,751 matches have exactly 10 distinct fixed IDs; the rest have a different count (IDs 11-19 appear in the raw and fixed columns) | VERIFIED (probe 2) |
| Death rows vs info-derived mapping | 253,126 agree, 285 disagree, 69 have no mapping (total 253,480) | VERIFIED (probe 2) |
| Global key scan | Only basic_player_personal has a steam-like column (steam_id). No other table has a steam or account key. The name column (basic_player_personal.name; basic_player_name name_old, name_new) exists; stability across matches not tested | VERIFIED (probe 1 schema scan, probe 3) |
| steam_id in 2026 (347,278 rows) | One alphabetic character in every row; 19 distinct values; 17 of them occur in more than one match; each slot maps to one value per match | VERIFIED (probes 3 and 4) |
| steam_id in 2024 (781,793 rows, 4,435 matches on one page) | One alphabetic character; 10 distinct values | VERIFIED (probe 4) |
| team_code | Values 2 and 3 (and 0 in two rows). 16,397 slot-matches show both values, consistent with a halftime side swap | VERIFIED (values, probe 3); side-swap reading UNVERIFIED |

Answer to the question in R3: yes. Within a match, the slot IDs (fixed IDs 1-10, mapped from raw IDs) are sufficient for every method in section 2 except the cross-match rules. The premise that IDs are unique only within a match holds for this tome (VERIFIED). Cross-match player rating needs a key that these tomes do not contain. Either find one outside the tomes (steam IDs in the raw demos on E:, which I was told not to read) or accept match-level attribution only. Name matching is a weak fallback; its reliability is UNVERIFIED.

## 5. Implications for CS2 win-probability attribution (analysis, UNVERIFIED)

- Exact Shapley over the 10 slots per state: 1,024 coalitions. At about 145 deaths per match (253,480 / 1,751 = 144.8, from probe 2 counts), the event-state count is in the low hundreds per match. An exact enumeration over states is therefore about 1.5 x 10^5 evaluations per match (145 x 1,024). That is a feasibility estimate only, not a benchmark.
- The value function is the hard part. The 2020 CS:GO model uses team-level aggregates (players remaining, equipment, HP), not player identities. A Shapley game over players needs a player-level win model that can be evaluated for absent players. Options: (i) train a model with player-slot features and mask absent slots (MVP-Shapley style); (ii) use team-level aggregates with a counterfactual "player i removed" that adjusts players remaining and HP, which is a simpler but cruder v(S).
- Simplest defensible per-action credit: Xenopoulos-style WPA on all event types (not only damage), with per-match clustering of uncertainty (Brill et al.). Shapley adds fairness across simultaneous contributions but adds model dependence.

## 6. Gaps

- Semantic Scholar search was rate-limited (HTTP 429) and was not retried.
- Springer full text for the generalized Shapley paper was not reachable.
- No full text was read for arXiv 2603.11016, arXiv 2602.19513, or the Rowe paper.
- The FanGraphs exact LI formula (The Book) was not opened.
- Esports coverage: no CS or LoL paper computing player-level Shapley over round win probability was found in arXiv or OpenAlex keyword searches.
- The 2024 tome was checked only for steam_id, not for the raw-to-fixed slot mapping.

## 7. Sources opened

- arXiv 2202.05594 (Shapley in ML survey): https://arxiv.org/pdf/2202.05594 (PDF pp.2-4, VERIFIED)
- arXiv 1802.07127 (VAEP): https://arxiv.org/pdf/1802.07127 (PDF pp.1-4, VERIFIED)
- arXiv 2011.01324 (Xenopoulos CS:GO): https://arxiv.org/pdf/2011.01324 (PDF pp.3-6, VERIFIED)
- arXiv 2506.04602 (MVP-Shapley): https://arxiv.org/pdf/2506.04602 (PDF pp.2-3, VERIFIED)
- FanGraphs WPA: https://library.fangraphs.com/misc/wpa/ (fetch, VERIFIED summary)
- FanGraphs LI: https://library.fangraphs.com/misc/li (fetch, VERIFIED summary)
- Abstracts only (UNVERIFIED): arXiv 2603.11016, 2602.19513, 2011.09426, 1810.02252, 1705.08926, 1907.05707, 2304.07520, 2603.21563, 1804.05327, 2107.11357, 2302.00736, 2501.10049, 2403.04873, 2409.05052; DOIs 10.1007/s10479-022-04653-z, 10.1515/jqas-2013-0100, 10.1515/jqas-2018-0010, 10.1515/jqas-2019-0056, 10.1515/jqas-2024-0130, 10.1145/3549501; https://ojs.zefr.org/index.php/intplanet/article/view/48.
- Probe scripts: G:\pureskill\csds-cs2\scripts\swarm\r3_schema_probe.py, r3_id_probe.py, r3_steam_probe.py, r3_steam_shape_probe.py.
