# R2: Industry and product disclosures for CS2 win probability (new sources only)

Source: Haiku research subagent R2, 2026-10-13. Does not repeat `notes/research/industry_winprob.md` (Leetify 2022/2024, PandaScore 2022, GRID CS2 page, Winio 80-factor article, Scope.gg first-kill, Riot/AWS, Xenopoulos 2020, Choi 2023, Xenopoulos 2021, Rubin listing, nomiun).

Labels. VERIFIED = page opened this session (WebFetch; summaries extracted by a small model, so quotes are paraphrased) or script run (named). UNVERIFIED = seen only in a search summary, or the page was blocked. Benchmark arithmetic: `scripts/swarm/r2_industry_benchmarks.py` (run via pixi, see section 4).

## 1. Bottom line

- No new product or organizer page discloses a model class for a CS2 round-level win probability. The exceptions are a hobby GitHub README (five classes named), a 2026 undergraduate thesis (pre-match, L2 logistic regression), and LHM.gg Scout AI (says "neural network", no architecture). VERIFIED.
- Validation numbers. Only academic or hobby sources publish defined metrics: Xenopoulos 2020 (already in the earlier note), Rocha 2026 (AUC and accuracy), and lc-leonardo README (AUC, Brier, log loss). Vendor accuracy claims carry no definition, sample, or baseline: Winio "up to 89% model accuracy" (three articles, VERIFIED), Winio FAQ "70 to 80% long-term accuracy" (VERIFIED, one article), BLAST 2020 "73%" (UNVERIFIED, page blocked), sslip.io "75.7% vs 50.2% base rate" (UNVERIFIED, anonymous operator). No vendor page reports Brier, log loss, or calibration. VERIFIED.
- Round-level product claims seen: LHM.gg Scout AI says it predicts round results in CS2 (VERIFIED, no metrics). Leetify is the only product that names round-start win-probability inputs in detail (VERIFIED). Bayes and ESL FACEIT Group broadcast feature shows map, match and overtime probabilities, updated during the match (VERIFIED, from an advertising feature; Bayes primary page unreachable).
- Blocked or unreachable: HLTV (403), Esports Charts (redirect to a parked-domain sale page, then socket error), Bayes primary site (DNS failure), Strafe.gg and counterstrafe.pro (DNS failure), Valve GSI wiki (403 on both CS2 and CS:GO pages), BLAST 2020 article (403), csdb.gg (429), bo3.gg match center (blank render).
- "Strafe": no company, product, or model under that name found. VERIFIED negative search (search only). "Valve": no official CS2 win probability found; the GSI documentation that would list payload fields was blocked. UNVERIFIED negative.

## 2. Sources opened this session (VERIFIED)

| # | Source (date on page) | URL | What it says the model or feature uses | Model class disclosed | Validation numbers | Notes |
|---|---|---|---|---|---|---|
| 1 | PandaScore, CS2 live stats API developer guide (2026-05-26) | https://www.pandascore.co/blog/cs2-live-stats-api-dashboard-developer-guide | Describes data feeds only: Events feed (kill, round start and end, bomb plant and defuse, game end) and Frames feed (every 2 seconds: K/D/A, cash and equipment value, health, round score). No win probability feature described. | No | None | Events feed and some Frames fields are gated to a Pro Live plan. Contrast with its 2022 CS:GO Markov post (already in earlier note). |
| 2 | Winio, "What actually predicts wins in CS2 matches" (2026-05-26) | https://winio.ai/blog/articles/what-actually-predicts-wins-in-cs2-matches | Opening duels, economy, utility, map control; tournament K/D, AWP rating and map win rate called weak. Example weights 0.4 / 0.3 / 0.2 / 0.1 labelled illustrative. Displayed forecasts are pre-match (e.g. 74 vs 26). | No ("modern machine learning" mentioned, not named) | "Up to 89% model accuracy" (no definition); FAQ "70 to 80% long-term accuracy" (no definition) | Vendor. |
| 3 | Winio, "Pre-match and live predictions: what's the difference" (2026-07-06) | https://winio.ai/blog/articles/pre-match-and-live-predictions-what-s-the-difference | Live inputs for CS2: economy, rounds, map control, momentum. Live update rate not stated. | No | "Up to 89%" repeated, undefined | Vendor. Says live predictions update "when live data is available". |
| 4 | Winio, "Why even strong esports prediction models get matches wrong" (2026-06-19) | https://winio.ai/blog/articles/why-even-strong-esports-prediction-models-get-matches-wrong | Pre-match framing ("information available before the match"). No numeric accuracy ceiling. | No | "Up to 89%" in a stats block, undefined | Vendor. |
| 5 | Winio, CS2 forecasts page | https://winio.ai/forecasts/cs2 | Says users can track confidence and win-probability shifts for live and upcoming CS2 matches. Past matches can be compared under "Ended matches". | No | No accuracy %, hit rate, count, Brier or log loss shown | Track record exists in the UI but no aggregate is printed. |
| 6 | Abios press release, Bayes data partnership (2024-07-11) | https://abiosgaming.com/press/bayes-esports-official-data-partnership/ | Odds feeds and widgets built on Bayes official live data (BLAST Pro Series, ESL Pro League, IEM Katowice). "Proprietary modelling" for odds, no detail. | No | None (claims "more accurate pricing", no metrics) | Confirms Bayes is the data supplier to Abios. No win-probability output described. |
| 7 | Bayes Esports and ESL FACEIT Group live predictions, as carried by iGaming Business (2025-02-06, labelled "Advertising feature") | https://igamingbusiness.com/company-news/bayes-esports-and-esl-faceit-group-expand-partnership-to-enhance-broadcast-with-live-match-predictions/ | Per-map win probability, overall match, and overtime likelihood. Updated as in-game events occur. Input: EFG official live data. Launch at IEM Katowice Play-In. | No | None | Says "over 25 markets for CS2" (Bayes claim). Bayes primary release unreachable (see section 3). |
| 8 | GRID, GRID Insights launch (2025-06-10) | https://grid.gg/grid-insights-launch/ | "Predictive insights" in milliseconds: streaks, weapon stats, record chases. Described as enhanced by large language models, on official structured data. Early customer: European Pro League. Debut: Champion of Champions Tour S3 (CS2). | No | None | Win probability, round forecasts and milestones are NOT named as outputs. |
| 9 | GRID, widgets page | https://grid.gg/products/widgets/ | Game Overview widget shows a 68% win probability in a "Live preview". Insights widget surfaces momentum and milestones. | No | None | The page does not say whether the 68% is sample or live model output, or who computes it. Earlier note recorded a 62% sample panel on the GRID CS2 page. Both are unexplained. |
| 10 | Leetify, "Leetify Rating has been updated for CS2" (2024-11-25) | https://leetify.com/blog/leetify-rating-update | Four economy groups per side (equipment value), players alive is not named, previous-kill side matters, kill-chain grouping removed in favour of kill-by-kill. Calibrated on CS2 pro matches (earlier version: CS:GO pro matches). Applies to matches from 2024-11-25 onward. | No (win-probability chart only, no formula or lookup named) | None for the win model (rating thresholds only) | Sidebar lists a 2026-03-19 post "clutch weighting changed" that was not opened. |
| 11 | Leetify, "What is Leetify Rating" (legacy, posted 2021-09-16) | https://leetify.com/blog/what-is-leetify-rating/ | Economy of both teams at round start, living players, kill chains within 4 seconds grouped. Side not named as an input in this version. Credit shares 35/30/20/15 are allocation rules, not the probability model. | No | None | Page says it "only applies" to matches before 2024-11-26. Minor conflict with item 10 (2024-11-25). |
| 12 | Scope.gg, CS2 prematch analytics guide (dated 2022-08-03) | https://scope.gg/guides/prematch_scopegg_en/ | CT-side enemy position probabilities and push likelihood, from each player's last five matches during the 5-minute FACEIT warmup. Not win probability. | No | None ("90%" is a display example) | Confirms Scope.gg has no published win model in this guide. |
| 13 | LHM.gg, Scout AI feature page (date not shown) | https://lhm.gg/features/scout-ai | Claims per-round and whole-match result predictions for CS2 (Scout Predict, built into the Ultra and Premium CS2 HUDs). Uses predicted events to pick camera focus. "Own AI" that "learns from thousands of games". | "Neural network" named; architecture not given | None (mentions in-house "review algorithms", no metrics) | Only product seen that claims round-result prediction in CS2 with a stated class. Says not affiliated with Valve. |
| 14 | mwasoft CS2 live odds API, dev.to post by its developer (2025-11-04) | https://dev.to/__975b3bb4/real-time-cs2-odds-api-no-scraping-no-delays-just-clean-data-4o1g | "50+ in-game signals" (economy categories, momentum, map control, clutch, utility). Calculates match probabilities from game state. | No | None; free tier of 100 requests per day | Vendor and promotional. Claims latency without numbers. |
| 15 | zest.win, Product Hunt listing (2026) | https://www.producthunt.com/products/zest-win | AI win probabilities and bookmaker odds for LoL, CS2 and Dota 2. Says predictions are "settled and scored publicly". Live probabilities not claimed (a commenter asked for them). | No ("ML-backed") | None given | Vendor. Paid breakdowns. |
| 16 | Recon, FACEIT CS2 Predictor (Firefox add-on, v1.0.0.33, updated 2026-06-22) | https://addons.mozilla.org/sr/firefox/addon/faceit-cs2-predictor/ | Pre-veto per-map win probability. Claims a model trained on "more than 30 million" FACEIT matches. Uses team composition, map history, recent form. | No | None ("statistical estimates") | Third party. States it is not affiliated with FACEIT. |
| 17 | CounterStrafe Analytics, Chrome Web Store (updated 2026-05-13, v2.1.0) | https://chromewebstore.google.com/detail/gcneoaolbpddgicmkjebeffdeaecfocj | Win probability "powered by an ML model trained on 2.5M+ FACEIT matches". Updates during map veto. | No | None | Third party. Website counterstrafe.pro failed DNS. |
| 18 | lc-leonardo/cs2-match-prediction, GitHub README (undated; 5 commits; MIT; 2 stars) | https://github.com/lc-leonardo/cs2-match-prediction | Logistic regression, random forest, XGBoost, LightGBM and MLP on BO3.gg data (last 6 months, 2,400+ map records from 1,000+ matches). Calibration analysis mentioned, not shown. | Yes (five classes named) | LightGBM: ROC-AUC 0.72, Brier 0.231, log loss 0.659, accuracy 67.2%. Test split not specified. | Hobby project. Map and match level, pre-match. Not live. |
| 19 | Rocha 2026, undergraduate thesis, UFSCar (repository abstract and metadata only) | https://repositorio.ufscar.br/items/05d81da4-11fa-4a99-a98c-2935fbead599 | Pre-match, match-level. Selected model: L2 logistic regression (C = 0.3), 11 variables. Holdout is the 2026_s1 window. | Yes (logistic regression) | Holdout ROC-AUC 0.6828; accuracy 0.6293; log loss, Brier and calibration not stated | Full PDF not read. Data source not stated. |

## 3. Found but not opened, or blocked

Blocked or unreachable (direct fetch attempted this session):
- HLTV, https://www.hltv.org/ : HTTP 403. Only academic papers were found that use HLTV as a data source (search summary, UNVERIFIED).
- Esports Charts, https://esportscharts.com/ : 307 redirect to a parked-domain sale page. https://www.esportscharts.com/ : "Socket is closed". No methodology page reached. Search also found nothing from this domain.
- Bayes Esports primary site (bayesesports.com and www.bayesesports.com): DNS failure from this environment for the EFG press release and the Live Match Predictions product page. The IEM Katowice write-up and the tournament-organizer blog post appeared in search results but were not fetched. Syndicated copies used instead (items 6 and 7).
- Trade-press copies of Bayes items: gamblinginsider press release (403), sbcnews EFG article (403), sbcnews Bayes interview 2025-03-19 (403), igamingfuture probabilities-as-a-service release (403).
- nordot.app Bayes "How to measure esports' natural born killers" (404).
- esportsinsider, BLAST 2020 prediction tool article (403).
- csdb.gg, player-form and hot-picks (429 on both).
- strafe.gg (DNS failure). counterstrafe.pro (DNS failure).
- Valve Game State Integration, CS2 and CS:GO pages (403 on both).
- 178-156-191-67.sslip.io/methodology/models (TLS error; anonymous operator; not relied on).
- bo3.gg/de/matches/current (blank render; "Prediction" column seen only in a search summary, UNVERIFIED).
- europeangaming.eu GRID Insights article (403; replaced by grid.gg page, item 8).

Search summaries only (UNVERIFIED):
- Bayes: "How to measure esports' natural born killers" (CS:GO, 6,423 matches, over 170K rounds, "well calibrated" self-assessment, round-winner model with map, bomb state, equipment as inputs). Source copy was 404; details from search summary only.
- Bayes: "first-ever CS2 event with live match probabilities" (vendor claim); "three markets shown on broadcast" versus "three of twenty markets" in an SBC interview (conflicts with item 7's "over 25"); "28% overtime at round 21 on Mirage" (BLAST or Bayes post-event write-up).
- Bayes probabilities-as-a-service release (2023 era): accuracy-advantage claim from a Predex press release; no methodology shown.
- BLAST Fall Final 2023 AI predictions: a gamereactor article (2023-11-22, https://www.gamereactor.eu/?p=88442) is VERIFIED. It describes AI in-game predictions that estimate the likelier match winner. It names no vendor, model, or accuracy figure. BLAST 2020 Skybox CS:GO prediction tool "about 73% accuracy" is UNVERIFIED (search summary; the 2020 article was blocked; a different event from the 2023 feature).
- Sportradar: probability feeds for tennis and soccer on developer.sportradar.com. No CS2 live probability found. UNVERIFIED negative.
- ESL FACEIT Group: no in-house methodology found. Only the Bayes-supplied feature (item 7).
- FACEIT official: none found. Third-party store listings (Faceit Predictor by Blalex, Faceit EGO Enhancer by L1BERI) describe ELO, RWS and recent-form models before veto. Listing text from search snippets only; UNVERIFIED.
- csdb.gg "fair-price" model (recent form, head-to-head, veto strength, ratings). Search summary only; page rate-limited.
- Anonymous sslip.io pages: a "round_wp_cs2" logistic-regression baseline and a 75.7% accuracy claim against a 50.2% base rate. Operator not named. Do not rely on.
- CS2 Hub Chrome extension (bo3.gg data, toggles win predictions). Search only.
- ITB Makalah 2026 (dynamic-programming recurrence on CS2 match win probabilities). PDF stream not parsed by the fetch tool. Not read.
- Polymarket trackers (Polyguana): market-implied prices, not a model. Excluded.
- GRID Open Access: whether CS2 win probability is in the free tier, not confirmed. GRID Portal docs need login.
- Valve: no official CS2 win probability found in search. Negative, UNVERIFIED.

## 4. Benchmark arithmetic (script run, VERIFIED)

Script: `G:\pureskill\csds-cs2\scripts\swarm\r2_industry_benchmarks.py`, run with `pixi run python -I -X utf8`. Coin-flip reference: Brier 0.25, log loss ln 2 = 0.6931. Brier skill is computed as 1 minus Brier over 0.25, which assumes balanced outcomes. The lc-leonardo base rate is not stated, so treat its skill column as approximate. Xenopoulos rows are per game state (18M test states), not per match, so they are not directly comparable to match-level rows.

| Source | Model | Log loss | Gain vs ln 2 (nats) | Brier | Brier skill vs 0.25 | ROC-AUC |
|---|---|---|---|---|---|---|
| Xenopoulos 2020 | XGBoost | 0.5353 | +0.1578 | 0.1842 | +0.263 | 0.7913 |
| Xenopoulos 2020 | CatBoost | 0.5443 | +0.1488 | 0.1875 | +0.250 | 0.7851 |
| Xenopoulos 2020 | Logistic regression | 0.5539 | +0.1392 | 0.1912 | +0.235 | 0.7743 |
| Xenopoulos 2020 | Map-average baseline | 0.6917 | +0.0014 | 0.2493 | +0.003 | 0.5303 |
| lc-leonardo README | LightGBM | 0.6590 | +0.0341 | 0.2310 | +0.076 | 0.7200 |
| lc-leonardo README | Random Forest | 0.6650 | +0.0281 | 0.2330 | +0.068 | 0.7100 |
| lc-leonardo README | XGBoost | 0.6720 | +0.0211 | 0.2350 | +0.060 | 0.7000 |
| lc-leonardo README | MLP | 0.6750 | +0.0181 | 0.2380 | +0.048 | 0.6900 |

Reading: the hobby CS2 match-level models sit at roughly 3 to 8 percent Brier skill over a coin flip, and the Xenopoulos map-average baseline is essentially a coin flip at game-state level. Rocha 2026 reports AUC 0.6828 and accuracy 0.6293 without a base rate, so no skill figure can be computed. VERIFIED (arithmetic run).

## 5. Contradictions and corrections

- Leetify effective date: the legacy page says "before 2024-11-26"; the update post says from 2024-11-25. VERIFIED on both pages. The difference is one day and is probably boundary wording.
- Leetify inputs: the legacy page does not name side as an input; the update post names previous-kill side. Consistent with the update being a change. VERIFIED.
- Bayes market count: "over 25 markets for CS2" (VERIFIED, item 7) versus "three of twenty" (UNVERIFIED, search summary of an SBC interview that returned 403).
- GRID CS2 win-probability panel: the earlier note recorded a 62% panel labelled as a mockup on the GRID CS2 page. The GRID widgets page (item 9) shows a 68% "Live preview" with no label on whether it is sample data. Neither page documents the method.
- The earlier note says Bayes was unreachable. This session confirms the primary Bayes domain fails DNS, but syndicated copies were reachable.

## 6. Suggested next steps

1. Retry Bayes primary pages and HLTV from a different network path. Both block or fail from this environment.
2. Request Winio's "Ended matches" data (if exported) to compute a real out-of-sample Brier score. Its aggregate accuracy claims are undefined.
3. Ask LHM.gg for a validation note on Scout Predict, since it is the only CS2 round-result claim with a named model class.
4. Check the lc-leonardo test split before citing its Brier and AUC numbers.
5. Download nothing from the sslip.io or dev.to sources. Treat them as data only.
