# Industry win probability for Counter-Strike: what is actually disclosed

Source: Haiku research agent, 2026-10-13. 13 sources opened; BibTeX in `industry_winprob.bib`. The agent could not write this file itself, so the findings were saved from its reply. Claims are tagged VERIFIED (source opened) or UNVERIFIED.

## Verified (sources opened)

| Company or source | What it discloses | Model class disclosed? |
|---|---|---|
| Leetify, "Leetify Rating explained" (2024) | CS2 round inputs are economy per side, players alive, and the previous kill | No. No formula or validation |
| PandaScore, "CS:GO round modeling" (2022) | A Markov-binomial baseline for CS:GO. Not a live product | Yes, but not a production model |
| GRID, CS2 data page | Win probability "direct from the game server" | No. The 62% panel on the page is a labelled mockup |
| Winio blog | "Multi-factor approach" | No |
| Scope.gg (2026) | First-kill round win rates of 64.8 to 72.6 percent | Descriptive statistics only, no model |
| Riot and AWS League of Legends (sportsvideo.org, 2023, secondary source) | ML with per-second predictions, no metrics | Not CS |

## Unverified or not found

- Bayes Esports: domain unreachable.
- BLAST and its data partner: search snippets only.
- HLTV Round Swing: HTTP 403.
- FACEIT, Esports Charts, Strafe: nothing found.

## What is established (academic)

- Xenopoulos et al. 2020 (arXiv 2011.01324, full PDF opened). CS:GO 2016 to 2019 data. Test log loss: XGBoost 0.535, CatBoost 0.544, logistic regression 0.554, map-average baseline 0.692. The authors cite boosted ensembles' tendency to produce well calibrated probabilities as the reason for choosing XGBoost.
- Choi et al. 2023 (arXiv 2309.06248) argues for Brier score and expected calibration error over accuracy.

## Bottom line

No industry source opened names its model class, so XGBoost is not an industry-established choice. The only CS paper opened that compares model classes picked it by a small margin, on its own data, for calibration. Lighter alternatives were not tested in that paper.

## Correction

An earlier note credited the CS:GO round model to Sunde et al. That author wrote a chess paper. The CS:GO paper is by Rubin (listing only, unverified).
