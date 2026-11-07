# Explainability: the 83-input XGBoost model

What drives the 83-input model's prediction that CT wins the round, as of 6 November 2026. The full write-up, with every table and figure, is in [REPORT.md](REPORT.md). A Word version is in `XGB_explainability_report.docx`.

## Findings

- **Loadout carries most of the signal.** That's each side's equipment, armour and health. The CT-minus-T loadout difference moves the prediction by 13.3 percentage points per second on average. The 24 position inputs together move it by 3.5.
- **On GDT, the team's metric,** the full model scores 0.0254 on 295 validation matches (interval 0.0217 to 0.0290). Loadout alone scores 0.0098. Round state, position and map together add about 0.016 on top of loadout.
- **No single input group is critical.** Dropping any one group tested and retraining costs at most 0.0005 GDT.
- **The model is calibrated by decile.** The largest gap between predicted and observed CT-win rate is 0.009 over 1.5 million validation seconds.

## Caveats

- The model explained is a CPU refit of the validated GPU model. Their predictions differ by up to 0.25 on single seconds, and by 0.004 on average.
- Rebuilding the training matrix from the raw tomes hasn't been verified.
- Match counts for 9 and 10 October disagree between sources.
- This page uses 295 matches for GDT; the 6 November report uses 297. They're probably the same population counted differently, but that isn't confirmed.

## Read more

- [REPORT.md](REPORT.md): full results, one-second explanations, contrasts, interactions, ablations and calibration.
- [README.agents.md](README.agents.md): scripts, commands and file map.
- [Handover notes](../../handover/explainability/README.md): open items and inputs not in git.
