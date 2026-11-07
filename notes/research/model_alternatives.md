# Model alternatives to XGBoost for calibrated, interpretable round win probability

Source: Haiku research agent, 2026-10-13. 25 sources opened; BibTeX in `model_alternatives.bib`. The agent could not write this file itself, so the findings were saved from its reply. Items marked UNVERIFIED were not confirmed in the official docs.

| Option | Install | Monotone constraints | Calibration | Interpretability |
|---|---|---|---|---|
| CatBoost 1.2.10 | about 100 MB wheel | yes, CPU only | UNVERIFIED | UNVERIFIED |
| EBM (interpret 0.7.8) | 15.6 MB plus numpy, pandas, sklearn | yes, post-fit monotonize advised | not stated | high |
| pyGAM 0.12.0 | 84 KB plus numpy and scipy | at term level | not stated | high |
| XGBoost | moderate | yes (1, 0, -1) | UNVERIFIED | low |
| Logistic regression with SplineTransformer | sklearn only | no | sklearn says "more likely calibrated" | highest |

## Recommendations (agent)

1. Lightest and easiest: logistic regression on spline features.
2. Most interpretable: EBM, or pyGAM if a lighter install is wanted.
3. Likely most accurate: monotone XGBoost with CalibratedClassifierCV fitted on held-out data.

## Evidence on XGBoost

- Xenopoulos et al. (arXiv 2011.01324, section IV-C) tested logistic regression, CatBoost and XGBoost on CS:GO round states. Test log loss: LR 0.5539, CatBoost 0.5443, XGBoost 0.5353. Their stated reason for XGBoost was calibration.
- Xia et al. (IEEE CoG 2025) also report XGBoost best; the numbers were not retrieved.
- Boosted outputs are often miscalibrated (Niculescu-Mizil and Caruana, arXiv 1207.1403), so calibrate whichever model is chosen.
- Our own bake-off tied XGBoost, LightGBM and HistGradientBoosting within 0.001 log loss, which suggests lighter options may match it.

## CatBoost and numpy 2

CatBoost issue 2671 closed on 2025-03-23 and is fixed in CatBoost 1.2.8 and later. Upgrade CatBoost rather than pinning numpy below 2. This explains why the CatBoost import failed in our pixi environment.

## Open items

- EBM's default for interactions differs between the docs ("3x") and the dev source (0.9).
- EBM and CatBoost calibration behavior is unverified.
- pyGAM LogisticGAM constraint syntax is unverified.
