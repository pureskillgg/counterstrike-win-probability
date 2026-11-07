# CS2 round win probability

Models that estimate, second by second, the probability that the CT side wins a Counter-Strike 2 round. This repo holds the code, the trained models, the documentation and the working notes from October and November 2026.

## Start here

- **[models/README.md](models/README.md)**: which model file is which. The serving code uses `winprob_v6_drop_heavy` (XGBoost, 57 inputs).
- **[TRAINING.md](TRAINING.md)**: how to retrain each model, and which upload days each one used.
- **[DATA.md](DATA.md)**: where the data lives, how it was built, and what's known about personal data.
- **[sequence/README.md](sequence/README.md)**: the sequence-model screen (GRU, LSTM, transformers, xLSTM and others) against XGBoost.
- **[docs/explainability/README.md](docs/explainability/README.md)**: what drives the 83-input XGBoost model.
- **[evaluation/intern_gdt/README.md](evaluation/intern_gdt/README.md)**: GDT, the team's metric, which is the gain over a fixed lookup table.

## Results in brief

All numbers are validation results unless marked. Lower log loss is better; higher GDT is better.

- **Serving-default XGBoost (v6, 57 inputs):** validation log loss 0.4536; frozen-test log loss 0.4564.
- **XGBoost with positions (83 inputs):** validation log loss 0.4503, the mean of three seeds. GDT 0.0254 on the 297-match validation proxy. October 26 to 31 holdout, read once: 0.4523.
- **Sequence models:** none beats the GRU control on GDT by the pre-registered bar. Training the GRU for 25,000 steps instead of 30,000 lowered its validation log loss on all five seeds, by 0.0011 on average.

## What's where

| Folder | Contents |
|---|---|
| `scripts/data/` | building the data: tomes, match tables, per-second grid, player status and positions |
| `scripts/model/` | training and experiments for every model family |
| `scripts/serving/` | the serving function and its parity test |
| `experiments/sprint/` | the 83-input XGBoost sprint, with its run records |
| `sequence/` | the sequence-model screen: trainer, pod scripts, analysis, notes |
| `evaluation/`, `analysis/` | the GDT grader and analysis scripts |
| `models/` | trained models (XGBoost JSON, PyTorch weights) |
| `data/benchmark/`, `data/manifests/` | test keys, split manifests, per-day match counts |
| `notes/`, `docs/`, `deliverables/` | reports, research notes, earlier review packages |
| `scratch/` | working scripts from the analysis sessions |

## Conventions

- Each `DOCUMENT.md` is for people. An optional `DOCUMENT.agents.md` next to it carries the detail an agent needs.
- Choices are made on validation data only. Frozen-test numbers were reported for several candidates, so treat them as descriptive. The 26 to 31 October holdout was read once, after the 83-input choice was fixed.
- Models are compared on the same rows. Comparisons over several seeds pair each seed with the same seed of the other model.

The code is under the MIT License, © FPS Critic, Inc.
