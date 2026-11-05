# Training objective and evaluation metric

This note says what the models were trained on and what they were judged on. It is a record, not a pin. The rules that govern choices are in Meridian.

## What the models are trained on

- **Training objective: log loss.** The XGBoost models use `binary:logistic`, with `logloss` as the early-stopping metric on the early-stop rows.
- **GDT is not a training objective.** No model in this note was trained on GDT, and no GDT-based sample weights were used. GDT only appears in scoring.

## What the models are judged on

- **Validation: GDT** from the intern grader (`evaluation/intern_gdt/score_gdt.py`) is the selection metric. GDT is the table's log loss minus the model's, on the same sampled datapoints, with one datapoint per round per situation. Positive means the model beats the fixed lookup table.
- **Frozen test: not read for choices.** Any GDT read on it needs a pin and the owner's go, and it happens only at the final stage.
- **Oct 26-31 holdout: one final read** for the chosen candidate, after it is named to Dr Freeman. Its metric needs pinning before the read.
- **Log loss is reported alongside GDT.** Table log loss, model log loss and GDT always appear together, so the two measures can be read against each other. GDT equals the table's log loss minus the model's, so the rankings agree.
- **GDT_sit is not used for selection.** On the 297-match validation population it rests on 10 situations, and the 294 weak situations all fall below its 500-datapoint threshold.

## Validation numbers (297 matches, Sept 1 to Oct 8)

| Model | Log loss | GDT (95%) |
|---|---|---|
| Lookup table | 0.4610 | 0 |
| 59-input control | 0.4406 | 0.0204 |
| 57-input | 0.4399 | 0.0211 |
| 83-input (positions) | 0.4356 | 0.0254 (0.0218 to 0.0291) |

On the full 1,168-match validation set: 83-input 0.4502 and 57-input 0.4536 log loss. The 297-match and 1,168-match log losses are different populations and should not be compared directly.

## Not yet tested

- **Training on a GDT-based objective** (visit-weighted log loss). It would be a separate pinned A/B arm against plain log loss, validation only. Nothing in this note depends on it.
