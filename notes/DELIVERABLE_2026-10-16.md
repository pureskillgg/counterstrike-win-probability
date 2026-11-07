# Deliverable wrap-up, 2026-10-16

> **Superseded:** the model and metric figures below are from before the final run. See `FINAL_RESULTS_2026-10-16.md` for the current models (`winprob_v4_basic` is the served default), the tuning and sweep results, and the provenance.

Scope: competitive 5v5 matches on the seven maps (ancient, anubis, cache, dust2, inferno, mirage, nuke). Wingman and hostage matches are excluded. The map list is the scope we chose, not a checked copy of the current Active Duty pool.

## 1. What ships now

| Item | Choice | Why |
|---|---|---|
| Served model | `models/winprob_v3_basic.json`, 13 inputs | It needs only the event channels the DSDK tome carries, so it runs on any tome. The parity test reproduces the grid to 0.0001 percentage points. |
| Best model | `models/winprob_v3.json`, 59 inputs | 0.45647 log loss, but not servable: its loadout features need per-tick player status, and the 2026 tome has no player-status channel. |
| Function | `scripts/serving/win_probability.py`: `win_probability(data, team="CT", unit="percent")` | `data` is a dict of DSDK DataFrames as-is. It returns `{tick: percent}` (or fractions with `unit="fraction"`). `team="T"` gives 100 minus CT on every tick. |

**The served model is 0.043 log loss worse than the best model** on the same 2026 test matches (below). Shipping it means accepting that gap until a servable 59-input version exists.

## 2. Metrics

Log loss is per second, on held-out matches, with 95% match-cluster intervals.

| Model | Inputs | Training matches | Test matches | Test log loss | 95% interval |
|---|---|---|---|---|---|
| `winprob_v3_basic` (served) | 13 | 12,563 (2024 + 2026) | 3,061 (2026) | 0.49957 | — |
| `winprob_v3_basic` (served) | 13 | 12,563 | 3,611 (all) | 0.50289 | [0.50047, 0.50540] |
| `winprob_v3_basic`, 550 2024 test matches only | 13 | — | 550 (2024) | 0.52281 | — |
| `winprob_v3` (best, not servable) | 59 | 10,577 (2026) | 3,059 (2026) | 0.45647 (deployed, every other second in training) | [0.45338, 0.45924] |
| `winprob_v3`, full-second comparison | 59 | 10,577 (2026) | 3,059 (2026) | 0.45641 | [0.45333, 0.45914] |

The 2026 rows (0.49957 against 0.45647) are the like-for-like comparison. The two test sets differ by two matches (3,061 against 3,059), and I have not checked that the 59-input model's test matches are a subset of the current manifest.

Calibration and AUC for the served model have not been measured on this test set.

## 3. Features

**Served (13):** alive_ct, alive_t, ct_alive_minus_t, bomb_planted, seconds_since_round_live, seconds_since_plant, and seven map one-hots (ancient, anubis, cache, dust2, inferno, mirage, nuke). Monotone constraints: alive_ct +1, alive_t −1, ct_alive_minus_t +1.

**Best (59):** the 13 above, plus 40 per-side loadout totals (health, armour, helmets, defuse kits, defusing, rifle, AK, M4, AWP, sniper, SMG, heavy, no primary, grenades by type, equipment value) and 6 CT-minus-T differences. Full list in `models/winprob_v3.meta.json`.

Not used by either model: positions, facing direction, smokes, fire, blindness, score lead. Positions give a paired gain of 0.0033 in a half-data run, but the tome carries no per-tick positions.

## 4. Splits and ticks

Current manifest (`data/combined/split_manifest.json`, 17,563 matches):

| Split | Matches | Share |
|---|---|---|
| Train | 12,563 | 71.5% |
| Validation | 1,389 | 7.9% |
| Test | 3,611 | 20.6% |

The 59-input model used the older 2026-only split: 10,577 / 1,168 / 3,059.

**Tick rule (all splits):** a row for every second of live play, at tick = freeze end + 64 × k. The rows stop at the round's effective end (the decisive kill or bomb event). With no bomb planted, they are capped at 116 seconds after freeze end. Rounds marked unused are dropped, along with rounds with a missing player.

**One example per split** (round 1 of a 2026 match):

| Split | Match | Round 1 seconds | First tick | Last tick | Bomb planted |
|---|---|---|---|---|---|
| Train | `csds/2026/09/01/01f19adb-166f-4666-95fb-fc75f186c486/csds` | 57 | 1441 | 5025 | yes |
| Validation | `csds/2026/09/01/7ba62e3e-f244-4ef4-a436-7af14e41019a/csds` | 56 | 1761 | 5281 | yes |
| Test | `csds/2026/09/01/10b093be-5095-4def-b6ab-660f09705c5d/csds` | 46 | 1441 | 4321 | no |

Round 1 of the test match is the worked example in `notes/METHODOLOGY_2026-10-16.md` §4.

## 5. Provenance

Served and best model files (sha256, current contents):

| File | sha256 (first 16) |
|---|---|
| `models/winprob_v3_basic.json` | 2c86590c24eed7d9 |
| `models/winprob_v3_basic.meta.json` | e0cf43ccabbcdc3c |
| `models/winprob_v3.json` | 2bd2852c1ae2479d |
| `models/winprob_v3.meta.json` | c4cf4a2f781e80fd |
| `data/combined/split_manifest.json` | a17f219b5287ff1e |
| `scripts/serving/win_probability.py` | 6eaa66e84047065a |
| `scripts/model/wp_common.py` | dc478d29a1e81e45 |

**Drift to fix before this is final:** both `.meta.json` files record `split_manifest_sha256 = 2088a5a6…`. The current manifest is `a17f219b…`, so the manifest was extended after training. The served model's training counts (12,563 / 1,389) match the current manifest, not the one its meta records. Each model should record the manifest hash it was actually trained on.

The writer of the two model files is not in `scripts/model/export_model.py`, so it is unidentified. Both models are recorded in Meridian (`record_provenance`) with this note.

## 6. Decisions and open items

1. **Which model ships.** My recommendation: ship `winprob_v3_basic` now, labelled as the event-only model with the 0.4996 test figure, and treat the 59-input model as the target for a servable version. Your call.
2. **Retrain on the current manifest and record its hash.** Needed before anything is called final.
3. **Find event-derived features the tome carries** (grenades, weapon fire, player hurt, deaths). Not measured; this is the route to closing the 0.043 gap without per-tick status.
4. **Positions and the every-n-seconds sweep.** Running on pod `z8og5z7t0rm1nc`, CPU fallback (XGBoost reports no visible GPU in the pod venv). Results not in yet.
5. **Not run:** the 40-trial hyperparameter search, the one-second vs quarter-second grid comparison, and the DSDK PR #49 test suite.
6. **Not done:** emailing the models. That needs your explicit permission.
