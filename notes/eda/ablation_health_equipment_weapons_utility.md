# Ablation 1: health, equipment, weapons, utility over the coarse round state

Script: `scripts/eda/ablation_hp_equip.py`. Log: `logs/eda/ablation_hp_equip.log`. Results JSON: `logs/eda/ablation_hp_equip.json`. Runtime 651 s. Command: `pixi run python -I -X utf8 scripts\eda\ablation_hp_equip.py`.

Question: which in-round information (team health and armor, equipment, weapon categories, utility, defuse kits) improves prediction of the round winner beyond the coarse BASE state?

## Data and sample

- Inputs: `data/state_sample/players_1hz.parquet` (1,231,817 rows), `data/state_sample/rounds.parquet` (2,673 rounds). Read-only. Nothing read from E:. VERIFIED (run).
- Preflight passes against the extraction validation JSON: kept set 1709/47/1521/28683/234265; alive agreement 179,457 of 179,466. VERIFIED (run).
- Alive-gap rounds: one round, `csds/2026/10/07/22e9c3e7-a333-4ebc-a51c-19ebbcc13ef1/csds` round 16 (train split, `alive_gap_snapshots` = 9). It is dropped from all main results. It has 74 snapshots, which are removed. Round counts after the drop: train 2,184, test 488. VERIFIED (run).
- Matches: train 120, test 30 (split column as given). Inner validation: 15% of train matches with seed 0, giving 18 validation matches (23,319 snapshots) and 102 fit matches (123,189 snapshots). Test: 32,884 snapshots. CT-win share: fit 0.509, validation 0.602, test 0.557. VERIFIED (run).
- Death check: alive counts match G-side `basic_player_death` on 179,457 of 179,466 snapshots. The 9 mismatches are the dropped round only. VERIFIED (run).

## Feature definitions

- BASE: `alive_ct`, `alive_t`, `alive_diff`, `bomb_planted`, `seconds_since_live`, `seconds_since_plant` (-1 if none), `round_number`, `score_diff` (ct_score minus t_score from rounds). BASE_NS is BASE without `score_diff`.
- HP: per-team health sum, armor sum, helmet count, count with health below 50.
- EQUIP: per-team `equipment_value_calc` sum, money sum, and the CT minus T equipment difference.
- WEAPON: per-team count of players whose active weapon (`player_vector.weapon_code`) is in each category: AWP, rifle, SMG, pistol.
- UTIL: per-team counts of flashbang, smoke, HE, molotov plus incendiary (inventory), and defuse kits.

### Weapon-code assumption

ASSUMPTION: `weapon_code` is the CS2 item definition index. This is the project's own statement in `notes/schema_dictionary.md` ("Weapon / item code lookup"), which says `weapon_code` equals `weapon_fire.player_weapon_code` at the same tick and player (checked on all matched shots). In this sample, 6,024 of 6,024 exact-tick pairs agree. VERIFIED (run).

Mapping used (`CODE_CAT` in the script):
- AWP: 9.
- Rifle: 7, 8, 10, 11, 13, 16, 38, 39, 60.
- SMG: 17, 19, 23, 24, 26, 33, 34.
- Pistol: 1, 2, 3, 4, 30, 32, 36, 61, 63, 64.
- Other: everything else, including SSG 08 (40), shotguns (25, 27, 28, 29, 35), M249 (14), zeus (31), knives, grenades (43 to 48), and C4 (49).

Each mapped code was checked against the G-side `weapon_fire` names: at least 99.8% of events for each mapped code carry a name in the same category (the check fails below 95%). VERIFIED (run). The codes 8, 11, 19, 23, 24, 26, 32, 38, 39 and 63 are absent from the schema dictionary's lookup table and were assigned from the standard CS2 index. The fire names confirm them in this sample. VERIFIED for this sample; UNVERIFIED for other days.

Code 49 (C4) has 7 `weapon_fire` events labelled m4a1 (agreement 0.43). These are asof-merge noise. C4 is treated as other. The first version of this script used a dominant-name lookup and aborted on exactly this code. It is replaced.

## Results: test split (30 matches), score_diff included

| Model | Test logloss | Brier | AUC | Best iter |
|---|---:|---:|---:|---:|
| CONST | 0.69132 | 0.24909 | 0.500 | - |
| BASE | 0.51683 | 0.17226 | 0.8190 | 343 |
| BASE+HP | 0.50737 | 0.16797 | 0.8290 | 331 |
| BASE+EQUIP | 0.49686 | 0.16526 | 0.8336 | 96 |
| BASE+WEAPON | 0.50986 | 0.16962 | 0.8264 | 289 |
| BASE+UTIL | 0.51725 | 0.17251 | 0.8207 | 316 |
| BASE+ALL | 0.50006 | 0.16614 | 0.8323 | 167 |

VERIFIED (run).

Cluster bootstrap over the 30 test matches (2,000 resamples), gain = logloss(BASE) minus logloss(model), positive means the model helps:

| Model | Gain | 95% CI | P(gain > 0) |
|---|---:|---|---:|
| HP | +0.00946 | [-0.00983, +0.02864] | 0.83 |
| EQUIP | +0.01997 | [-0.00011, +0.04038] | 0.975 |
| WEAPON | +0.00697 | [-0.00770, +0.02217] | 0.83 |
| UTIL | -0.00042 | [-0.01609, +0.01440] | 0.52 |
| ALL | +0.01677 | [-0.00315, +0.03836] | 0.95 |

VERIFIED (run).

## Results: test split, score_diff removed (sensitivity)

| Model | Test logloss | Brier | AUC |
|---|---:|---:|---:|
| BASE_NS | 0.55000 | 0.18518 | 0.7871 |
| BASE_NS+HP | 0.50998 | 0.17005 | 0.8238 |
| BASE_NS+EQUIP | 0.50338 | 0.16782 | 0.8293 |
| BASE_NS+WEAPON | 0.52195 | 0.17525 | 0.8141 |
| BASE_NS+UTIL | 0.52102 | 0.17450 | 0.8167 |
| BASE_NS+ALL | 0.49948 | 0.16643 | 0.8315 |

Gains over BASE_NS: HP +0.040 (CI +0.022 to +0.060); EQUIP +0.047 (+0.023 to +0.071); WEAPON +0.028 (+0.008 to +0.046); UTIL +0.029 (+0.007 to +0.050); ALL +0.051 (+0.027 to +0.076). VERIFIED (run).

`score_diff` alone is worth 0.033 logloss in BASE. Once the groups are in, it adds nothing (BASE+ALL 0.50006 with it, 0.49948 without it). The groups' gains shrink by about 60 to 70% when `score_diff` is included. The equipment and score signals overlap.

## Importance

- XGBoost gain, BASE+ALL, top features: `equip_diff` 25.4%, `alive_diff` 18.2%, `bomb_planted` 5.0%, `seconds_since_plant` 2.5%, `t_equip` 2.3%, `ct_armor` 2.2%. Group shares: BASE 0.318, EQUIP 0.324, HP 0.131, UTIL 0.122, WEAPON 0.105. VERIFIED (run).
- Permutation on test (5 repeats, log-loss increase, baseline 0.50006): `alive_diff` +0.0759, `equip_diff` +0.0516, `seconds_since_plant` +0.0244, `t_health` +0.0142, `t_armor` +0.0066, `t_equip` +0.0049, `score_diff` +0.0049, `ct_armor` +0.0044. Every other feature is below 0.001. Joint group permutation: EQUIP +0.078, HP +0.023, WEAPON +0.001, UTIL -0.0006. VERIFIED (run).

## Secondary: match-grouped 5-fold CV over all 150 matches (score_diff included)

| Model | Pooled OOF logloss | AUC | Gain over BASE | 95% CI |
|---|---:|---:|---:|---|
| BASE | 0.49638 | 0.8277 | - | - |
| BASE+HP | 0.48156 | 0.8420 | +0.01482 | [+0.00856, +0.02124] |
| BASE+EQUIP | 0.48155 | 0.8425 | +0.01483 | [+0.00795, +0.02141] |
| BASE+WEAPON | 0.48896 | 0.8363 | +0.00742 | [+0.00333, +0.01160] |
| BASE+UTIL | 0.49177 | 0.8342 | +0.00461 | [+0.00055, +0.00904] |
| BASE+ALL | 0.48127 | 0.8427 | +0.01511 | [+0.00788, +0.02218] |

VERIFIED (run). Fold 2 is the exception: BASE+ALL is worse than BASE there (0.52665 vs 0.52565).

## Conclusions (predictive gain, not causal)

1. Equipment difference (team equipment value) is the strongest addition. Its gain is clear without `score_diff` and borderline on the 30-match test with it.
2. Team health and armor (HP) add a similar amount.
3. Weapon categories add a small amount. Utility adds nothing on the test split and about 0.005 in CV.
4. With `score_diff` included, all groups together give +0.017 on the test split, with a CI that just includes zero. The 150-match CV gain (+0.015, CI +0.008 to +0.022) excludes zero.

## Caveats

- The test split is 30 matches. Snapshots within a round are dependent, so bootstraps resample matches.
- Early stopping uses 18 validation matches. Best iterations range from 96 to 343, so model-to-model differences partly reflect early-stopping noise.
- The test CT-win share (0.557) differs from the fit share (0.509).
- The 150-match sample is a subset of 464 eligible matches. Representativeness is UNVERIFIED (see `state_sample_extraction.md`).
- The weapon mapping for codes absent from the dictionary is verified only on this sample's days.
- `score_diff` is side-attached per the extraction notes, so the with-score results should be read as a diagnostic.
