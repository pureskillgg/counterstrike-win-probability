# Grading a round win-probability model

We grade a round win-probability model by how much it beats a fixed lookup table, the **dumb table**. For a coarse situation (who's alive, the bomb, the time left, and each side's gear), the table says how often CT went on to win the round. The grade is **GDT, the gain over the dumb table**: the table's log loss minus your model's, on the same datapoints. Positive means your model beats the table.

Your shipped model, `winprob_v6_drop_heavy`, scores **GDT 0.0180**. A new model is better if its GDT is higher than v6's on the same datapoints.

## What's in this folder

| file | what it is |
| --- | --- |
| `README.md` | These instructions. |
| `table.csv` | The dumb table: all 1,530 situations, each with the chance it gives CT. |
| `score_gdt.py` | Scores your models against the table: GDT with its interval, GDT_sit, the difference between two models, and the gain in every situation. |

## Step 1: the matches

Score on the **test days, the uploads of October 14 to 25, 2026**. In your bucket's archive copy they're `data/csds/2026/10/14/` to `data/csds/2026/10/25/`, one folder per match. The day folder is the upload day.

- Don't score on October 9 to 13. The table was built on those days.
- Don't score on any match your model trained on. If a new model trained on uploads after October 13, tell us before you score it.

Keep the matches that are:

- 5v5 (`header.is_wingman` false), with `header.max_rounds` 24;
- on one of the seven maps: de_ancient, de_anubis, de_cache, de_dust2, de_inferno, de_mirage, de_nuke.

In those matches, keep only the rounds that start 5 against 5 (5 alive on each side at second 0) and that CT or T won (`round_end.winner_team_code` 3 or 2). We got 12,566 matches this way.

## Step 2: the rows

Write one row for every second your model scores in those rounds: every whole second of the live round, from the end of the freeze time to the end of the round, before and after the plant. The fields are the ones in your serving function's feature matrix.

| column | what it holds |
| --- | --- |
| `match` | the match id (its folder name) |
| `round` | the round number |
| `second` | seconds since the round went live, rounded to a whole second (`seconds_since_round_live`) |
| `alive_ct`, `alive_t` | players alive on each side |
| `equip_ct`, `equip_t` | each side's total equipment value over its alive players (`st_equip_ct`, `st_equip_t`: the sum of `player_status.equipment_value_calc` over the side's alive players, so the price of what they hold, not their money) |
| `planted` | 1 once the bomb is planted, else 0 (`bomb_planted`) |
| `since_plant` | seconds since the plant (`seconds_since_plant`); ignored before the plant, and may be empty there |
| `y` | 1 if CT won the round, 0 if T won |
| one column per model | that model's P(CT wins the round), for example `p_v6` and `p_new` |

- Use Parquet or CSV, in one file or in a folder that holds only these rows.
- Put every model you compare in the same rows, side by side as columns. Then they're scored on the same datapoints, and the comparison is paired.
- Always include v6 (`p_v6`) as the reference.

## Step 3: score

```
pip install polars numpy
python score_gdt.py --rows <your rows> --models p_v6,p_new --out results
```

The first model in `--models` is the reference. On the full test days the script takes under a minute. It prints the results, and writes them to `results/summary.json` and `results/by_situation.csv`.

For each model it reports:

- **table** and **model**: the mean log loss of each.
- **GDT**, with its 95% interval. This is the grade.
- **rGDT**: the share of the table's loss the model removes.
- **GDT_sit**: GDT with every situation that has 500 or more test datapoints counting once. It shows a change that helps the rarer situations, such as the endgames. Report it beside GDT, never instead of it.
- The share of matches where the model beats the table.

**`p_new minus p_v6`** is the paired difference in GDT and in GDT_sit, with its interval. If the whole GDT interval is above 0, the new model is better.

**`by_situation.csv`** has one row per situation: its datapoints, the share of rounds CT won, the table's P(CT), and each model's mean P(CT), log loss and gain over the table. It shows where a change landed. For the early round alone, look at the rows with `planted` 0 and `tb` `lot`.

**Report** GDT and GDT_sit for both models, and their difference with its interval.

## How the score is computed

The script does exactly this, so you can check it or do it yourself.

1. **The situation of a second.** It's six fields: `alive_ct`, `alive_t`, `planted`, `tb`, `econ_ct` and `econ_t`.
   - `tb` is the time left, in thirds:
     - Before the plant, from `second` (thirds of the 115 s round): under 38.33 is `lot`, under 76.67 is `medium`, and the rest is `little`.
     - After the plant, from `since_plant` (thirds of the 40 s bomb): under 13.33 is `lot`, under 26.67 is `medium`, and the rest is `little`.
   - `econ_ct` and `econ_t` come from the side's equipment per alive player: `equip / alive`, or 0 when nobody is alive.
     - CT: under $2,837.50 is `low`, under $4,200 is `mid`, and $4,200 or more is `high`.
     - T: under $2,420 is `low`, under $3,900 is `mid`, and $3,900 or more is `high`.
2. **The datapoints.** For every round, and for every situation the round passes through, take one second at random from the seconds it spent there.
   - So a round counts once in each situation it reaches, however long it stays there. Counting every second instead would let the long, quiet stretches early in the round outweigh everything else.
   - The draw is seeded (`--seed`, default 0), and the rows are sorted first, so the same rows always give the same datapoints.
3. **The table's forecast.** Look up the datapoint's situation in `table.csv`, and take its `p_ct`.
4. **Log loss.** −[y ln p + (1 − y) ln(1 − p)], with the natural log (so in nats), and p clipped to between 1e-6 and 1 − 1e-6.
5. **GDT.** The mean over the datapoints of the table's log loss minus the model's.
6. **The interval.** Resample the test **matches** with replacement 1,000 times, not the datapoints, and take the 2.5th and 97.5th percentiles. A match is the unit of independence.
7. **GDT_sit.** The same, except that every situation with 500 or more test datapoints counts once (the mean of their gains). Situations with fewer get no weight.
8. **The difference between two models.** The mean of the first model's loss minus the other's, which equals GDT(other) − GDT(first). Its interval comes from the same resampling of matches.

## The table

`table.csv` has one row per situation: 1,530 of them, every situation that can occur. It was built on the uploads of October 9 to 13, 2026, with one datapoint per round per situation (1,517,301 datapoints from 7,543 matches).

| column | meaning |
| --- | --- |
| `alive_ct`, `alive_t` | players alive on each side, 0 to 5 (never both 0) |
| `planted` | 1 once the bomb is planted |
| `tb` | the time left in thirds: `lot`, `medium` or `little` |
| `econ_ct`, `econ_t` | each side's equipment per alive player: `low`, `mid` or `high` (a side with nobody alive is `low`) |
| `p_ct` | **the table's chance that CT wins the round** |
| `level` | where `p_ct` comes from: `full situation`; or, when the situation had fewer than 30 training datapoints, `without the economy`, then `without the time`, then `the base rate` |
| `n_train`, `ct_wins_train` | the training datapoints in the situation itself, and how many of them CT won |
| `n_level` | the training datapoints at the level `p_ct` comes from |

`p_ct` is (wins + 1) / (n + 2) at that level. 1,093 situations have enough data of their own. Of the rest, 254 back off to the situation without the economy, 48 to the situation without the time, and 135 to the base rate. Those 135 can't happen in a live second, because the round is already over: a side with nobody alive before the plant, or no CT alive after it.

## The reference: v6

We ran v6 through this script on our own rows of the test days (12,566 matches, 2,508,752 datapoints):

| | |
| --- | --- |
| table log loss | 0.4474 |
| model log loss | 0.4294 |
| GDT | 0.0180 (0.0175 to 0.0184) |
| rGDT | 4.0% |
| GDT_sit | 0.0217 (0.0210 to 0.0223) |
| matches where v6 beats the table | 76.9% |

v6 on your own rows should land close to these. If it doesn't, check the rows first: the match and round filters, the seconds, and the equipment values.

## Rules

- **No score lead as an input, ever,** even when it improves the score. Each round is treated as a round on its own.
- **The table is fixed.** Don't rebuild or retune it to suit a model. If you think it should change, ask us first.
- **Same datapoints for every model you compare:** the same rows and the same seed.
- **Test days only:** October 14 to 25, and never matches a model trained on.
