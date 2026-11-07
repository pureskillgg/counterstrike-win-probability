# MB: momentum across rounds

Code: scripts/momentum/MB_momentum.py. Outputs: data/viz/MB.json, data/viz/MB.svg. Intervals: match-cluster bootstrap, 1000 resamples, seed 0.

## Readout

After a team wins a round, it wins the next one 63.7% of the time (95% CI 63.2 to 64.2) against 36.3% after a loss, across 57,030 team-rounds from 1,535 complete 24-round matches. The gap is about whether you won the last round, not how long the streak is: once prev_won is in the model, the signed streak adds nothing positive (full sample coefficient -0.053 per step, CI -0.070 to -0.033, so slightly negative). Controlling for score lead and round number leaves the prev_won effect large (log-odds 1.30, OR 3.68, CI 3.35 to 4.00). In the 125-match 1 Hz sample, the same association shrinks once freeze-end equipment is in. Teams that won the previous round carry about $8.2k more equipment than their opponent at freeze end, against about -$8.2k after a loss. Adding the equipment difference moves the prev_won coefficient from 1.02 (CI 0.73 to 1.34) to 0.10 (CI -0.25 to 0.44), and the streak coefficient stays at zero. That pattern fits an economy channel better than a standalone momentum effect. Equipment is itself downstream of the previous result (loss bonus, saving), so this is an association, not a causal split between economy and psychology, and the equipment-adjusted interval still allows a moderate effect.

## Exact numbers

Population. Header max_rounds 24: 1,603 matches. Round count equals header total: 1,535 primary matches (68 excluded for a round count that differs from the header total; 18 of the primary matches have tied header totals and no decided winner; 156 are overtime, whose regulation rounds are used). Matches with a decided winner (header totals differ, round count equals their sum): 1,517 of the 1,603. Reference bake-off kept set: 1,521 (not applied). Target team-rounds: rounds 2 to min(n, 24) excluding 13, both teams.

Crude (full sample, 57,030 team-rounds). P(win | prev won) 63.7% (63.2 to 64.2); P(win | prev lost) 36.3% (35.8 to 36.8); difference 27.4 pp (26.4 to 28.5). P(win round 2 | won round 1) 83.1% (81.2 to 84.9); the reference value is 81.3% (79.6 to 83.1), so it is consistent but not reproduced exactly.

Logit, full sample (prev_won, streak, lead, round). prev_won 1.303 (1.208 to 1.385), OR 3.68 (3.35 to 4.00). streak -0.053 per step (-0.070 to -0.033), OR 0.949 (0.932 to 0.968). lead 0.026 per round point (0.017 to 0.034), OR 1.026. round 0 (see caveats). Point estimates: prev_won with lead and round only 1.088; streak with lead and round only 0.179. Within-match linear probability model (match fixed effects, point only): prev_won +0.315, streak -0.012.

Crude (1 Hz, 4,070 team-rounds). P(win | prev won) 62.5% (60.5 to 64.6); P(win | prev lost) 37.5% (35.4 to 39.5); difference 24.9 pp (21.0 to 29.1).

Logit, 1 Hz without equipment. prev_won 1.020 (0.734 to 1.343), OR 2.77. streak 0.009 (-0.056 to 0.067). lead -0.017 (-0.042 to 0.007).

Logit, 1 Hz with equipment and money (USD thousands, team minus opponent). prev_won 0.101 (-0.254 to 0.437), OR 1.11. streak 0.013 (-0.072 to 0.091). lead -0.014 (-0.040 to 0.012). eq_diff 0.069 per $1k (0.056 to 0.083), OR 1.071. money_diff -0.0096 per $1k (-0.0205 to -0.00003), OR 0.990.

Equipment by prior result (1 Hz). Mean eq_diff after a win +$8.21k, after a loss -$8.21k; gap $16.42k (15.80 to 17.03). Money gap $18.93k (17.69 to 20.38).

Streak curves (1 Hz, predicted win rate at average equipment, marginal over the observed rows). Crude observed at streak +1 / -1: 61.4% / 38.6%. Lead and round controlled, no equipment: 62.7% / 37.3% (at +5 / -5: 63.5% / 36.5%). With equipment and money at sample mean: 51.6% / 48.4% (at +5 / -5: 52.9% / 47.1%). Full table with 95% intervals is in data/viz/MB.json.

## Caveats

1. Associations only. Equipment at freeze end is partly caused by the previous round's result, so conditioning on it is a mediator adjustment, not a decomposition into economy and psychology.
2. Team-strength confounding is not controlled in the logit. The within-match LPM (match fixed effects) keeps the prev_won effect at 0.315, so the signal is not just match-level strength, but it is a linear point estimate with no interval.
3. Each round contributes two mirror-image team rows, so the rows are not independent. All intervals use match resampling, and the point estimates use every row.
4. The round-number coefficient is exactly zero in this design (corr(round, win) = 0 exactly, because the two team rows of each round have complementary outcomes). It is kept as the requested control but contributes nothing.
5. Overtime: the simple side-swap rule for overtime does not reproduce the header team totals (of 156 overtime matches, 129 pass under one parity and 27 fail under the other). Regulation (1,379 non-overtime matches) passes 100%. Only rounds up to 24 are used as targets or history, so the analysis is unaffected, but the overtime parity is unresolved.
6. The 1 Hz sample has 125 of 150 matches in the primary set (days 01 to 07). Freeze-end equipment is missing for 534 team-rounds (no freeze snapshot), and 2 team-rounds are dropped for fewer than five players. The 1 Hz intervals are wide, and the equipment-adjusted interval cannot exclude a moderate effect, so the result is "not distinguishable from zero", not "absent".
7. One primary match is a single round (1-0 in the header) and contributes no target rows, so 1,534 matches enter the regression.
8. Not checked: the bake-off kept set (1,521 matches, event-state filters) was not applied. I did not verify the reference round-2 figure on that set.
9. Colours in the SVG are series colours (crude grey, lead and round blue, equipment amber); they do not encode CT or T.
