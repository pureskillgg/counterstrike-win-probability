# Speaking notes: CS2 round win probability

Deck: https://claude.ai/artifact/Q3wq55RwipnGNHgm9UZGu6 (private until you share it from the page's Share menu).
Core talk is about 12 minutes; the chart slides after "features" are for questions. Status words: **verified** means a second, independent recompute agreed; **single run** means one run, not yet re-checked.

## If you only say three things
1. We evaluate the way Bill asked: split by match, log loss as the target, a calibration plot, and intervals that treat ticks in one round as correlated. At every second of the round, XGBoost scores 0.4916 log loss against 0.5146 for the simple alive-count model (a gain of 0.023, interval 0.020 to 0.026).
2. Our first headline, 0.4366, was too flattering: 11% of those states were already decided. Use 0.4916 (0.478 when each round counts once).
3. The data are enough for what we do now: more matches buy about 0.001, and a second independent batch matters more. And a GPU run found that the score-lead monotone constraint was costing 0.014: keeping the alive-count constraints and dropping the score one gives 0.4781 (single run; the gain's interval is -0.017 to -0.010).

## Talk track (about 12 minutes)

**1. Cover (20 s).** "The task is a probability that CT wins the round, available at every tick, built from PureSkill event data. I will show what we built, how we judge it, and what we need from you."

**2. The brief (45 s).** "Your five rules and where each stands. Run at every tick: done on a one-second grid, because the features are constant between events. Split by match: done. Log loss: the target. Calibration plot: done. Correlated ticks: every interval resamples whole matches."

**3. Data (30 s).** "1,752 matches in the 2026 tomes, 1,521 after filters: 28,683 rounds. Test is 304 matches, 5,849 rounds, about 405,000 seconds. The 2024 tome has 4,435 matches but its round-13 labels have a defect, so it is only a transfer test. Everything is registered with provenance."

**4. Evaluation (60 s).** "A random split would put ticks of one round on both sides and leak the answer. We split by match. One round has one winner, so its ticks are not independent. Counting seconds as separate rows makes the standard error 5.6 times too small on our data; every interval here resamples matches."

**5. Result (75 s).** "At every second: XGBoost 0.4916 [0.4836, 0.4993]. The simple model, logistic regression on alive difference, is 0.5146. The paired gain is 0.023 with an interval of 0.020 to 0.026. That is not huge in absolute terms, but it is well outside noise. Training on the second-by-second grid is slightly better than training on event states only (0.0018)."

**6. The correction (60 s).** "Our first headline was 0.4366. That score included the state right after the last kill, which is trivial. They were 11% of states and scored 0.0145. The other 89% score 0.4887. So quote the per-second number. The model comparisons are unaffected because both sides were scored on the same states."

**7. Calibration (60 s).** "Each point is a tenth of the test seconds. Close to the diagonal at both ends: predicted 0.016 against observed 0.015, and 0.972 against 0.977. The middle runs high for CT by up to four points. ECE is 0.010. Recalibration with a temperature or isotonic regression did not help, so we keep the raw probabilities."

**8. Where in the round (45 s).** "Early in the round almost nothing is known: log loss 0.68, near a coin flip. The model's edge over the simple model grows from 0.013 in the first ten seconds to 0.058 by 80 to 90 seconds. So any new feature should be judged by where it helps."

**9. GPU models (60 s).** "XGBoost on the GPU reproduces the CPU result within 0.0002. A tuned XGBoost and CatBoost both land near 0.478, 0.014 better. The cause is not the library: we had forced the score lead to help CT monotonically. Constraining only the alive counts gives 0.4781; constraining only the score lead gives the old 0.4916. So we drop the score constraint and keep the alive ones. ECE improves to 0.0067. This is a single run; the next step is repeating it over seeds on deduplicated folds."

**10. Is the data enough? (45 s).** "Doubling the training set from 548 to 1,095 matches buys 0.0014. A model trained on days 01 to 07 handles day 08 almost as well as its own held-out slice, but a model trained on day 08 alone does not transfer back, with a gap of 0.016. So a second batch is worth more than more of the same batch."

**11. Features (45 s).** "On a small 30-match test: equipment difference helps most, about 0.02. Health and armor help in cross-validation, about 0.015. Utility counts, positions and map add nothing measurable yet. The model has none of these today. One correction: re-tested at every second on 150 matches, bought equipment alone adds nothing detectable (+0.002), while live health and armor help (round-weighted -0.011). So the signal is what the living players hold and how healthy they are, not what was bought."

**12. Pistol rounds (60 s).** "CT wins round 1 49.7% of the time and round 13 48.3%. The pistol winner takes the next round 81% of the time, mostly because of money. For the match: the round-1 winner wins the match 57.9%, odds ratio 1.89. The 'about three times' figure we saw was a CS:GO snippet; we do not reproduce it. A fair-coin simulation gives 58.0%, so one round is worth exactly that."

**13. Data quality (45 s).** "Six problems found. The one to watch: 2024 round 13 disagrees with the bomb outcome in 434 of 605 rounds on 12 to 25 February, which is the first round after halftime. 2026 is clean on the same check."

**14. Decisions and questions (2 min).** Read the questions slide. The two that block us: which earlier PureSkill model we must beat, and what we may share given the CC BY-NC-SA licence.

## Numbers cheat sheet
- Data: 1,752 matches (18 Sep to 8 Oct 2026); kept 1,521 matches, 28,683 rounds, 234,265 event states, 1,962,000 one-second rows. 2024: 4,435 matches.
- Per second (single run; baseline recompute **verified** 0.4916): constant 0.6936; LR alive difference 0.5146; LR all features 0.5068; XGBoost 0.4916 [0.4836, 0.4993]; round-weighted 0.4781; Brier 0.1672; AUC 0.8269; ECE 0.0102.
- Cluster standard error 5.6 times the naive one.
- Decided states: 11.0% of event test states, log loss 0.0145 against 0.4887 for the rest.
- By time: 0.679 at 0 to 10 s to 0.321 at 80 to 90 s; edge over simple model 0.013 to 0.058.
- Calibration: largest gap 4.1 points (predicted 0.549, observed 0.508).
- GPU (single runs): XGBoost CUDA same settings 0.4918; alive-count constraints only 0.4781 [0.4692, 0.4864], paired -0.0136 [-0.0170, -0.0103], ECE 0.0067, AUC 0.8387; score constraint only 0.4916; no constraints 0.4781; tuned XGBoost 0.4780; CatBoost GPU depth 6 0.4775. Temperature scaling 0.4919, isotonic 0.4932 (no help).
- Other models (single runs): LightGBM CPU 0.4904 (-0.0012 [-0.0019, -0.0005]); blend of the best models 0.4773, only 0.0003 better than CatBoost alone, so a near tie. XGBoost seed-to-seed spread is about 0.0001, so differences under 0.0002 are noise. Not finished: torch MLP (environment still installing), engineered features, sklearn HistGB and the spline logistic.
- Equipment and health at every second (150 matches, grouped CV): base 0.4986; plus bought equipment 0.5007 (+0.0021, interval -0.0093 to +0.0129); plus health and armor 0.4942 (-0.0044, interval -0.0157 to +0.0058; round-weighted -0.0108 [-0.0207, -0.0023]).
- Momentum (full tome): a team wins the next round 63.7% after a win and 36.3% after a loss; with equipment controlled (150-match sample) the previous-round effect falls to 0.10 in log-odds (interval -0.25 to 0.44), so it looks like economy.
- Double eco (150 matches, unverified cut-offs): a team that ecos one round ecos the next in 11.2% of cases (5.8% after a non-eco round); the team wins 1.9% of the first eco round of a double eco and 26.9% of the second (15 to 40, n = 52); after a double eco it buys full in 25 of 48 cases. Both teams eco in the same round: 8 of 2,352 rounds.
- Armor wear: only 65 of 21,408 armored players (0.30%) had armor reach zero while alive; 0.63% fell below 50; the mean armor lost by their last alive second is 6.6.
- Fire on the planted bomb: T-thrown fire within 150 units after the plant in 0.9% of planted rounds (134 of 15,468); explosion 38.1% against 15.3%; defuse attempts under fire finish 29.7% against 59.8% (n = 37).
- Callout stack features: +0.0042 log loss worse (+0.0012 to +0.0074); no change in the first 10 s.
- Eras: kills per shot rose for all ten top weapons between Dec 2023 to Feb 2024 and Sep to Oct 2026 (AWP 0.324 to 0.360).
- Sufficiency (**verified** against the bake-off log): learning curve 0.44357 to 0.43659; batch gaps +0.002 forward (not large), +0.016 reverse (large); 2026 to 2024 transfer 0.504 against 0.437.
- Pistol: CT round 1 49.7% [47.3, 52.0]; round 13 48.3% [45.8, 50.7]; round 2 after winning round 1 81.3%; match after winning round 1 57.9% [55.3, 60.4]; both pistols 67.9%; odds ratio 1.89 [1.53, 2.33].
- Plants: planted in 46.3% of rounds, median at 45.8 s; CT wins 25.4% after a plant.
- First kill: the side with it wins about 70%; median at 16.2 s.
- 2024 round 13: 434 of 605 mismatches on 12 to 25 Feb; 0 of 180 on 1 to 7 Feb.

## Likely questions
- **Why not a random split?** Ticks of one round would be on both sides; the answer leaks.
- **Is XGBoost much better than the simple model?** 0.023 log loss at every second, interval 0.020 to 0.026. Modest in absolute terms, clearly real. Logistic regression on all features gets part of it (0.5068).
- **Why is the early round so uncertain?** Only alive counts and time are known; nothing has happened. This is where economy, positions and callout stacks should help.
- **CS:GO or CS2?** All matches are CS2 (CS2 launched 27 Sep 2023). The two tomes are different periods: the weapons differ (AWP kills per shot 0.324 then, 0.360 now), so we keep them separate. The header has a build number; patch dates are unverified.
- **Do you have weapon stats?** No weapon table or patch field. We measure weapon performance from the matches (kills per shot, damage per hit).
- **What about economy and good or bad buys?** Deferred. The model is a round probability from the live start; match probability and buy quality come later.
- **Half buy or hero buy?** Not in the model. At freeze end (150 matches, 2,352 non-pistol rounds) sides split into full 49.9%, partial 28.4%, light 11.8%, hero 5.1%, eco 4.8%; they win 58.8%, 48.7%, 37.7%, 32.9% and 14.7%. A hero side is a low-spend shape (about $10.3k, against $18.3k partial and $24.6k full). But in a round-level test, composition adds nothing beyond total equipment (+0.0049 log loss, slightly worse), nor do utility counts (+0.0081). These are holdings, not purchases (carried-over weapons count).
- **Smokes, molotovs on the bomb, plant quality?** Not in the model. The data allow plant time, planter position, nearby defenders, smoke cover and fire on the bomb; fire-on-bomb is being measured.
- **Rushes, stacks, strategies?** The raw per-tick files name a callout for every player at every tick (16 to 47 per map). A first test of site-stack counts did not help (+0.0042 worse); named plays as sequences are untested.
- **Double eco?** In the 150-match sample (cut-offs unverified): 52 team double-ecos out of 463 eco rounds; the second eco round is won 26.9% of the time, the first 1.9%. Both teams eco in one round is very rare (8 of 2,352). Not in the model.
- **A hero dies and nobody picks up the weapon?** Not handled in the current model: it counts how many are alive, not who. We tested adding each living player's weapon, health and armor (150 matches, grouped CV): log loss 0.4910 to 0.4757 (-0.0153, interval -0.0226 to -0.0082), almost all in the first 45 s (first 20 s: -0.0300). When a side's last rifle or AWP holder dies, the model with those features takes its win probability from about 0.35 to 0.13. The average drop per death is not larger for heroes, because those states start from a lower level. Weapons are rarely picked up within a second (under 1%, a lower bound). Freeze-end equipment alone made it worse (+0.008).
- **Armor wearing out mid-round?** Very rare: 0.30% of armored players reached zero armor alive. Not worth a feature on its own; health and armor together did help in cross-validation (+0.015).
- **Molotov on the planted bomb?** Rare (0.9% of planted rounds) but strong: those rounds end in an explosion 38% against 15%, and defuses under fire finish 30% against 60%.
- **Why did the score constraint hurt?** We think the score lead acts through the round number and the half, so forcing it to help CT in every case is wrong. We have the result, not the mechanism.

## What not to claim
- Do not quote 0.4366 as the model's level; quote 0.4916.
- Do not call CatBoost the winner: a tuned XGBoost without the score-lead constraint matches it (0.4781 against 0.4775).
- Do not say pistol rounds or momentum cause anything; these are associations.
- Do not use the 2024 labels as truth; round 13 is defective and the cause is unknown.
- Do not say neural networks help; none has been tested beyond the literature.
- The buy-type cut-offs ($10,000 and $20,000) are unverified.

## Still running when you read this
The torch MLP environment install, engineered features, sklearn HistGB and the spline logistic, the alive-loadout / hero-death test, and the defect and audit swarm. If any lands before you present, the deck can be refreshed in a minute.
