# Internal reading guide: win probability, 27 Oct 2026

For you, to read before any deck or report. Plain words first; the full tables are linked at the end. Times are Central. Numbers are log loss unless said otherwise: lower is better, and always guessing 50/50 scores 0.693.

## 1. The short story

- **Goal.** Estimate the chance the Counter-Terrorist side wins the round, every second, from the state of the round.
- **Baseline.** A tree model (XGBoost) on 57 inputs. Validation log loss **0.4533**.
- **What helped.** Distance to the two bomb sites. Straight-line: **−0.0028** on validation, every training seed passes. Walking distance on the map graph: **−0.0026**, but on the test set it did not beat straight-line.
- **Sequence model (GRU).** Reads the last 20 seconds of a round in order. **−0.0022** vs the baseline on validation, every seed passes. It gains more as the number of training matches grows (see section 4).
- **Best so far.** 50/50 blend of the sequence model and the distance model: about **−0.0040** vs baseline on validation (computed from −0.0011 and −0.0028). Not yet read on any clean test data.
- **Dataset.** 23,705 matches with data rows on seven maps: training 19,478, validation 1,168, test 3,059 (687 of the test matches are on Mirage). A further 3,417 matches from 22 to 25 Oct are in the tomes but not yet integrated into the model tables, so no run uses them. The fresh holdout (653 matches) is part of those 3,417.
- **Biggest problem.** The test set has been looked at many times (section 3).

## 2. The building blocks, and what each one did

Each block: what it is, how we used it, result, and why. Validation unless marked test.

| Block | What it is | What we did | Result |
|---|---|---|---|
| Tree model (XGBoost), 57 inputs | Many small if-then rules, combined | The baseline. Settings fixed in the frozen recipe | 0.4533 |
| Distance to bomb sites (Xenopoulos et al. 2020) | Nearest living player's distance to each site | Straight-line distance; site centres fitted on training matches only | −0.0028 (passes) |
| Walking distance (navigation mesh, awpy data) | Shortest path on the map's walkable graph | Area centroids plus Dijkstra; our own method, not from a paper | −0.0026 (passes). Test: not better than straight-line |
| Plain neural network (Xenopoulos and Silva 2022, ESTA) | A network on the 57 inputs | Pre-registered 26 Oct | +0.0008, fails. Overfits after 1–2 epochs. Same on a rented GPU (+0.00077) |
| Word2vec position embeddings (Chahrour 2023) | Turn player positions into vectors | Replicated on the paper's 200-match protocol and on our recipe | Paper's +0.027 does not reproduce: −0.009 on a 4-feature base, +0.0001 on the 57 |
| Learned set embedding (our Stage 2) | Learned summary of the round's player positions | Trained on next-state prediction | −0.00054 on the baseline (passes), but −0.00009 once distances are in (fails) |
| Player-to-player attention (Xenopoulos and Silva 2022, GNN) | Players attend to each other | Added to the sequence model | +0.00015, fails |
| THGNN-lite (Szmida and Toka) | Graph network over regions | Graph from the navigation mesh vs graph from k-means | Test: +0.00003, not adopted |
| Sequence model (GRU) | Reads the last 20 seconds in order | Pre-registered 26 Oct against the tree model | −0.0022, passes every seed |
| Blend, 50/50 (GRU and distance model) | Average of two models | Fixed weight, pre-registered | −0.0011 vs distance model, −0.0018 vs GRU. Passes |
| TabPFN, TabICL (tabular foundation models) | Pretrained table models | Run against XGBoost, no tuning, 14 Oct | Worse by 0.009 to 0.013 |
| Map control (reachability) | Share of map each side reaches first | Pre-registered | Wrong direction on validation (+0.00005) |
| Event counts (kills, utility, bomb timers) | Recent events | Pre-registered, two versions | Missed the bar by a little (−0.00044 vs −0.0005) |
| Per-weapon run speed in defuse timing | Speed from the weapon held | Pre-registered | −0.00026, fails |
| Death in next 3 seconds | Engagement signal | Pre-registered | Adds nothing, fails |

Bar for "helps": at least 0.0005 better on each of three training seeds, with a confidence range that stays below zero.

## 3. Why a sequence model, and why 20 seconds

**Why a sequence model.**
1. **14 Oct** (pin f1a15aff): kept XGBoost. Research at the time found no sequence, set or additive network beating a tuned tree model on a match-grouped split.
2. **22 Oct** (pin 29815cdf, an assistant proposal, no owner approval pinned): plan for a causal temporal model over the last 20 to 40 seconds, plus a set encoder.
3. **25 Oct** (pin 2e4130fe, test set): the earlier XGBoost edge over the GRU came from training scope (the tree model had more maps), not from the model. On matched scope the GRU was better, by 0.00166.
4. **26 Oct** (pin 62e3f1d6, validation): pre-registered the test "does the GRU beat XGBoost on validation, per seed". It passed on every seed.

So the first evidence that the GRU helps came from a test-set comparison (step 3). The validation confirmation came after (step 4).

**Why 20 seconds.** It is the lower end of the 20 to 40 second range in the 22 Oct plan, and it is the code default. No pinned test compares 20 with 40 or any other window. It was not tuned.

## 4. Learning curve (more matches)

Run on 27 Oct. Validation log loss, one training seed:

| Training matches | Tree model | Sequence model | Sequence minus tree |
|---|---|---|---|
| 25% (4,870) | 0.45422 | 0.45459 | +0.00037 (sequence worse) |
| 50% (9,739) | 0.45371 | 0.45325 | −0.00046 (sequence better) |
| 100% (19,478) | 0.45334 | 0.45138 | −0.00196 (sequence better) |

The sequence model's advantage grows with more matches. Caveats:
- One seed. The sequence model's epoch was chosen on validation.
- **Not pinned yet.** The pre-registration (pin dc66a027) says the training set is 21,468 matches. The runs used 19,478 matches that have data rows. The gap is 1,990 matches. The pin needs a correction, and the reason for the gap needs checking.

## 5. The test set: the problem and what is being done

**The problem.**
- The test set is 687 matches on one map (Mirage). At least six result files name that count, and more read its rows.
- Many choices were checked on it before validation-only rules existed. Rules that restrict decisions to validation are dated from 26 Oct.
- Examples: distance features and their site centres (25 to 26 Oct), walking vs straight-line (25 Oct), defuse timing (25 Oct), word2vec (25 Oct), THGNN-lite graph (25 to 26 Oct, test).
- Site centres were fitted using test matches too. The straight-line gain was re-checked on validation with training-only centres (pin ee61ba3a) and held (−0.0028).
- Candidate B (the blend) was chosen after the test numbers for its two parts were already on record.

**What is being done.**
1. A fresh holdout of **653 matches** (from 22 to 25 Oct) is built and its features checked. Never scored.
2. From 26 Oct, decisions are pinned as validation-only, with the rule written before each run.
3. The one-time read of candidate B waits for an audit review and your go. Pin 56e6e642 says the rule must be re-pinned before any read, because the audit found earlier test reads that changed choices in B and A.
4. Still to do: validation-only re-checks of walking vs straight-line, the defuse timing, and the THGNN graph.
5. The reproduction check for the saved tree-model predictions needs the GPU, which another project is using.

Full audit: `audit_review_2026-10-27.md` in this folder.

## 6. Where the navigation mesh and THGNN came from

- **Navigation mesh.** awpy-data release 2000935, published **23 Oct 2026** (2026, not 2025). Owner approved the download on 25 Oct for internal research only. The files derive from Valve game data, so nothing ships without a licence review (pin c432a055).
- **Walking distance idea.** Xenopoulos, Doraiswamy and Silva 2020 (arXiv 2011.01324, Section III), the navigation-mesh graph distance. The idea predates THGNN. Our walking method is our own version.
- **THGNN-lite.** Szmida and Toka's framework. Our version is from the 22 Oct plan. The navigation-mesh graph was one variant, tested against the k-means graph on the **test set** (pins 6787d3b7, 4edc74c3). Both assistant proposals, no owner approval pinned.
- **Year of the THGNN paper.** Paper ID says 2025; the listing says 2026. Still open.

## 7. Decisions: who approved what

- **Owner approvals in pins:** 2024 kept separate and its rules (a7b482f6, 81e18a80, afa62544, 6923f58c); round-state window (5a1dec05); navigation-mesh download (c432a055); run decision for the attention test (da8ee067); 22–25 Oct extension and cut-off (5f8dea45, b2f4a9d7); candidate B choice (56e6e642, read still gated); 7 maps (9ec38dc9); score lead is not a feature (ab37b667).
- **Assistant proposals with no owner approval pinned:** almost all tests (sanity check, attention, event counts, defuse speed, map control, contested rounds, sequence model, walking distance, blend, learning curve), the Stage 2 plan, THGNN-lite, the sequence-model choice.
- **Open:** split ratio (2569aac7); whether 2024 goes into any pre-registered test (recommended no for the final read); which of the 124 unpinned result files to pin, retire or label descriptive (88 of them touch test numbers).

## 8. Known errors and gaps to fix

- Deck slide 17 had no MLP or word2vec rows. Fixed in the draft; the 27 Oct sequence result is now in the table.
- Deck slide 19 (did-not-work) labels the THGNN-lite graph result as test. To correct: the label is right, but the slide needs to say the decision used test numbers.
- Learning-curve pin needs an outcome and the 19,478 vs 21,468 correction.
- Huang et al. 2026 (defuse race): not found online. The contested-round definition and the 250 units per second speed come from it.
- Chahrour 2023: no paper found. The replication is of a repo and a manuscript page.

## 9. Full tables

- `pins_register.md`: all 79 pins, with who decided and the verdict.
- `papers_status.md`: the 27 numbered papers and what was tested from each.
- `unpinned_results.md` and `unpinned_test_ledger.md`: the 124 result files without pins, and which touch test numbers.
- `audit_review_2026-10-27.md`: the test-set audit and the fixes.
- `README.md`: the register index.
