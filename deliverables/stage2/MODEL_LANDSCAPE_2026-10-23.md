# CS2 model landscape, 2026-10-23

Two read-only research workflows (wf_5e3b6bcf-dee: CS2, esports and foundation models; wf_17863350-b4a: other team shooters and spatial or team-value methods) with adversarial verification of every licence, weights and reported-number claim. Numbers in section 1 are ours, on the same test seconds. Numbers in sections 2 and 3 are the sources' own, on their own data, and are not comparable with ours unless stated.

## 1. Paired metrics on the same de_mirage test seconds

Test set: de_mirage, 687 matches. Baseline: winprob_v6_drop_heavy (57 inputs).

| Temporal model | Seconds | Matches | Baseline log loss | Model log loss | Diff log loss [95% CI] | Diff Brier [95% CI] | Diff AUC [95% CI] | Baseline AUC | Model AUC | Baseline ECE | Model ECE | Baseline acc | Model acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| de_mirage_gat_loadout_gru | 865,974 | 687 | 0.4512 | 0.4527 | +0.0015 [+0.0005, +0.0026] | +0.0004 [-0.0000, +0.0009] | -0.0007 [-0.0015, +0.0001] | 0.8607 | 0.8601 | 0.0051 | 0.0125 | 0.7667 | 0.7641 |
| de_mirage_thgnn_lite | 865,974 | 687 | 0.4512 | 0.4680 | +0.0168 [+0.0135, +0.0206] | +0.0040 [+0.0032, +0.0050] | -0.0064 [-0.0081, -0.0048] | 0.8607 | 0.8544 | 0.0051 | 0.0243 | 0.7667 | 0.7608 |
| de_mirage_thgnn_lite_bestval | 865,974 | 687 | 0.4512 | 0.4553 | +0.0041 [+0.0020, +0.0062] | +0.0010 [+0.0002, +0.0017] | -0.0018 [-0.0031, -0.0005] | 0.8607 | 0.8589 | 0.0051 | 0.0071 | 0.7667 | 0.7646 |

Bootstrap: 1000 match-cluster resamples, seed 0. Log loss and Brier use the same match draws as compare_paired_de_mirage.py, which reproduces its interval. AUC is a weighted rank statistic, checked against sklearn on the full sample. Positive log loss or Brier difference means the temporal model is worse; a negative AUC difference means it ranks worse.
Reading: the 57-input baseline is the best-calibrated and the most accurate model on these seconds. Every temporal model loses on AUC and calibration as well as log loss. The gap is largest for the graph model, and the GRU with attention is closest, with its AUC and Brier intervals touching zero.

## 2. CS2 and esports research (wf_5e3b6bcf-dee)

### Summary

Synthesis for the CS2 de_mirage win-probability comparison. Baseline to beat: 57-input XGBoost, test log loss 0.4512 (95% CI 0.4454 to 0.4574, match bootstrap) [brief]. The GRU with attention is +0.0015 worse on paired seconds [brief]. The owner's gate: paired log-loss delta <= 0 with upper CI bound <= 0. No build is promoted yet; every expected gain is an untested hypothesis. The repo (Henry1145141919810/CS2_Win_Probability_Model) is a comparator only: de_inferno, 220 training matches, its log losses are not comparable to ours, and it has no licence, so its code must be re-implemented rather than copied. Verified repo facts [repo appendix_metrics_intime.csv and holdout_2026.csv, via verifier]: logistic economy-only A AUC 0.8465 in-time; EB2 0.8508; EFB2 0.8519 in-time, falling to 0.8236 out-of-time. EFB2 uses HLTV-derived firepower, which the owner excludes. EB2, the repo's owner-compatible recommendation, includes BOMB_LIVE_COLS, which the research flags as a possible outcome leak (unverified in code). Verified foundation candidates: TabICLv2 (BSD-3 code; weights metadata bsd-3-clause at huggingface jingang/TabICL; fine-tuning class documented), TabDPT (apache-2.0 card metadata), TabFM (non-commercial, comparator only), TimesFM 2.5 (Apache-2.0, LoRA example), Chronos-Bolt Base (Apache-2.0), Lag-Llama (Apache-2.0), Sundial-base-128m (Apache-2.0, zero-shot only). Refuted and held out: TabPFN-3.5, TabPFN-v2, Mitra, LimiX-2, Chronos-2, TimesFM 3.0, Toto-Open-Base-1.0, TiRex, CS-NET v4, TD-MPC2, the sports-GNN 9%/20% headline, the VALORANT significance claim, and the DOTA 2 accuracy claim. Ranking reversals: if build 2 beats build 1 by >= 0.0010 on validation, the shared-weight encoder moves to rank 1; if build 3 beats build 1 by >= 0.0010 within the latency budget, TabICLv2 moves to rank 1.

### Ranked builds (untested; each has a reversal rule set before the test)

### 1. Lag and difference features on the 57-input XGBoost baseline
- Why: Cleanest paired test of whether tick-to-tick information adds to the baseline; TreeSHAP gives additive per-tick contributions. Gain expected small: the GRU with 20 s attention is +0.0015 worse [brief], which suggests short history adds little beyond the current state. Not shared-weight; a test of information, not the final architecture.
- Data needed: Our 1 Hz arrays: alive, health, armour, money, equipment value, speed, yaw sin/cos change; bomb state and time-left from the snapshot table; deltas at 1, 3, 5, 10 s from past seconds only. No HLTV, rating, KAST or per-player history; no round-end or bomb-outcome fields.
- Cost: Lowest: minutes per fit on CPU or GPU (estimate, not measured); inference at baseline cost.
- First experiment: Add deltas with baseline hyperparameters, then one validation-only tuning pass. Report paired delta log loss vs baseline on validation, then test once, with match-cluster bootstrap B=2000.
- Reversal threshold (stated before the test): Keep only if validation delta <= -0.0005 with bootstrap upper bound < 0; drop if delta > +0.0005 or within +/-0.0005 after tuning. Reversal: if build 2 without lags beats baseline by >= 0.0010 on validation, build 2 moves to rank 1.

### 2. Shared-weight per-slot encoder (MLP or GAT-style attention) on the same inputs
- Why: Directly meets the owner's shared-weight and reduced-dimensionality target; per-slot tokens allow per-player attribution. Reference: the repo's GAT-style network (per-second, 11 fields, no time window; in-time AUC 0.8465, unverified). Risk: the GRU lost by +0.0015, so extra capacity is not evidence by itself.
- Data needed: Per-slot 12-channel loadout, x/y/z, yaw sin/cos, alive; build-1 deltas; bomb state and time-left. Owner-compatible inputs only.
- Cost: Low to moderate: minutes to tens of minutes per fit on one GPU (estimate).
- First experiment: 3 seeds; 2 layers; dropout 0.1 to 0.3; weight decay; early stop on validation log loss. Run the slot-permutation equivariance test before any per-player attribution is shown.
- Reversal threshold (stated before the test): Drop if validation delta > +0.0010 without lags, or if the 3-seed spread > 0.0015. Reversal: if it beats build 1 by >= 0.0010, it moves to rank 1.

### 3. TabICLv2 as a tunable tabular foundation model on compact features with whole-match context
- Why: The owner's tunable-foundation candidate. Confirmed: BSD-3 code; weights metadata bsd-3-clause at huggingface jingang/TabICL; FinetunedTabICLClassifier documented [verifier]. Self-reported: beats tuned GBTs on about 80% of TabArena datasets (README; per-dataset figures not checked). Binary TabArena average rank 4.43 vs AutoGluon 1.5 extreme 3.83 (arXiv 2602.11139 prose, verified). No evidence yet on 1 Hz autocorrelated game states. The feature cap (about 100 columns) and row range come from a research FAQ read, not verified. TabPFN-3.5 was considered and dropped: refuted.
- Data needed: Compact set of at most about 100 columns (baseline plus selected deltas). Context drawn by whole matches from the train split only; test labels never enter.
- Cost: Unknown; GPU in-context inference; latency unmeasured. README claim of 50K x 100 in under 10 s on one H100 is self-reported and unverified.
- First experiment: 50k to 100k context rows sampled by round across train matches; predict validation in chunks of about 1,000; time one average test match (about 1,260 states, from 865,974 s over 687 matches [brief]).
- Reversal threshold (stated before the test): Drop if validation delta > +0.0010 vs baseline, if one average match takes > 1 s (proposed; owner to confirm), or if fine-tuning gains < 0.0005 over zero-shot. Reversal: if build 1 or 2 shows gain >= 0.0010, TabICLv2 stays a comparator only.

### 4. Causal temporal CNN, 20 s vs 40 s window, per-slot shared encoder
- Why: Retests the sequence family with a different inductive bias from the GRU that lost. The repo TCN is team-aggregate with a 160 s window and reports in-time AUC 0.8488 (unverified); a per-slot version is a new model. Bai et al. 2018 abstract claims TCNs beat LSTMs on sequence tasks (abstract only; not CS2 evidence). 40 s matches the bomb timer (from CS2 guides, unverified).
- Data needed: Build-2 inputs over a past-only 20 s or 40 s window.
- Cost: Moderate: convolutions run in parallel over time; tens of minutes per run (estimate).
- First experiment: Causal TCN at 20 s and 40 s, 3 seeds each. First run the future-perturbation test: changing the input at t+1 must not move the output at t.
- Reversal threshold (stated before the test): Drop if the best configuration is worse than +0.0010 vs baseline, or if 40 s is not better than 20 s by 0.0005. Moves up if 40 s beats 20 s by >= 0.0015 and build 2 fails.

### 5. Monotone-constrained XGBoost (explainability variant)
- Why: Explainability, not gain: fixed-sign per-feature attributions per tick. Expected log-loss cost small; XGBoost docs say monotone constraints can give shallow trees (mitigate with larger max_bin; unverified here). The sign priors are our assumption.
- Data needed: Baseline 57 inputs. Sign priors on 6 to 8 features: CT alive (+), T alive (-), CT equipment (+), CT kits (+), CT health (+), T bomb carrier (-), defuse-race margin (+). Not on time-left.
- Cost: Same as baseline.
- First experiment: Constrained vs unconstrained on validation; count SHAP sign violations of the unconstrained model on validation seconds.
- Reversal threshold (stated before the test): Drop if validation delta is worse than +0.0010, or if the unconstrained model violates the priors on fewer than 0.5% of validation seconds.

### 6. Discrete-time hazard model for plant, defuse and explosion
- Why: Targets post-plant structure the baseline learns only implicitly; each tick's hazard change explains the probability change. The repo post-plant log loss 0.187 to 0.172 (XGB, verified in paper section 7.4) cannot be used as evidence because EB2 may contain outcome-revealing bomb state (unverified). Post-plant seconds are a minority, so the overall gain is small.
- Data needed: Post-plant state: site, seconds since plant against the 40 s timer, CT kits, CT positions relative to bomb, alive counts, economy; defuse event ticks from tomes. De_mirage nav-path distances are not in hand (repo uses Euclidean fallback); a de_mirage nav mesh is needed.
- Cost: Low on CPU; the nav-mesh build is the real cost.
- First experiment: Fit a defuse-success cloglog hazard on train post-plant seconds; add hazard and race margin to the baseline; compare validation log loss post-plant and overall.
- Reversal threshold (stated before the test): Drop if overall validation gain < 0.0005 or post-plant gain < 0.003, or if the hazard does not beat a constant-by-site baseline on validation.

### 7. Forward-filter latent-regime calibration layer (4 to 6 state HMM, logistic emissions)
- Why: Targets calibration rather than discrimination. Repo contested AUC is about 0.58 to 0.60 while pooled AUC is near 0.85 (appendix CSV, verified: EB2 cAUC 0.5963 logistic, 0.5861 XGB, 0.5898 LightGBM). No CS2 HMM precedent found. Forward filtering only; smoothing would use future seconds.
- Data needed: Causal features only: economy, alive counts, bomb state, time-left; baseline logit.
- Cost: CPU minutes.
- First experiment: 4-state forward filter with per-state logistic emissions fit on train; isotonic and Platt on validation; log loss and ECE vs baseline; per-round PIT.
- Reversal threshold (stated before the test): Drop if validation log loss is worse than baseline by > +0.0010, or if ECE gain is < 0.005 after isotonic calibration.

### 8. Out-of-fold stack of GBT, shared-weight encoder and TCN outputs
- Why: Gain only if base errors are diverse. The repo's own CV reports its ensemble members statistically indistinguishable (repo claim, unverified); soft-vote AUC 0.8531 is an unverified summary figure. A gain of 0.0005 to 0.002 is a research guess, not an estimate from data.
- Data needed: Grouped out-of-fold predictions by match inside train; the stacker sees no raw features.
- Cost: Sum of base-model costs; latency is additive.
- First experiment: Logistic stacker on baseline XGBoost, the best build-2 encoder and the best build-4 TCN; weights from train OOF; score once on validation.
- Reversal threshold (stated before the test): Keep only if the stack beats the best single model by >= 0.0005 on validation with CI excluding zero. Drop if worse than the best single model by > 0.0002, or if latency exceeds budget.

### 9. THGNN-lite controls: shuffled-edge null and no-graph encoder
- Why: Graph structure has no shown gain yet. Without controls, any THGNN-lite gain cannot be attributed to graph structure rather than shared encoding. The model is already training, so the marginal cost is two control runs.
- Data needed: 10 player nodes per second; 48 data-driven map nodes built from train matches only; a bomb node; edges from positions at t only.
- Cost: Moderate; controls only.
- First experiment: THGNN-lite vs the same architecture with edges shuffled within each second vs build 2 without graph; validation paired differences.
- Reversal threshold (stated before the test): Drop the graph variant if worse than +0.0010 vs baseline, or if its gain over the shuffled-edge null is < 0.0005.

### 10. Time-series foundation probes: TimesFM 2.5, Chronos-Bolt Base, Lag-Llama
- Why: Pretrained dynamics might add information, but forecast targets differ from win probability and the window is short. Confirmed: TimesFM 2.5 Apache-2.0 with LoRA example; Chronos-Bolt Apache-2.0; Lag-Llama Apache-2.0 with finetune script. Lag-Llama context 32 is research-sourced (unverified). Chronos-2 is held (refuted on fine-tuning); its covariate claims are unverified.
- Data needed: Per-channel state series over a 20 s past window; pretrained weights are the only external input.
- Cost: GPU or CPU inference per channel; per-match cost unmeasured.
- First experiment: Next-second forecast error as a diagnostic; then frozen embeddings into a logistic model compared with baseline.
- Reversal threshold (stated before the test): Drop if there is no paired gain >= 0.0005 over baseline, or if per-match cost exceeds 2x baseline (placeholder).

### 11. Event-token transformer over tome events
- Why: Could capture multi-kills, trades and flashed defusers that 1 Hz snapshots miss. Precedent: CS:GO action valuation (arXiv 2011.01324 abstract; numbers unverified). Highest leakage risk: round_end, bomb_defused, bomb_exploded and any event after t reveal the outcome.
- Data needed: Tome player_death, player_hurt, bomb events, grenade_state, molotov_state before t only. Tokens: type, actor and victim slot (shared weights), time gap, position, baseline logit.
- Cost: Highest: event-table build; a few GPU-hours per run (estimate).
- First experiment: 20 train matches; last 20 s of events; baseline logit as input; missed-play check on event-dense seconds.
- Reversal threshold (stated before the test): Drop if the pilot is worse than +0.0010 vs baseline, or its gain on event-dense seconds is < 0.0010, or if the future-event perturbation test changes any output.

### Evaluation suite

- 1 paired_log_loss_delta: LL = -mean[y log p + (1-y) log(1-p)], p clipped to [1e-4, 1-1e-4], y = 1 if CT wins. Delta = LL_cand - LL_base on identical (match, round, second) keys; report per second, round-weighted and match-weighted; 95% CI from 2,000 match-cluster resamples applied to both models. Gate: point delta <= 0 and upper bound <= 0. Publishable: delta <= -0.002 and upper bound < 0. Optional pre-registered non-inferiority: upper bound <= +0.001. Clip sensitivity at 1e-3. Status: proposed, owner to ratify.
- 2 paired_brier_murphy: delta Brier with the same bootstrap; Murphy decomposition on 15 equal-mass bins. Gate: upper bound of delta BS <= 0; RES_cand >= RES_base; REL_cand <= REL_base + 0.001. Proposed.
- 3 calibration: pooled ECE with 15 adaptive bins <= 0.01; per-slice ECE <= 0.02; recalibration slope in [0.95, 1.05]; |mean(p) - mean(y)| <= 0.005. Equal-width 10-bin ECE as sensitivity. Proposed.
- 4 time_sliced_discrimination: log loss and AUROC by seconds since round start (0-15, 15-45, 45-75, 75+) and by phase (pre-plant; post-plant fuse > 30 s, 10-30 s, < 10 s); contested AUROC on seconds where baseline p is in [0.35, 0.65], subset chosen on the baseline. Gate: delta AUROC >= -0.003 in every slice and >= 0 on contested states. Proposed.
- 5 reference_floors: base-rate, time-only logistic and economy-and-alive logistic, fit on train only. Candidate must beat the economy-and-alive logistic by >= 0.003 nats per second. Proposed.
- 6 missed_play_rate: per event type on identical events (see missed_play_definition). Candidate rate <= baseline for each type; absolute rate <= 10% for kill, multi-kill, plant and defuse. Proposed.
- 7 false_move_rate_quiet: flagged quiet windows / all quiet windows <= baseline and <= 2%. Proposed.
- 8 wrong_way_move_rate: <= 2% of eligible directional events and not above baseline. Proposed.
- 9 outcome_directed_move_rate: clutch success >= baseline and >= 0.70. Retake needs de_mirage site polygons, not in brief; blocked until supplied. Proposed.
- 10 volatility_ratio (martingale QV ratio on each round's 1 Hz p series, terminal y): VR in [0.90, 1.15] and not above baseline; VR > 1.30 flagged as over-reaction. Proposed.
- 11 reversal_chatter: quiet pre-plant sign reversals per minute with both |delta p| >= 0.005; <= baseline and <= 0.5 per minute; also report thresholds 0.002 and 0.01. Proposed.
- 12 direction_agreement_after_events: kill, plant and defuse agreement >= 0.90 at W = 3 s; magnitude-weighted ratio >= 0.80; CT-favoring and T-favoring reported separately. Proposed.
- 13 reaction_latency: median <= 1 s, 90th percentile <= 2 s, non-response <= 5% for kill, plant and defuse; median not more than 0.5 s slower than baseline. The owner's latency target is undefined; inference latency is item 24. Proposed.
- 14 change_completeness_residual (per-player additive attributions over 10 slots): median RR <= 0.01; 95th percentile RR <= 0.05; median |R_t| <= 0.005. Proposed.
- 15 attribution_event_localization: top-slot hit rate >= 0.60 and >= 2x alive-adjusted chance at kills, plants and defuses. Proposed.
- 16 coalition_deletion_insertion: exact over 1,024 slot subsets with 32 same-phase training backgrounds; F_method - F_random >= 0.10 and >= 0.80 of greedy oracle; also magnitude-weighted curves. Proposed.
- 17 exact_shapley_rank_agreement: median Spearman rho >= 0.70 over alive slots. Proposed.
- 18 gnn_fidelity (THGNN-lite): median fidelity+ minus fidelity- >= 0.05 at k = 3; node and edge masking reported separately. Proposed.
- 19 team_permutation_equivariance: max |delta p| <= 1e-6 for symmetric designs; <= 0.003 for slot-aware designs. Proposed.
- 20 subgroup_paired_delta: side, half, equipment quintile, buy type, post-plant site, pre/post plant. No subgroup with point delta LL > +0.005 and Holm-adjusted upper bound > +0.01 with >= 30 matches. Proposed.
- 21 causal_invariance: max |delta p_t| <= 1e-6 when states t+1 to t+10 are replaced; any violation fails. Audit the feature build for round-level or outcome columns. Proposed.
- 22 noise_timeshift: jitter x and y by N(0, 1% of map extent) and yaw by N(0, 2 deg): mean |delta p| <= 0.005 and <= 1.2x baseline; time shift of +/-1 s: delta LL <= 1.2x baseline. Proposed.
- 23 date_blocked_holdout: retrain on earliest 70% of the 43 days; delta LL <= 0 and no more than 0.002 worse than the match split. Proposed.
- 24 compute_cost_per_match: ms per state at batch 1 on one CPU thread; per match at about 1,260 states (865,974 / 687 [brief]). Placeholder gate: <= 2x baseline per-state latency; owner to set.

### Missed-play definition (proposal, not yet calibrated against the baseline)

Sources: T = tome events (player_death, bomb plant, defuse and explode, grenade_state, molotov_state; tick and slots); S = 1 Hz states (alive, loadout incl. defuser, defusing, positions, bomb state); R = round winner. Tome-tick-to-second mapping is floor, fixed in advance and reported. Events: (1) kill: T player_death, cross-checked with S alive going 1 to 0 within 1 s; favored = attacker side; W = 3 s; delta = 0.02; eligible if a in [0.05, 0.95]. (2) multi-kill: two or more T deaths by one attacker within 5 s; favored = attacker; W = 5 s; delta = 0.03. (3) entry: first T death of the round; favored = attacker; W = 5 s; delta = 0.02. (4) plant: T plant, cross-checked with S bomb-state change; favored T; W = 3 s; delta = 0.03. (5) defuse completion: T, cross-checked with S defusing channel and bomb state; favored CT; W = 3 s; delta = 0.03. (6) last-alive clutch onset: S only (own team 1 alive, opponents >= 2); scored with R; eligible if a in [0.10, 0.90]; W = min(20 s, time to round end); delta = 0.03. (7) retake: S positions inside the planted site polygon after plant, with CT win in R; eligible if a in [0.10, 0.90]; W = 10 s; delta = 0.03; blocked until polygons are supplied. Notation: d = +1 for CT-favored, -1 for T-favored; a = p at second floor(tau) - 1; M = max over tau in [floor(tau), floor(tau) + W] of d * (p_tau - a). Event missed iff M < delta. Per type, report missed rate for candidate and baseline on identical events, with match-cluster CI, and shared misses. Materiality cells (type x a-quintile) are fixed from training outcomes only: a cell is material if the mean of d * (y - a) over training events >= delta; the test set never sets materiality. Sensitivity: drop overlapping windows (multi-kills, trades). Companion checks: false move (quiet window = pre-plant second with no T death, bomb, grenade or molotov event in [t-2, t+10], no S alive or loadout change, no plant by t+10; flagged if |p(t+5) - p(t)| >= 0.02 and |p(t+10) - p(t)| <= 0.5 |p(t+5) - p(t)|); wrong-way (directional events; min over [floor(tau), floor(tau) + 3] of d * (p - a) <= -0.02, excluding events with another directional event in [tau - 2, tau + 6]). Thresholds are evaluation_suite items 6 to 8. Tome schema (event names, tick rate, slot fields, grenade detonation times) is not verified.

### Repo comparison plan (Henry1145141919810/CS2_Win_Probability_Model)

Compare on our split only: 72/8/20 by match; touch-once test of 687 matches and 865,974 s [brief]. Use 5-fold GroupKFold by match inside train for out-of-fold selection, and 2,000 match-cluster bootstrap resamples for paired log-loss deltas against the 57-input XGBoost baseline (0.4512) and the GRU (+0.0015). Re-implement the repo models; do not copy its code (no licence; reimplementing a described method needs an owner or legal decision). Comparators: logistic economy-only (repo A-like) and logistic with money; LightGBM, CatBoost and RF on owner-compatible subsets (drop every FIREPOWER and HLTV column; check bomb_state for post-defuse or post-detonate values before use); unweighted soft vote of logistic, XGBoost, LightGBM and CatBoost; cross-fitted logistic stack; causal TCN at 20 s and 40 s; Transformer; GAT-style per-slot network (11 owner-compatible fields). Foundation: TabICLv2 and TabDPT on at most about 100 compact features; TimesFM 2.5, Chronos-Bolt and Lag-Llama as zero-shot probes. Do not compare against repo log losses (de_inferno, not comparable). Select on validation only. Report log loss, Brier, ECE (15 adaptive bins), calibration slope and intercept, contested AUC, per-slice log loss, missed-play rates, false-move rates, latency and cost together. The repo's claim that its classical and deep models tie in-time is unverified; expect the test to show whether any build clears the gate, not to reproduce the repo's ranking.

### Foundation-model verdicts

- TabICLv2: use as build 3 (licence and fine-tuning confirmed; benchmark self-reported; feature cap unverified).
- TabDPT: comparator (Apache-2.0 card metadata; fine-tuning undocumented; Table 1 numbers confirmed as author-reported).
- TabFM: comparator only (non-commercial weights; README bars commercial and production use; 100-row default context).
- TabPFN-3.5, TabPFN-v2, Mitra, LimiX-2: held out (refuted on one or more claims; do not promote).
- TimesFM 2.5: probe in build 10 (Apache-2.0 and LoRA path confirmed; no metric values on card).
- Chronos-Bolt Base: probe in build 10 (Apache-2.0 confirmed; fine-tune via AutoGluon; research-only note on companion library).
- Lag-Llama: probe in build 10 (Apache-2.0 confirmed; finetune script; context 32 unverified).
- Sundial-base-128m: zero-shot probe only (Apache-2.0 confirmed; no fine-tuning route documented).
- Chronos-2, TimesFM 3.0, Toto-Open-Base-1.0, TiRex, CS-NET v4, TD-MPC2: held out (refuted on fine-tunable or other claims).
- Moirai 2.0 R-small: unverified; non-commercial weights; not promoted.
- CS2 Hierarchical World Model (cbctr): reference evaluation protocol only; pixel-based, cannot consume tick arrays; numbers unverified.
- CounterStrike-1K: external evaluation candidate if access and licence are confirmed; not a build input.
- DreamerV3, TD-MPC2, IRIS, DIAMOND, V-JEPA 2, Oasis, GameNGen, Genie 3, CS2Dream: excluded from builds (pixel-based or control-oriented; or licence or refuted).

### Open questions

- Owner: is the non-inferiority margin 0 or +0.001 (pre-registered), and what are the latency and cost budgets? The brief does not state them.
- Confirm the tome schema: event names, tick rate, slot fields, bomb event fields, grenade detonation times, and round-winner table.
- Supply a de_mirage nav mesh and site polygons; both are needed for the hazard model and the retake event.
- Confirm that academic research use of non-commercial weights (TabFM, TimesFM 3.0, LimiX-2M, Moirai 2.0) is acceptable for publication.
- Confirm whether the 2,000-resample match-cluster bootstrap is acceptable, since the owner's CI uses 'match bootstrap' without a count.

### Verification failures (claims that were refuted, corrected, or could not be checked)

- REFUTED TabPFN-3.5 (arXiv 2609.17895): abstract scope (non-i.i.d., relational, time-series, multimodal) and 'tuned baselines' wording overstated; 'fine-tunable false' is absence-only; attribution clause belongs to TabPFN-2 weights licence. Non-commercial weights confirmed.
- REFUTED TabPFN-v2: 'fine-tunable false' wrong (Nature paper calls fine-tuning proof of concept; v2 licence permits derivatives). Prior Labs licence and 10k-row scale confirmed.
- REFUTED Mitra: 'consistently outperforms TabPFNv2 and TabICL' not supported by its Table 2 (AUC 0.896 vs TabPFNv2 0.901) or Table 4 (TabPFNv2 lower RMSE). Apache-2.0 metadata and fine-tune path confirmed; no LICENSE file in repo tree.
- REFUTED LimiX-2 (400M): 'fine-tunable false' wrong; LICENSE permits derivative models for non-commercial use. LimiX-2M licence conflict (card says academic free; LICENSE.txt reads as a broad grant); 2M weight location unconfirmed.
- REFUTED Chronos-2: 'fine-tunable false' wrong (AutoGluon fine_tune lora or full; 15 HF finetunes). Apache-2.0 confirmed; no WQL, MASE or CRPS on card or abstract.
- REFUTED TimesFM 3.0: 'fine-tunable false' wrong (README LoRA example; licence permits non-commercial fine-tuned derivatives). Non-commercial licence and absence of metrics confirmed.
- REFUTED Toto-Open-Base-1.0: 'fine-tunable false' wrong; DataDog README documents fine-tuning scripts. Apache-2.0 confirmed; card scores not independently checked.
- REFUTED TiRex: 'fine-tunable false' wrong; NXAI Community License permits modification; README has a finetuning section (contact-based).
- REFUTED CS-NET v4: 'fine-tunable false' wrong; repo ships finetune_spatial_only.py. MIT licence and checkpoint list confirmed.
- REFUTED TD-MPC2: 'fine-tunable false' wrong (HF card names fine-tuning as expected use; paper Sec 4.1 fine-tunes); 'no values in text' wrong (paper Table 1 has values). Checkpoint MIT licence confirmed.
- REFUTED arXiv 2207.14124 (GNN sports outcomes): the 9% and 20% headlines are not like-for-like. The 20% compares against a different dataset (0.5353 from arXiv 2011.01324 ar5iv summary, unverified); the paper's own no-graph baseline is already 18.7% below that; the GAT adds 1.7%; the esports result is a single split with no significance test; the NFL 9% compares against single-state models.
- REFUTED arXiv 2510.17199 (VALORANT): 'no numeric baseline' is false (Table I gives minimap-only 72.28%); 'significantly above' is an author claim with no test reported.
- REFUTED arXiv 2106.01782 (DOTA 2): 'accuracy' is R-squared on predicted gold, not match win accuracy; the 80/20 split is stated only for two models.
- UNVERIFIED Moirai 2.0 R-small: fine-tuning for 2.0 not documented on card or uni2ts README; cc-by-nc-4.0 weights and Apache-2.0 code confirmed.
- UNVERIFIED cbctr/cs2-hwm-scale300: fine-tunable claim unsupported either way; card labels 18.6% as chance vs 16.7% on the same card; the 8 s position figure comes from a direct head, not the world model; offline probe loses to a zone-transition prior, conflicting with 'beats prior'; CC BY-NC for CS2-10k comes only from news coverage.
- UNVERIFIED arXiv 2309.02449 (LoL): dataset, split and full tables not readable; a related thesis snippet names Random Forest best at 81.57% for the same window, so the model identity is unresolved.
- UNVERIFIED CounterStrike-1K: only the poster page was read; no download location, licence or access route confirmed.
- PARTIAL repo logreg: extended_metrics.csv gives EFB2 AUC 0.8515 (BSS 0.3719, cAUC 0.5963) vs appendix 0.8519 (0.3726, 0.6033); 2026 EB2-to-NoMoney gap is 0.007, not unchanged; intercept +0.11 holds for set A only (E and EB2 about +0.10); 95% CI 0.8451 to 0.8582 not found in CSVs; 'touched once' not checked (lag-2025 and same-year holdout files exist).
- PARTIAL repo rf: 'scikit-learn RandomForestClassifier' not confirmed (draft says only random forest, 300 trees; requirements list scikit-learn>=1.4).
- PARTIAL repo xgb: 'depth-3 tuned' not confirmed (selected depth absent from outputs); tune_xgb selects on the reporting folds, so in-time numbers are optimistic; draft 7.8 out-of-time contested-AUC baseline 0.590 conflicts with draft 7.2 EB2 0.586.
- PARTIAL repo lgbm: draft 7.1 calls 0.8493 in-sample; draft 7.8 calls it in-time; code is match-grouped out-of-fold.
- NO VERIFICATION repo catboost: EFB2 0.8465 and the paper's -0.0026 delta called significant with no CI.
- NO VERIFICATION repo soft-vote: 0.8531 is from an extended_metrics summary; member sets unnamed; SOFT-VOTE(4) vs SOFT-VOTE(5) conflicts with paper section 6; ensemble_oof.py defaults to two members.
- NO VERIFICATION repo logistic stack: ECE 0.042 only; no AUC number given.
- NO VERIFICATION repo TCN 0.8488, Transformer 0.8473, GAT 0.8465: GAT reads trajectory_dataset.parquet, not documented in README; TCN default mode reports on the checkpoint set (optimistic); tcn.py docstring vs CLI defaults conflict (hidden 64 vs 48; dropout 0.2 vs 0.3; epochs 30 vs 40).
- NO VERIFICATION repo round-to-map M0-M4: M4 in-time 0.5566 and out-of-time 0.5214 unchecked; M0 uses test labels as a flagged diagnostic.
- NO VERIFICATION repo t0 round-start pilot (4,866 rounds; leaky variant included).
- NO VERIFICATION repo leakage flag: BOMB_LIVE_COLS bomb_state = 3 after defuse or detonate; effect on EB2 post-plant gains (paper 7.4) unknown.
- NO VERIFICATION repo internal conflicts: assemble.py vs economy.py and tcn.py on round_num; the 'nine architectures tie in-time' claim.
- NO VERIFICATION ESTA Mirage numbers (MLP 0.430, LightGBM 0.439, XGBoost 0.440) from the ideas research; Table 1 not checked.
- NO VERIFICATION TabPFN-3 report claims (1M rows; large-data results in Figure 15; SHAP 1.08 s per row).
- NO VERIFICATION TabICLv2 FAQ limits (300 to 48K pretraining rows; about 100 features) and README row range: research-sourced only; the first experiment must check the feature cap.
- NO VERIFICATION Nature TabPFN v2 limits (10k rows, 500 features) from search excerpts only.
- NO VERIFICATION Chronos-2 covariate and multivariate claims; TimesFM 2.5 XReg re-add (Oct 2025); Lag-Llama context 32.
- NO VERIFICATION Szmida and Toka (Sloan): 76% frame accuracy and Shapley claims from landing page only; no baseline or log loss.
- NO VERIFICATION arXiv 2011.01324 numbers (XGBoost log loss 0.5353, Brier 0.1842, AUC 0.7913) from an ar5iv summary; not comparable with our data.
- NO VERIFICATION CS2 mechanics (40 s bomb timer; 10 s and 5 s defuse; 1:55 round) from guides; a July 2026 C4 rework claim is single-source and not used.
- NO VERIFICATION missed-play, false-move and evaluation thresholds: all are proposals, not calibrated against our baseline.
- CORRECTION TabICLv2: weights are at huggingface.co/jingang/TabICL (default ckpt names), not soda-inria (research URL returned HTTP 401).
- CORRECTION TabDPT: HF card metadata states apache-2.0 (not unstated); fine-tuning is undocumented rather than false; its XGBoost row is taken from McElfresh et al. and not rerun; card says v1.1 while repo table says v1.3.
- CORRECTION arXiv 1906.05029 (Bayesian soccer): first posted 2019 (KDD 2019); 2021 is a revision date.
- CORRECTION arXiv 2002.02644: abstract says 'traditional calibration techniques', not 'existing methods'.
- CORRECTION Sundial-base-128m: 'first on GIFT-Eval MASE' is the May 2025 release claim; current rank not checked; card is inconsistent on pre-training scale.
- CORRECTION Rocket League random forest (PMC8481284): the target is goal difference, and the study is Rocket League, not CS2.
- CORRECTION ISI 31260 Precision Under Fire: 'XGBoost best' comes from an abstract summary; metrics, split and match count not seen.

### Reading list

Reading order: 1. Henry1145141919810/CS2_Win_Probability_Model repo and working paper (de_inferno; read for method; no licence). 2. arXiv 2109.12990, Optimal Team Economic Decisions in Counter-Strike (abstract confirmed). 3. arXiv 2002.02644, Temporal Probability Calibration (confirmed). 4. arXiv 2010.00781, Evaluating real-time probabilistic forecasts, NBA (confirmed). 5. arXiv 2209.09861, ESTA dataset (confirmed; external data, not a build input). 6. ISI publication 31260, Precision Under Fire, CS:GO 2025 (confirmed; metrics not seen). 7. arXiv 1802.07127, VAEP (confirmed). 8. arXiv 2512.10355, DEFCON graph defense valuation (confirmed). 9. arXiv 2011.01324, Valuing Player Actions in CS:GO (abstract only; numbers unverified). 10. arXiv 2207.14124, GNN sports outcomes (headline refuted; read for method only). 11. Szmida and Toka, CS2 temporal heterogeneous GNN, Sloan (unverified; landing page only). 12. arXiv 2602.11139, TabICLv2 (confirmed licence and fine-tuning). 13. arXiv 2410.18164, TabDPT (repo confirmed; paper not checked). 14. arXiv 2310.08278, Lag-Llama (confirmed). 15. arXiv 1803.01271, Bai et al., TCN (abstract only). 16. arXiv 1906.05029, Bayesian in-game win probability in soccer (confirmed; arXiv 2019, not 2021). 17. arXiv 1408.0777, basketball multiresolution possession model (confirmed). 18. PMC8481284, Rocket League random forest (confirmed). 19. JQAS 2014, random forests for NFL win probability (confirmed). 20. arXiv 2309.02449, LoL real-time result prediction (unverified). 21. CounterStrike-1K, NeurIPS 2026 poster (confirmed as dataset; access and licence unconfirmed). 22. cbctr/cs2-hwm-scale300 model card, CS2 hierarchical world model (unverified; pixel-based). 23. Nixon et al. 2019, adaptive ECE (cited from memory; unverified). 24. Chen et al. 2020 and Hooker et al. 2019, attribution faithfulness (cited from memory; unverified). Grouped by Zotero subfolder: Foundation models: 12, 13, 14, 15. CS2 and esports win probability: 1, 2, 5, 6, 9, 18, 20, 21, 22. Sports tracking and graph models: 7, 8, 10, 11, 17, 19. Evaluation and calibration: 3, 4, 16, 23, 24.

## 3. Other team shooters, spatial control and team-value methods (wf_17863350-b4a)

### Summary

Synthesis for the CS2 1 Hz de_mirage win-probability model, covering team-FPS work beyond CS2 and shared-weight designs from soccer, Dota 2 and StarCraft. Bottom line: no tick-level, match-split log-loss result from another team shooter was found, and no family has an evidenced paired gain on our data yet. What VALORANT, Overwatch and Rainbow Six Siege publish is duel-level (VALORANT XGBoost, test Brier 0.228, Winning Duels), round- or map-level tabular models (VALORANT ultimate and loadout logistic regression with no metrics; Overwatch Elo and map win rates; R6 operator models scored by cross-validation), or minimap video (excluded). Halo, Call of Duty and Apex: nothing usable found. The families that fit our 1 Hz arrays come from CS:GO/CS2 papers, soccer control models, and set or pooling designs from Dota 2 and StarCraft. Ranked by expected paired gain per cost and risk: (1) time-indexed calibration with per-second delta attribution on the existing XGBoost, which gives the explainability target at near-zero cost; (2) a plain MLP on the same 57 inputs, since ESTA's MLP had the lowest log loss on all seven maps (arXiv 2209.09861, though its split was per round, not per match); (3) navmesh-geodesic and economy features added to XGBoost (navmesh provenance unconfirmed); (4) a shared-weight slot encoder with a global state branch, the only 1 Hz CS:GO graph result with a reported gain over a state-only net (log loss 0.4276 vs 0.4351, arXiv 2207.14124), not against a tree model; (5) per-player death-hazard heads; (6) THGNN-lite, the current build, whose published gain is over logistic regression on its own data and which uses rating-from-history features that must be removed. Ranks 11 to 15 are not tick-state compatible as published or are excluded by the owner's rules. Reversal thresholds use one scale: 0.0015 paired log loss against the 57-input XGBoost, the GRU gap the owner already measured, with 95 percent intervals from a match-cluster bootstrap on the 865,974 test seconds. The verification pass changed several survey claims (see dropped_or_refuted). The most material: the CS-NET de_mirage OBJ files are not confirmed as a navmesh, ESTA's benchmark code is in a Colab notebook rather than the repo, and the Valuing Player Actions model is not tick-state compatible as published.

### Data-access findings (what other studios expose)

- Riot, VALORANT match data: Riot's match API as documented in third-party docs (valaw KillDto; HenrikDev v4) exposes positions only at kill, plant and defuse events (playerLocations x, y, viewRadians; plantPlayerLocations; defusePlayerLocations), plus per-round per-player loadout_value and remaining, ability casts and damage events. No per-tick stream is described. Riot's developer docs (developer.riotgames.com/docs/valorant, verified) say personal keys are not supported, RSO needs a production key, and player data needs opt-in. Riot's own VAL-MATCH-V1 reference did not render, so field names are unverified against Riot.
- Riot via GRID (commercial): the GRID VALORANT product page (verified) lists round economy and weapon spend, agent abilities and ultimates, spike plant and defuse states, clutch probability and round forecasts, sourced from Riot. Positions and tick data are not listed. The Riot-GRID VALORANT Data Portal may offer replay files (a February 2023 thespike.gg article, from the survey; contents not described). Tick-level content is unverified.
- Blizzard, Overwatch: OWL statslab match-map CSVs for 2018 to 2022 at round-row granularity, per the Turing Data Stories page; no explicit redistribution licence. The OWL API (api.overwatchleague.com) is scraped by OverwatchLeaguePredictor (README). No Overwatch position telemetry was found. Both sources are match- or round-level.
- Ubisoft, Rainbow Six Siege: the AliNaraqi repo (verified) describes a Season 5 ranked dataset of about 24M rounds and 445K matches collected from the Ubisoft platform; its method and terms are not described. It has 31 columns and no positions or tick timing beyond round duration. The repo sidebar shows MIT for repo code only. No official Ubisoft positional data or API was found. Treat provenance and terms as unverified.
- Epic: no Epic-run competitive dataset with positions was found. Unreal Tournament-era bot frameworks (Pogamut 3; project page only; last version 3.6.1 in 2014 per survey) expose game state to scripted bots, not match data. Fortnite was not searched.
- Valve, CS2: no official Valve match-data documentation was found (survey not_checked; not verified against Valve). Public routes are HLTV-hosted demo files (used by ESTA, THGNN and Valuing 2020) parsed with awpy (MIT; verified: per-tick player position, eye angles, health, armour, economy, grenade trajectories, bomb events, ticks(props)) or demoparser2 (MIT; verified; events not enumerated; per-tick Z unconfirmed). Valve terms and the licence of parsed derivatives were not checked.
- Activision, Call of Duty Warzone: Caldera has breadcrumb positions about every 2 seconds, no team, player or outcome labels, releases v1.0.0 and v1.0.1, and an Activision Non-Commercial Use License (verified). Not usable for win models or commercial work.
- Open CS2 and CS:GO research assets: ESTA (CC BY-SA 4.0, verified; 1,558 demos; 2 Hz; data only; benchmark code in a Colab notebook, not the repo). csknow (MIT, verified; 16 Hz; de_dust2; 4 percent sample; full set on requester-pays S3). CS-NET (MIT code, verified; Hugging Face checkpoints; no metrics; training data not described; OBJ maps described as pre-optimised for raycasting, not as a navmesh). DECOY (MIT code and checkpoints, verified; de_dust2 only; processed dataset not clearly released).
- Dota 2 (Valve): the Time to Die data and code are not released (survey). The OpenAI Five repository returned HTTP 404 in the survey.

### Ranked families (untested; each has a reversal rule set before the test)

### 1. Time-indexed calibration and per-second delta attribution on the existing 57-input XGBoost
- Source: CS:GO. Temporal Probability Calibration (arXiv 2002.02644; game-level GRU; numbers from survey, not verified): NLL 0.3830 uncalibrated to 0.3793 with per-round-time temperature; ECE 0.0316 to 0.0235. Valuing Player Actions (arXiv 2011.01324; verified): change in predicted probability per event used as attribution; the authors note attribution is not unique.
- Why: Differencing any calibrated per-second probability gives the one-tick-to-next explanation the owner asked for and leaves baseline predictions unchanged. Time-indexed calibration is cheap. Its gain was measured on a game-level sequence model, so it may not transfer to per-second XGBoost.
- Data needed: Existing XGBoost probabilities on validation and test seconds; round-time index; per-second round outcome labels.
- Tick-state compatible: True
- Cost: Very low: differencing predictions and one fit on the validation split.
- First experiment: Fit temperature per round-time bin (about 10 bins) on the validation split only. On test, compare paired by match against uncalibrated XGBoost on log loss and ECE by bin. Emit per-second deltas and check per-round sums against the round-start to round-end change.
- Reversal threshold (stated before the test): Drop the calibration step if paired test gain over uncalibrated XGBoost is below 0.0005 log loss, or if ECE by bin is not lower. Delta attribution is kept either way, since it does not change predictions.
- Risks: Attribution is not unique. Per-second deltas summed over a round are not causal effects. A 2026 cricket preprint (arXiv 2608.14696) argues exact attribution and calibration can conflict; this is an unverified hypothesis.

### 2. Plain MLP on the same 57 flat inputs
- Source: CS:GO. ESTA (arXiv 2209.09861): Mirage log loss MLP 0.430, LightGBM 0.439, XGBoost 0.440 (survey values). Verification of Table 1 confirms the MLP is lowest on all seven maps. Caveat: one random state per round with a random split, not by match.
- Why: If a neural net on flat features beats tree models on this kind of data, the XGBoost baseline may be leaving gain on the table. It needs no new data, costs little, and tests whether model class rather than features is limiting.
- Data needed: The existing 57 inputs and labels.
- Tick-state compatible: True
- Cost: Low.
- First experiment: Two or three layer MLP with dropout on standardised inputs, early stopping on validation matches. Paired match-cluster bootstrap against XGBoost on test.
- Reversal threshold (stated before the test): Keep only if paired delta to XGBoost is at most -0.0015 with a 95 percent interval below 0. Drop if the validation delta is 0 or worse.
- Risks: ESTA's split is not by match, so its gap may shrink or vanish on a match split. ESTA uses a different map set and split. Overfitting risk with 687 test matches and comparatively few training matches.

### 3. Geometry and economy features added to XGBoost (navmesh geodesic distances, alive counts, loadout value sums)
- Source: CS:GO. Valuing Player Actions (verified): bombsite distance from navmesh A* distance; XGBoost log loss 0.5353, better than CatBoost 0.5443 and logistic regression 0.5539 on its team-level features (different data). GNN sports outcomes (arXiv 2207.14124; verified feature list): distance to both bombsites, alive, grenades, helmet, defuse kit. FDG 2022 (survey; not verified): enemies in range and equipment value among top features.
- Why: A cheap route to a paired gain, since the features are computable from our arrays plus a navmesh. XGBoost was the best of the tabular models compared in the verified CS:GO work. Whether these features add information depends on what the 57 inputs already contain, which is unknown.
- Data needed: x, y, z, alive and loadout channels per slot; a validated de_mirage navmesh (provenance open).
- Tick-state compatible: True
- Cost: Low: one geodesic query per player per bombsite per second (not measured).
- First experiment: Add per-team minimum geodesic time to each bombsite and nearest-alive-enemy distance, plus alive counts and loadout value sums. Retrain XGBoost with the same hyperparameters; paired match-cluster bootstrap on test.
- Reversal threshold (stated before the test): Keep if paired delta to the current XGBoost is at most -0.0015 with a 95 percent interval below 0. Drop if the validation delta is 0 or worse.
- Risks: Navmesh provenance unconfirmed. Distance-based enemy counts ignore line of sight. The FDG feature list may include an average death count that could be history-based, which the owner excludes. Overlap with existing inputs unknown.

### 4. Shared-weight slot encoder over the 10 players with a global state branch (Deep Sets, Set Transformer, attention over player tokens)
- Source: CS:GO. GNN sports outcomes (arXiv 2207.14124; verified Table I): GAT+State log loss 0.4276 vs State 0.4351 (about 1.7 percent lower); GCN without the state branch 0.6111, so the state branch is needed. ESTA (verified): Deep Sets and Set Transformer did not beat the vector MLP on any map (Mirage 0.440 and 0.447 vs 0.430). Generic designs: Deep Sets (arXiv 1703.06114), Set Transformer (arXiv 1810.00825), PIC permutation-invariant critic (arXiv 1911.00025; RL; parameter count independent of agent count). Dota 2 Time to Die (arXiv 1906.03939): shared per-hero sub-network.
- Why: Meets the reduced-dimensionality constraint: one set of weights for all 10 slots. ESTA shows plain set models do not improve log loss, and the GNN gain appears only with a global state branch, so this family earns its place on parameter count unless it beats the flat MLP.
- Data needed: Per-slot x, y, z, alive, loadout channels and side flag; global 57-input vector for the state branch. No identity features.
- Tick-state compatible: True
- Cost: Low to moderate.
- First experiment: Deep Sets encoder (shared per-slot network, mean pooling, side embedding) plus the 57-input global vector. Paired comparison against the flat MLP on the same inputs and against XGBoost.
- Reversal threshold (stated before the test): Drop if paired delta to the 57-input XGBoost is above +0.0015 with a 95 percent interval excluding 0, or if it does not beat the flat MLP on the same inputs.
- Risks: The GNN gain is against its own state network on different data (2017 to 2020 pro LAN, per-map models), one split, and no CS:GO significance test. The GNN 20 percent claim is against another paper's XGBoost (see dropped_or_refuted).

### 5. Per-player death or kill hazard heads as auxiliary targets on the shared encoder
- Source: CS:GO. FDG 2022 (Marshall et al.; landing page verified): LSTM F1 0.38 for death within 3 s across 10 players; CNN and RNN also compared. Dota 2 Time to Die (arXiv 1906.03939; verified precision 0.377 and recall 0.725 at threshold 0.9; average precision 0.5447 unverified). CS2 CS-NET (Hugging Face card, verified): alive_end and future_kill heads exist, with no reported metrics.
- Why: Death labels come from player_death events in our tomes. A per-player hazard explains a state change and may regularise the shared encoder. No paired win-probability gain is published for it.
- Data needed: player_death and player_hurt events from the tomes aligned to 1 Hz states; per-slot states.
- Tick-state compatible: True
- Cost: Low to moderate: event-derived labels plus one head per slot.
- First experiment: Add a 3-second death-hazard head per slot as a multi-task loss on the shared encoder. Compare win log loss against the same encoder without it; report per-slot hazard average precision.
- Reversal threshold (stated before the test): Drop the auxiliary head if it raises paired win log loss by more than +0.0015 against the single-task encoder.
- Risks: Rare events (about 1 percent positives in Time to Die), so calibration matters. The FDG features may include history-based inputs. F1 0.38 is low.

### 6. THGNN-lite: heterogeneous player-map temporal GNN (current build)
- Source: CS2 (Szmida and Toka; already known, not re-listed). Verified in the paper's table: log loss 0.4587, F1 0.7463, AUC 0.8549, ECE 0.0307 vs logistic regression 0.4688, on 114 Inferno matches (Oct 2023 to Sep 2024). The XGBoost 0.5353 in that table comes from the Valuing 2020 data, not the same dataset.
- Why: Closest published architecture to our build: per-player tokens, a map graph, and temporal state. Its published gain is over logistic regression on its own data, and its split is not stated.
- Data needed: Per-slot positions, velocity from differencing, view yaw sin and cos, alive, loadout; navmesh map graph; no rating or history features.
- Tick-state compatible: True
- Cost: High: about 4.34 million seconds of graphs at 1 Hz across all splits (not measured).
- First experiment: Train THGNN-lite on 1 Hz graphs with a match-level split and no history features. Compare paired against XGBoost on validation before opening test.
- Reversal threshold (stated before the test): Stop if validation paired delta to XGBoost is above +0.0015 over two seeds, or if inference exceeds the latency budget. The brief gives no latency number; the owner must set it before the test.
- Risks: Published features include HLTV Rating 2.1, KAST and Impact from the prior year, which breaks the owner's rule and changes the model. Split not stated, so leakage is possible; overlapping 5-second windows with 2.5-second offsets can leak across windows. Code link (bit.ly) returns 410. Velocity at 1 Hz is noisy.

### 7. Duel-level engagement win model as a per-tick sub-model
- Source: VALORANT. Winning Duels (SSAC 2022; verified): XGBoost test Brier 0.228 (validation 0.231; in-sample CV 0.223) on 5,542 test gunfights from 37 matches; inputs are both players' x, y at the fight plus guns, armour and agents. Code MIT (github.com/jlirms/valorant-analytics). Analytic baseline: stochastic duels with varying kill probability (Operations Research 1967; year from volume, unverified).
- Why: Engagement win probability from positions and weapons is the team-FPS analogue of per-kill win probability added. Labels come from player_death events; per-tick duel margins come from our positions and loadout channels, giving an interpretable fight-state feature. Only the idea transfers; the published model is event-level.
- Data needed: Per-tick positions, primary and secondary weapon channels, armour, alive state; kill events from the tomes.
- Tick-state compatible: True
- Cost: Low.
- First experiment: Fit a kill-event duel model on CS2 tomes data; aggregate per-tick expected duel margin per team; add to XGBoost; paired bootstrap on test.
- Reversal threshold (stated before the test): Drop if the paired validation delta to XGBoost is 0 or worse after adding the duel margins.
- Risks: Reports Brier score, not log loss, so comparison needs conversion. Per-second duel margins may duplicate alive and HP features. VALORANT weapon and agent features do not map directly to CS2.

### 8. Pitch-control and space-control surfaces, and navmesh influence maps
- Source: Soccer. Space and Control in Soccer (Frontiers 2021; verified): the data-driven model gives positive final-pass control for the attacker about 75 percent of the time; SG_rec vs xG per 90 r = 0.66 (CI 0.54 to 0.76; CI level unstated); DFL data not public. LaurieOnTracking (MIT; verified): reference pitch-control code, needs velocity. Voronoi in soccer (arXiv 2107.05714; verified): equal-speed Voronoi is a special case. Spearman pitch control: primary paper not opened. CS: no evaluation against win log loss found. Game AI influence maps (GameAIPro 2 chapter 30; year and numbers unverified).
- Why: Gives each second a per-player control map, and the team control change explains that second's change. Weights are shared across players by construction. No CS evidence of log-loss gain exists, so this is mainly an explanation layer until tested.
- Data needed: Per-slot positions at 1 Hz, smoothed velocity, navmesh geodesic distances, loadout speed class (knife, pistol, rifle, AWP).
- Tick-state compatible: True
- Cost: Not measured: grid cells times 10 players per second; a bounded navmesh search per team would be cheaper than a full grid.
- First experiment: Per-second control scalar per team on coarse navmesh cells with speed classes; add to XGBoost; paired test; check sensitivity to velocity smoothing.
- Reversal threshold (stated before the test): Keep as a predictive feature only if paired delta to XGBoost is at most -0.0015 with a 95 percent interval below 0. Otherwise keep as an explanation view only.
- Risks: US patent 11989942 on dominance or occupancy of space; PDF unreadable, so a freedom-to-operate check is needed before publishing. Velocity noise from 1 Hz differencing (2026 soccer preprint arXiv 2609.10801, unverified, puts break-even near 1 m/s). Navmesh provenance unconfirmed.

### 9. Learned movement or next-position model as an auxiliary loss or counterfactual generator
- Source: CS:GO Retakes on de_dust2: Learning to Move (arXiv 2408.13934; verified): 16 to 59 percent higher TrueSkill human-likeness than commercial bots and scripted controllers; 123 hours; 16 Hz; code MIT (github.com/David-Durst/csknow); 4 percent sample plus full set about 30 GB compressed (about 230 GB uncompressed), requester-pays S3. CS2: CS-NET path head (Hugging Face card; no metrics). DECOY (arXiv 2509.06355; verified): waypoint simulator, DIP accuracy 90.8 percent, AUC 0.951; de_dust2 only.
- Why: One shared transformer over player tokens meets the weight-sharing constraint and reports under 0.5 ms per step on one CPU core (arXiv abstract, via survey). Useful for a next-second position loss, or to generate counterfactual states for stress tests. Not a win model.
- Data needed: Per-slot positions, alive, team, bomb timer; navmesh if DECOY-style discretisation is used.
- Tick-state compatible: True
- Cost: Moderate: retraining on de_mirage, resampling from 1 Hz to the 125 ms step, S3 data cost if used.
- First experiment: Add a next-second position head to the shared encoder and measure the win log-loss change. Separately, use counterfactual states to test the win model's sensitivity.
- Reversal threshold (stated before the test): Drop the auxiliary head if it raises paired win log loss by more than +0.0015 against the single-task encoder.
- Risks: Trained on de_dust2 pro data only. Human-likeness is not a win signal. Data access is requester-pays. CS-NET has about 138.7M parameters, which conflicts with the cost target.

### 10. Sequence models over longer history (GRU, LSTM, attention over more than 20 s)
- Source: CS:GO (owner's own test): GRU with attention over 20 s is +0.0015 worse on paired seconds (project context). Temporal Probability Calibration (arXiv 2002.02644; game-level GRU; numbers unverified). FDG 2022 LSTM (F1 0.38; verified). CS2 CS-NET transformer (no metrics).
- Why: Tick-compatible in input, but the owner's own result is negative and the one CS2 transformer found reports no metrics. Low expected gain per unit cost.
- Data needed: Windowed 1 Hz per-slot states over 60 s or longer.
- Tick-state compatible: True
- Cost: Moderate to high.
- First experiment: Only if justified: a 60 s causal attention encoder with the same split and a paired test against XGBoost. Otherwise leave dropped.
- Reversal threshold (stated before the test): Revisit only if a model with 60 s or longer context beats XGBoost by at least 0.0015 with a 95 percent interval below 0. Otherwise it stays dropped.
- Risks: Overfitting with 687 test matches; longer context may not help per-second prediction; latency grows with window length.

### 11. Multi-agent RL critics and value factorisation (QMIX, MAPPO, COMA, MAT) - not tick-state compatible as trained
- Source: StarCraft II micromanagement (SMAC) and multi-agent benchmarks. QMIX (arXiv 1803.11485), MAPPO (arXiv 2103.01955), COMA (arXiv 1705.08926), MAT (arXiv 2205.14953; licence not established). AlphaStar (arXiv 2308.03526; RTS): set-plus-map architecture.
- Why: Monotone mixers (QMIX) would give per-agent contributions, and centralised critics suit joint state. But published training needs simulator rollouts (COMA also needs per-agent policies). No validated de_mirage simulator is in our set. Only architecture ideas transfer.
- Data needed: A simulator or offline policy data; neither is in our data.
- Tick-state compatible: False
- Cost: High.
- First experiment: None. Port only the pooling idea (rank 4) as supervised learning.
- Reversal threshold (stated before the test): Not testable on our data. Revisit only if a validated de_mirage simulator exists.
- Risks: Out of reach without a validated simulator. The QMIX mixer is not symmetric over slots, which conflicts with full weight sharing unless slots are ordered. RL values are not calibrated probabilities.

### 12. Riot API-based VALORANT models (event-sampled positions only) - not tick-state compatible
- Source: VALORANT. valaw and HenrikDev docs (verified; third-party; Riot's VAL-MATCH-V1 reference did not render): positions only at kill, plant and defuse events; per-round loadout_value and remaining; no per-tick stream. Riot developer docs (verified): personal keys not supported, RSO needs a production key, player data needs opt-in. GRID product page (verified): round economy, ultimates, plant and defuse states; no positions or tick data listed.
- Why: The public match API cannot produce 1 Hz tick-state arrays, so a VALORANT version is out of reach without new data.
- Data needed: A Riot production key with RSO opt-in, or a licensed feed with positions. The Riot-GRID replay-file lead is unverified.
- Tick-state compatible: False
- Cost: Not applicable until access exists.
- First experiment: None until access is granted. Check GRID replay-file contents (a February 2023 thespike.gg article, from the survey, says teams can download replay files; contents not described).
- Reversal threshold (stated before the test): Reconsider only if tick-level positions (1 Hz or finer) with loadouts are obtained.
- Risks: Access and licence terms; Riot's terms were not read in full.

### 13. Round-, map- or match-level models from other team shooters and CS:GO or CS2 snapshots - not tick-state compatible
- Source: VALORANT: VCT 2024 round logistic regression (IEEE DataPort; 1,301 rounds; no metrics; subscription). Leiden halftime-lead baseline (76.40 percent over 1,898 games including ties; verified). Pawar XGBoost 73 percent (verified with corrections: 11 other models; 80:20 stratified split; labels may be GMM-simulated, unconfirmed). Overwatch: Turing OWL story (Elo 62.3 percent; map win rate 53.3 percent; Ridge 62.2 percent with a split bug; Elo excluded). OverwatchLeaguePredictor (74 percent win/lose, self-reported, 2019 Stage 2). R6 Siege: IJEECS model (93.75 percent under 5-fold CV; attribution to IBK inferred). CS:GO snapshots: Kaggle-derived random forest 86.9 percent (verified README; split unstated); jevinn 82.57 percent (verified; features chosen before the split). CS2 map-level cs2-match-prediction (LightGBM ROC-AUC 0.72, Brier 0.231, log loss 0.659; verified; MIT). PandaScore Markov round model (no numbers).
- Why: This is what published team-shooter win models outside CS2 look like. Useful only as sanity floors in the evaluation table, because they lack positions and timing, and several headline numbers are weak or leaky.
- Data needed: Round, map or match labels; no per-second positions.
- Tick-state compatible: False
- Cost: Low to build; no value for per-second state.
- First experiment: None for the per-second model. Report the halftime-lead heuristic (76 percent) and map win rate as floors in the evaluation table.
- Reversal threshold (stated before the test): Not applicable. Only a tick-level dataset would reopen this family.
- Risks: Leakage through random CV and split bugs. Small samples. Non-comparable metrics (accuracy, AUC, Brier).

### 14. Video- and pixel-input models - excluded by owner rule and not tick-state compatible
- Source: VALORANT minimap TimeSformer (arXiv 2510.17199; verified): 80.55 percent overall, minimap-only 72.28 percent; per-phase 56.32, 85.80, 90.64, 89.44 for 0-24, 25-49, 50-74 and 75-99 s; output is a round-level probability from video. CS:GO deathmatch behavioural cloning (arXiv 2104.04258; pixels; licence bars commercial use). FTW capture-the-flag (arXiv 1807.01281; first-person pixels). DIAMOND and CS2 world models (excluded in survey).
- Why: The owner excludes first-person and egocentric video; video work is in scope only if the output is tick-level game state. These outputs are round-level probabilities or actions.
- Data needed: Video frames; not in our data.
- Tick-state compatible: False
- Cost: High.
- First experiment: None (excluded). The FTW linear-probe method (probes on hidden state for interpretable targets) can be applied to our encoder without video.
- Reversal threshold (stated before the test): Reconsider only if the owner's rule changes or the output becomes tick-level game state.
- Risks: Owner exclusion. Some licences restrict commercial reuse.

### 15. Rating, Elo and skill-from-history models - excluded by owner rule
- Source: Overwatch League Elo (Turing story); CS:GO TrueSkill (Makarov et al. 2017, not opened); CS:GO player win-probability-change rating (csgo-wpc-rating, not opened); professional vs amateur win probability with Elo (WWW 2022; numbers unverified); LoL Elo models.
- Why: Excluded by the no rating-from-history constraint. Listed so the ranking is visibly done with them removed.
- Data needed: Match histories; not permitted.
- Tick-state compatible: False
- Cost: Not applicable.
- First experiment: None.
- Reversal threshold (stated before the test): Not applicable under the current owner constraint.
- Risks: Rule violation if used.

### Dropped or refuted

- REFUTED OR CORRECTED CLAIMS. THGNN (Szmida and Toka; already known): 'baselines and dataset size not stated' is wrong (the table has logistic regression and XGBoost rows; 114 Inferno matches). Its features include HLTV Rating 2.1, KAST and Impact from the prior year, which breaks the owner's rule. 'Accuracy above 76 percent' is in the text, not the table. Code link (bit.ly/3Zu78tz) returns 410 Gone.
- Valuing Player Actions (Xenopoulos 2020; already known): AUC 0.7913 confirmed. 'Tick-state compatible' refuted: states are team-level and sampled at footstep, damage and bomb events, with no per-player x, y, z. The repository now redirects to awpy (MIT). The paper also downloaded HLTV Rating 2.0 per-match stats; their role in the model is unconfirmed.
- Learning to Move (CSKnow, 2024): 'no code or data release' refuted. Code is MIT; data is released as a 4 percent sample and a full set on requester-pays S3. No trained weights were found.
- DECOY (2025): 'numbers None' and 'no repository' refuted. DIP accuracy 90.8 percent (AUC 0.951), outcome match 91.0 percent, and an MIT repository with checkpoints. Training data is the ESTA set at 2 FPS; only de_dust2 is used.
- ESTA (2022): 'benchmark code in repo' refuted (Colab only; the repo holds data). LAN date range conflicts between README and paper. The Table 1 log-loss range is unreconciled (0.415 to 0.472 in one verification, 0.415 to 0.442 in another). Its random per-round split means it is not comparable to our 0.4512 match-split baseline.
- GNN sports outcomes (Xenopoulos and Silva 2022): 'dataset not stated' and 'positions not described' refuted (7,222 CS:GO games; 14.3M states at 1 Hz; X, Y, Z node features). The 20 percent claim is against Xenopoulos 2020, and 0.4276 vs 0.5353 is about 20.1 percent lower (my arithmetic on two verified numbers from different datasets). The 9 percent claim is against Yurko 2020 on NFL. The CS:GO split is by game state, not match.
- VALORANT minimap video (arXiv 2510.17199): 'per-phase values not given' refuted (Table I has them). Excluded by the owner's video rule regardless.
- Caldera (Activision): 'releases empty', 'code not applicable' and 'numbers none' refuted (releases v1.0.0 and v1.0.1; a Python script; about 1 million end-of-match points; 17.5 million primitives). Excluded anyway: non-commercial licence and no outcome labels.
- r6-draft-helper: 'default run samples 10 percent' refuted. The 10 percent sample and 200,000-round cap apply only with the --sample flag. Its author says coefficients are associations, not a validated win model.
- Pawar VALORANT pre-match thesis: 'other models' accuracies not stated' refuted (Table 1 has 11 other models; 80:20 stratified split; 153 test matches). The 73 percent may be against GMM-simulated labels (unconfirmed).
- Overwatch League Turing story: '13,601 rows' is the full 2018 to 2022 export; the 2021 subset is 2,154 round rows reduced to 894 games. The Ridge combined split is inclusive, so game 447 appears in both train and test. Test slices include a trailing NaN row (effect at most about 0.2 points).
- LaurieOnTracking (pitch control code): 'tick-state compatible' refuted. The pitch-control code needs per-player x, y, vx, vy and ball position; its IO module reads positions and ball only. MIT licence verified.
- Wide Open Spaces (MIT Sloan SAC 2017): 'numbers not captured' refuted. Pitch-value model MSE 0.085 (sd 0.001, 10-fold CV) on TRACAB data from 20 matches (2.4M examples); 845 attacking situations from one match. No code, licence or data statement.
- CS-NET de_mirage OBJ: 'navigation mesh' is not confirmed. The README calls maps/optimized_obj_files pre-optimised for Three.js and Open3D raycasting. Code MIT verified; map-geometry provenance and licence unverified.
- FDG 2022 death prediction: 'no baseline' refuted (CNN and RNN are compared). The page reserves all rights, including text and data mining, which limits reuse.
- Winning Duels (SSAC 2022): 'numbers None' is true only for the landing page. The linked PDF reports test Brier 0.228 (XGBoost); its title is 'Winning Fights in VALORANT'.
- Time to Die (Dota 2): precision 0.377 and recall 0.725 verified; average precision 0.5447 unverified; survey entries disagree on tick-state compatibility; no data or code released; not an FPS.
- Precision Under Fire (CS:GO Dust 2, IEEE CoG 2025): numbers, split and granularity unverified; the tick-state claim rests on the ISI listing only.
- UNVERIFIED, NOT PROMOTED. Spearman pitch control (primary paper not opened). Physics-based pass model (secondary Substack: 80.5 percent on 5,471 passes; the 81 percent receiving-team claim may mislabel the metric). Dick, Link and Brefeld availability (Springer sign-in wall; 58-match claim unverified). Taki and Hasegawa dominant region (dblp blocked; runtime secondhand). Temporal Probability Calibration numbers (game-level; not in verification). SoccerMap numbers. Off-ball velocity preprint numbers. Human-like bots timing (9.59 ms per decision on CPU, survey). Confidence-calibrated MOBA ECE (0.57 vs 1.11 percent, survey).
- EXCLUDED FAMILIES. Elo, TrueSkill and rating families (Overwatch Elo; CS:GO TrueSkill; professional vs amateur Elo; CS:GO win-probability-change rating): excluded by the owner's rule. Video and pixel models (VALORANT TimeSformer; VALORANT CV tracking, SPP 2025, PDF unreadable; CS:GO deathmatch behavioural cloning; FTW; DIAMOND; CounterStrike-1K): excluded by the owner's video rule.
- PATENT RISK. US patent 11989942 (dominance or occupancy of space; PDF scanned and unreadable): freedom-to-operate risk for any pitch-control-like method before publication.
- NOT NEEDED. Opponent-position belief models (2009 CS:Source thesis; not opened). Our tomes hold full state for all 10 players, so partial-observability models are not required for the win model.

### Reading list

Reading order: 1. ESTA: An Esports Trajectory and Action Dataset (2022), https://arxiv.org/abs/2209.09861. Read Table 1 and the split. CC BY-SA 4.0. 2. Graph Neural Networks to Predict Sports Outcomes (2022), https://arxiv.org/abs/2207.14124. 1 Hz CS:GO graph model with a state branch. 3. Winning Duels in VALORANT (SSAC 2022), https://www.sloansportsconference.com/research-papers/winning-duels-in-valorant-a-visualization-of-optimal-positioning. Code: github.com/jlirms/valorant-analytics (MIT). 4. Enabling Real-Time Prediction of In-game Deaths through Telemetry in CS:GO (FDG 2022), https://research.tilburguniversity.edu/en/publications/enabling-real-time-prediction-of-in-game-deaths-through-telemetry/. Full text not read. 5. Overwatch League win prediction, Turing Data Stories (2023), https://alan-turing-institute.github.io/TuringDataStories/stories/2023-02-01-Overwatch-League/OWL_story.html. Read for the bootstrap method and the split bug. 6. Precision Under Fire (IEEE CoG 2025), https://www.isi.edu/results/publications/31260/precision-under-fire-analysis-and-predictive-modeling-in-counter-strike-global-offensive. Full text not read. 7. Temporal Probability Calibration (2020), https://arxiv.org/abs/2002.02644. 8. Rethinking Evaluation Metric for Probability Estimation Models Using Esports Data (2023), https://arxiv.org/abs/2309.06248. Bin-count sensitivity and sample size. 9. Possibilities of Applying Machine Learning to a Limited Esports Dataset (Leiden bachelor thesis, 2025), https://theses.liacs.nl/pdf/2024-2025-NielandJJesse.pdf. Halftime-lead baseline. 10. The Application of Machine Learning Techniques for Predicting Results in Team Sport: A Review, https://arxiv.org/abs/1912.11762. Random-CV leakage warning. 11. Deep Sets (2017), https://arxiv.org/abs/1703.06114. 12. Set Transformer (2019), https://arxiv.org/abs/1810.00825. 13. Permutation Invariant Critic for Multi-Agent Deep RL (2019), https://arxiv.org/abs/1911.00025. Symmetric pooling; parameter count fixed in the agent count. 14. Time to Die: Death Prediction in Dota 2 using Deep Learning (2019), https://arxiv.org/abs/1906.03939. Shared per-hero sub-network. 15. Optimal Team Economic Decisions in Counter-Strike (2021), https://arxiv.org/abs/2109.12990. Round-start economy prior; abstract only read. 16. Space and Control in Soccer (2021), https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2021.676179/full. 17. Wide Open Spaces: A statistical technique for measuring space creation in professional soccer (MIT Sloan SAC 2017), third-party mirror https://code.garrettmills.dev/Archives/papers-we-love_papers-we-love/src/commit/6f92d68b5e4bea94d9453e93e3801b6fd2dbb49a/sports_analytics/2017-A-statistical-technique-for-measuring-space-creation-in-professional-soccer.pdf. 18. The Voronoi Diagram in Soccer (2021), https://arxiv.org/abs/2107.05714. Speed-aware dominance. 19. Spatio-Temporal Analysis of Team Sports, survey (2016), https://arxiv.org/abs/1602.06994. Index only. 20. SoccerMap (2020), https://arxiv.org/abs/2010.10202. Per-cell surfaces; numbers not verified. 21. How Much Velocity Does Off-Ball Space Value Need? (2026 preprint), https://arxiv.org/abs/2609.10801. Velocity noise; unverified. 22. Expected Threat (xT), socceraction docs, https://socceraction.readthedocs.io/en/latest/documentation/valuing_actions/xT.html. Cell-value table. 23. LaurieOnTracking, pitch-control reference code (MIT), https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking. Needs velocity inputs. 24. Learning to Move Like Professional Counter-Strike Players (2024), https://arxiv.org/abs/2408.13934. Code MIT; data released. 25. DECOY: A Data-Driven Discretized CS:GO Simulation Environment (2025), https://arxiv.org/abs/2509.06355. Code github.com/HATS-ICT/decoy (MIT). 26. Human-like Bots for Tactical Shooters Using Compute-Efficient Sensors (2024), https://arxiv.org/abs/2501.00078. Numbers unverified. 27. QMIX (2018), https://arxiv.org/abs/1803.11485. Monotone mixer. 28. The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO), https://arxiv.org/abs/2103.01955. Centralised value input choice. 29. Dota 2 with Large Scale Deep Reinforcement Learning (OpenAI Five, 2019), https://arxiv.org/abs/1912.06680. Win-probability heads on an LSTM; qualitative. 30. Calibration-Leverage Tradeoff in Exactly Solvable Win-Probability Models (2026 preprint), https://arxiv.org/abs/2608.14696. Cricket; not peer reviewed. 31. The Blown Lead Paradox (2026 preprint), https://arxiv.org/abs/2601.18774. Peak statistics selected by outcome. 32. A Confidence-Calibrated MOBA Game Winner Predictor (2020), https://arxiv.org/abs/2006.15521. ECE; numbers unverified. 33. Pre-game paired-comparison modelling of professional League of Legends map outcomes (2026 preprint), https://arxiv.org/abs/2609.08060. Walk-forward and paired tests; its features are excluded. 34. Win Prediction in Multi-Player Esports: Live Professional Match Prediction (2021), https://eprints.whiterose.ac.uk/152931. Evaluation criteria; not retrieved. 35. Valorant Champions Tour 2024: Pacific and EMEA Round Data (IEEE DataPort, DOI 10.21227/v3bk-2n86), https://ieee-dataport.org/node/15540. Subscription; no positions. Zotero subfolders, same numbers: Team shooters beyond CS2: 3, 4, 5, 6, 35. Spatial control and positional value: 16, 17, 18, 19, 20, 21, 22, 23. Team value and set models: 1, 2, 11, 12, 13, 14, 15. Esports evaluation and calibration: 7, 8, 9, 10, 30, 31, 32, 33, 34. Bots and agents: 24, 25, 26, 27, 28, 29. CS-NET, Szmida and Toka, and Xenopoulos 2020 are excluded as already known.

## 4. Gate and next steps

Owner gate for any build (from the research brief): paired log-loss delta at or below zero with the upper bound of the interval at or below zero, against the 57-input baseline on the same seconds. None of the families above has passed it yet.
The strongest leads that would be cheapest to test: time-indexed calibration with per-second delta attribution on the existing XGBoost; lag and difference features on that same XGBoost; a plain MLP on the same 57 inputs; and a shared-weight slot encoder. Patent risk: a US patent (11989942, dominance or occupancy of space) overlaps with pitch-control-like methods, so freedom-to-operate needs checking before any publication.
