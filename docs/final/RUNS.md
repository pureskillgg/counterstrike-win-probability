# Run inventory and provenance (4 Nov 2026)

Compiled from the Meridian record (140 pinned decisions reviewed, 116 rows inventoried), 45 run folders (592 run records from 333 metric files), and 160 scripts. The raw inventories are in `docs/final/inventory/`. Numbers below are copied from pin bodies. "Validation" means the 1,168-match validation split unless stated.

## 1. Verdicts (pins)

| Pin | What was tested | Verdict | Test read? |
|---|---|---|---|
| 921fa3dd | All-maps GRU (loadout, no distances) minus 57-input XGBoost, validation, 3 seeds | GRU better on every seed, mean -0.00219 | Not used for the decision |
| 993fc8bb | Fixed 50/50 blend of GRU and distance XGBoost (B1), validation | Blend beats both: -0.00111 vs tree, -0.00176 vs GRU | Not used |
| 1db6312a | Learning curve, GRU vs 57-input tree, seed 0 | GRU advantage grows with matches: +0.00037 (25%), -0.00046 (50%), -0.00197 (100%) | Descriptive, one seed |
| 88d9fbb0 | 83-input XGBoost vs GRU, TCN and transformer arms at 30k steps, validation | All sequence arms worse; best transformer (width 128) +0.00107 over 3 seeds | Validation only |
| edb0aaa3 | Transformer width 128, five seeds vs candidate | No claim (pooled +0.00107; seed 2 interval crosses zero) | Validation only |
| 7085b272 | GRU 60 s minus 20 s, sweep schedule | No claim (pooled +0.000184) | Validation only |
| 0242913c (T3) | Attention GRU minus GRU, de_mirage | Fail (+0.000154) | Test described as descriptive |
| 474c14a9 (T4) | Map identity added to train-site distance XGBoost | Rule not met; per-map models unlikely to help | Validation |
| 4edc74c3, 6787d3b7 | Navigation-region graph vs learned k-means graph | Not adopted; k-means graph kept | Validation |
| 27589820 | Plain MLP on 57 inputs vs tuned XGBoost | No seed passes | Validation |
| 8212faf3 (T0) | Tuned XGBoost vs logistic regression | Pass, +0.0104 | Validation |
| 3796266f (T1) | Death-in-next-3-seconds feature | Fail | Validation |
| be851706 (T2-A) | Attacker-side feature set, 10 events | Fail, mean -0.000438 | Validation |
| acaea8a3 (T6) | Per-weapon run speed in defuse margin | Fail | Validation |
| 93d46735, d71bc655 | All-maps GRU with straight-line, then walking, distances | Straight-line mean -0.00085 (mean met, rule not met); walking not better than straight-line | Frozen de_mirage test read (flagged) |
| 428961a1 | Distance features in XGBoost, three seeds | Rule met for straight-line and walking | Validation |
| 8697407b (W2) | Walking distances added to XGBoost, all maps | Pass; walking vs straight-line not claimed | Validation |
| 370f5ec6 (C1) | Distance gain on contested rounds | Rule met | Validation |
| ee61ba3a | Train-only site centres | Rule met | Validation |
| 71080982 | GPU sprint, 126 cells, three seeds: full set vs base | Full beats base by about 0.0034 at every setting | Validation |
| 56316f69 | Holdout Oct 26 to 31, 83-input candidate | Seed-mean 0.452295 (95% interval 0.449288 to 0.454983). Scored once | Holdout read once |
| 4b699f9d | Production: five-seed XGBoost ensemble, 58 inputs | Test 0.455605 (3,059 matches) | Test scored once |
| aaecad58 | Submission: winprob_v6_drop_heavy, 57 inputs | Test 0.456411; worse than the 58-input production by +0.000806 | Test scored |
| 8ab4d4dc | 58-input score-lead ensemble (not servable) | Test 0.455605, scored once | Test scored once |
| e3ec1051 | Learning curve, 10-feature XGBoost, AWS archive | No claim | Validation |
| f06a5187, ffa3cdd3 | LSTM at 30k steps; LSTM at lr 3e-4 | No verdict pinned. f06a5187 records pooled LSTM minus SN +0.00142 (as recorded in c3856c53) | Validation; test rows loaded in memory |

## 2. Run folders (45)

- **XGBoost sprint:** `sprint/run/gpu_fast` (126 cells, three seeds), `gpu` (2), `gpu_repeat` (4), `smoke/cpu` (1), `data/combined` (257 records, including sklearn and XGBoost configs).
- **Phase-1 sequence family:** `phase1` (15), `phase1_30k` (15), `phase1_arms_final` (7), `phase1_large_dist` (14: GRU, LSTM, TCN and transformer width 128, five seeds), `phase1_lstm_final` (6), `phase1_tf_E`, `_F`, `_G` (3 each), `phase1_gru_ctrl` (3), `phase1_gru_dist` (3), plus two single-run checks.
- **Width megaruns:** GRU width 128 (2), 256 (5), 512 (5); TCN width 128 (2), 256 (5), 512 (5).
- **Learning-rate and schedule sweeps:** `lr_sweep_screen` (12), `lr_sweep_all_seed1` and `_seed2` (6 each), `cosine_arms_final` (9), `cosine_gru_dist_*` variants (3 each: 15k, 20k, 90k steps; lr 3e-3 and 3e-4; batch 4096; train sites; drop both; drop inference), `lstm_lr3e4_cosine` (1).
- **Windows:** `window_gru_final` (6, 20 and 60 s), `window_gru40_final` (3).
- **Other:** `gt_kmeans_s0` (graph transformer, one seed, 0.45249), `learning_curve` (3, XGBoost arms), `holdout_b` (8 records). The AWS learning curve (pin 238a70d6) lives on the SageMaker space, not here.

Test-row fields: 70 records show `test_rows_in_memory = true`. In every record `test_rows_used` is 0 and the numeric test-row counters are 0. The owner's ruling on earlier loads is still pending. Some CSV fields are blank because the source JSON didn't carry them.

## 3. Provenance conflicts to resolve before the meeting

1. **Sequence vs tree.** G1 and B1 (21 to 26 Oct) say the all-maps GRU beats the 57-input XGBoost and the blend beats both. The 30k-step comparisons (88d9fbb0, edb0aaa3) say the 83-input tree beats every sequence arm. They use different comparators (57-input baseline vs 83-input candidate) and different budgets (three epochs vs 30k steps). No pin reconciles them.
2. **Production vs submission.** 4b699f9d (19 Oct) names the 58-input XGBoost ensemble as production. aaecad58 (20 Oct) chooses the 57-input model as the submission, which scores worse on test. Our 83-input candidate appears in neither record.
3. **LSTM.** No verdict is pinned. f06a5187 has numbers but no conclusion, and ffa3cdd3 has no outcome. The earlier claim that LSTM lost is not supported by a pinned result.
4. **Test-row loads.** 70 records loaded test rows into memory. The owner's ruling is pending.
5. **Holdouts.** Oct 26 to 31 has been read once, for the 83-input candidate. Holdout B is contaminated by the owner's ruling. The Mirage test (687 matches) was read by six result files. The team's 3,059-match test set is a different population, so test numbers can't be compared across models.
6. **Data scaling.** 1db6312a says the sequence advantage grows with the number of matches. e3ec1051 found no clear gain for the 10-feature XGBoost. These are different models, but together they suggest more data may matter most for sequence models.

## 4. What to pursue (ranked by existing evidence)

1. **Blend with the 83-input tree.** A 50/50 blend of a GRU with position inputs and the 83-input XGBoost, on validation, five seeds. B1 found a blend beat either model alone by about 0.001. Use B1's rule.
2. **GRU with the 24 position inputs.** Sequence models have never had them. Compare with the plain GRU.
3. **GRU learning curve on more matches.** 1db6312a points this way, but the GRU needs the full round-state build on AWS, which doesn't exist yet. The AWS table is event-only.
4. **Pin the LSTM verdict** from the child pins (0cb0dc8f, ffa3cdd3) before any LSTM claim.
5. **Five seeds** for whatever wins.

**Drop or defer** (evidence says no): walking distances, navigation-region graph, per-map models, word2vec, plain MLP, attention GRU (T3), death-in-three-seconds (T1), weapon speed (T6), foundation models (keep XGBoost).

**Holdout rule for any winner:** score once on Oct 26 to 31 with the second-use disclosure, or on a fresh window.
