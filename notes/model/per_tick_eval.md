# Per-second win probability evaluation (advisor's rules)

Script: `scripts/model/per_tick_eval.py` (seed 0). Split by match (bake-off primary split). 1 Hz grid of ticks, strictly before round end. Intervals are match-cluster bootstrap (1000 resamples).

Test: 304 matches, 5849 rounds, 404833 one-second rows.

| Model | LL all ticks [95% CI] | LL round-weighted | Brier | AUC | ECE | SE cluster / naive |
|---|---|---|---|---|---|---|
| constant | 0.6936 [0.6927, 0.6945] | 0.6929 | 0.2502 | 0.5000 | 0.0152 | 10.1x |
| LR alive difference | 0.5146 [0.5071, 0.5218] | 0.5010 | 0.1747 | 0.8017 | 0.0092 | 5.4x |
| LR all features | 0.5068 [0.4989, 0.5142] | 0.4929 | 0.1719 | 0.8161 | 0.0122 | 5.5x |
| XGBoost (trained on 1 Hz grid) | 0.4916 [0.4836, 0.4993] | 0.4781 | 0.1672 | 0.8269 | 0.0102 | 5.6x |
| XGBoost (trained on event states) | 0.4934 [0.4857, 0.5010] | 0.4804 | 0.1678 | 0.8254 | 0.0129 | 5.7x |

Paired log loss differences (negative = first model better):

- XGB grid minus LR alive difference: -0.0230 [-0.0262, -0.0198]; round-weighted -0.0229
- XGB grid minus LR all features: -0.0151 [-0.0174, -0.0127]; round-weighted -0.0148
- LR all features minus LR alive difference: -0.0079 [-0.0095, -0.0061]; round-weighted -0.0081
- XGB event-trained minus XGB grid-trained: +0.0018 [+0.0010, +0.0026]; round-weighted +0.0022

Terminal-state inflation (event table): {'event_test_states': 47716, 'terminal_states': 5242, 'terminal_share': 0.10985832844328947, 'logloss_all_event_states': 0.4365863853094365, 'logloss_non_terminal_event_states': 0.488678641752389, 'logloss_terminal_event_states': 0.014501970933632745, 'grid_rows_with_a_side_dead': 7075}

