$py = 'C:\Users\13144\.pixi\workspace-envs\cs2gpu-15282324412032665829\envs\default\python.exe'
& $py -I -X utf8 'G:\pureskill\csds-cs2\scripts\model\feature_harness.py' --module feat_x4b_hp_only --store combined_test --seeds 0 *> 'G:\pureskill\csds-cs2\logs\harness_feat_x4b_hp_only.log'
& $py -I -X utf8 'G:\pureskill\csds-cs2\scripts\model\feature_harness.py' --module features_events --store combined_test --seeds 0 *> 'G:\pureskill\csds-cs2\logs\harness_hp_all_splits.log'
