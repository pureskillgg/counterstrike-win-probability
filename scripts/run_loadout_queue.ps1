$py = 'C:\Users\<user>\.pixi\workspace-envs\cs2gpu-15282324412032665829\envs\default\python.exe'
$exp = 'G:\pureskill\csds-cs2\scripts\model\exp_stack.py'
& $py -I -X utf8 $exp --store combined --eras 2026 --blocks status --tag status_only --splits random,chrono --stride 2 *> 'G:\pureskill\csds-cs2\logs\exp_status_only.log'
& $py -I -X utf8 $exp --store combined --eras 2026 --blocks hp,feat_x2_utility,feat_x1_kill_timing,feat_x5_bomb_state,feat_x3_shots,status --tag stack_2026 --splits random,chrono --stride 2 *> 'G:\pureskill\csds-cs2\logs\exp_stack_2026.log'
