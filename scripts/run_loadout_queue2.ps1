$py = 'C:\Users\13144\.pixi\workspace-envs\cs2gpu-15282324412032665829\envs\default\python.exe'
$exp = 'G:\pureskill\csds-cs2\scripts\model\exp_stack.py'
while (-not (Test-Path 'G:\pureskill\csds-cs2\data\combined\exp_stack_2026.json')) { Start-Sleep -Seconds 20 }
& $py -I -X utf8 $exp --store combined --eras 2026 --blocks status,feat_x2_utility,feat_x1_kill_timing,feat_x5_bomb_state,feat_x3_shots --tag stack_2026b --splits random,chrono --stride 2 *> 'G:\pureskill\csds-cs2\logs\exp_stack_2026b.log'
& $py -I -X utf8 $exp --store combined --blocks status,feat_x2_utility,feat_x1_kill_timing,feat_x5_bomb_state,feat_x3_shots --tag stack_all_eras --splits random --stride 3 *> 'G:\pureskill\csds-cs2\logs\exp_stack_all_eras.log'
