$py = 'C:\Users\13144\.pixi\workspace-envs\cs2gpu-15282324412032665829\envs\default\python.exe'
foreach ($m in 'feat_x4_hp_detail','feat_x3_shots') {
  & $py -I -X utf8 'G:\pureskill\csds-cs2\scripts\model\feature_harness.py' --module $m --store combined_test --seeds 0 *> "G:\pureskill\csds-cs2\logs\harness_$m.log"
}
