# End-to-end pipeline: finished tome collections -> one combined store -> catalog/rounds/labels -> 1 Hz grid -> living-player
# snapshots -> two models (snapshot model + alive-count fallback) -> checks. Stops at the first failing step. Run from anywhere:
#   powershell -NoProfile -ExecutionPolicy Bypass -File G:\pureskill\csds-cs2\scripts\run_pipeline.ps1 [-Store combined] [-SkipBuild]
#     [-Name winprob_v1] [-SnapshotSets map,status] [-SnapshotEras 2026] [-BasicName winprob_v1_basic] [-BasicSets map] [-Stride 1]
param(
  [string]$Store = 'combined',
  [string]$Name = 'winprob_v1',
  [string]$SnapshotSets = 'map,status',
  [string]$SnapshotEras = '2026',
  [string]$BasicName = 'winprob_v1_basic',
  [string]$BasicSets = 'map',
  [int]$Stride = 1,
  [switch]$SkipBuild,
  [switch]$Mutation
)
$ErrorActionPreference = 'Stop'
$base = 'G:\pureskill\csds-cs2'
$cpu = "$base\.pixi\envs\default\python.exe"
$gpu = 'C:\Users\13144\.pixi\workspace-envs\cs2gpu-15282324412032665829\envs\default\python.exe'
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
function Step($label, $exe, $argList) {
  Write-Host "=== $label ($(Get-Date -Format HH:mm:ss))"
  & $exe @argList
  if ($LASTEXITCODE -ne 0) { throw "step failed: $label (exit $LASTEXITCODE)" }
}
if (-not $SkipBuild) {
  # append_combined.py adds only the sources the store does not hold yet (a full build when there is no ledger yet; a channel whose schema
  # changes is rebuilt, the others are appended; verified equal to a full rebuild by test_append_equivalence.py)
  Step '1 combine tomes (append the new ones)'   $cpu @('-I','-X','utf8',"$base\scripts\data\append_combined.py",'--out',"$base\tomes\$Store")
}
Step '2 catalog, labels, rounds' $cpu @('-I','-X','utf8',"$base\scripts\data\prepare.py",'--store',$Store)
Step '3 1 Hz grid'               $cpu @('-I','-X','utf8',"$base\scripts\data\make_grid.py",'--store',$Store)
# the train/val/test assignment is frozen (data\<store>\split_manifest.json, created once with --freeze): new matches are added by a hash rule,
# no existing match ever changes set. Run before anything that fits or evaluates.
Step '3a extend the frozen match split' $cpu @('-I','-X','utf8',"$base\scripts\model\split_manifest.py",'--store',$Store,'--extend')
Step '3b living-player snapshots at the grid ticks (resumable per upload day)' $cpu @('-I','-X','utf8',"$base\scripts\data\extract_status_1hz.py",'--store',$Store)
Step '4a train + export the alive-count fallback model (GPU), GPU/CPU + calibration checks' $gpu @('-I','-X','utf8',"$base\scripts\model\export_model.py",'--store',$Store,'--sets',$BasicSets,'--name',$BasicName,'--stride',$Stride)
Step '4b train + export the snapshot model (GPU), GPU/CPU + calibration checks' $gpu @('-I','-X','utf8',"$base\scripts\model\export_model.py",'--store',$Store,'--sets',$SnapshotSets,'--eras',$SnapshotEras,'--name',$Name,'--stride',$Stride)
Step '4c look at single held-out rounds: probability curves with events, swing audit, behaviour table' $cpu @('-I','-X','utf8',"$base\scripts\model\inspect_rounds.py",'--pred',"$base\data\$Store\export_${Name}_test_predictions.parquet",'--store',$Store,'--out',"$base\data\$Store\round_inspection_$Name")
Step '5a stress test the CPU predictor (snapshot model + fallback)' $cpu @('-I','-X','utf8',"$base\scripts\model\stress_test_winprob.py",'--model',"$base\models\$Name.json",'--fallback',"$base\models\$BasicName.json")
Step '5b stress test the fallback model alone' $cpu @('-I','-X','utf8',"$base\scripts\model\stress_test_winprob.py",'--model',"$base\models\$BasicName.json",'--grid',"$base\data\$Store\export_${BasicName}_test_sample.parquet")
Step '5c train/serve parity on real rows (player table in, pipeline prediction out)' $cpu @('-I','-X','utf8',"$base\scripts\model\parity_check.py",'--model',"$base\models\$Name.json",'--fallback',"$base\models\$BasicName.json",'--sample',"$base\data\$Store\export_${Name}_test_sample.parquet")
Step '6a end-to-end invariants, snapshot model' $cpu @('-I','-X','utf8',"$base\scripts\model\e2e_checks.py",'--store',$Store,'--model',$Name)
Step '6b end-to-end invariants, fallback model'  $cpu @('-I','-X','utf8',"$base\scripts\model\e2e_checks.py",'--store',$Store,'--model',$BasicName)
Step '7a unit self-tests (split manifest, metrics, item codes ...)' $cpu @('-I','-X','utf8',"$base\scripts\model\selftests\run_all.py")
if ($Mutation) {
  # deliberately breaks the code 24 ways (serving, features, grid, labels, metrics) in a scratch copy and requires a check to fail each time
  Step '7b mutation test of the safety net (slow)' $cpu @('-I','-X','utf8',"$base\scripts\model\mutation_test.py")
}
Write-Host "=== pipeline finished OK ($(Get-Date -Format HH:mm:ss))"
