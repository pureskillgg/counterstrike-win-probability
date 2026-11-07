# Fine-tuning suite on the GPU, one run after the other, EVERY run under scripts\fm\guarded_run.ps1 (kills a run that holds too much committed memory or
# leaves the machine with too little). Results: data\combined\exp_finetune.json (TabPFN v2), exp_finetune_tabicl.json (TabICL v2); log: logs\ft_suite.log.
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\fm\run_ft_suite.ps1
$b = 'G:\pureskill\csds-cs2'
$py = 'G:\pureskill\envs\tabicl\Scripts\python.exe'
$guard = "$b\scripts\fm\guarded_run.ps1"
$log = "$b\logs\ft_suite.log"
function Say($m) { try { "$(Get-Date -Format s) $m" | Add-Content -Encoding utf8 $log } catch { } }
$runs = @(
  @{ name = 'tabpfn v2 lr5e-5 ep8';    limit = 12; script = 'finetune_tabpfn.py'; a = @('--version', 'v2', '--arms', 'finetune', '--epochs', '8', '--lr', '5e-5', '--ctx-query', '5000', '--ctx-rows', '10000', '--val-rows', '12000', '--n-test', '10000', '--per-round', '3', '--patience', '3') },
  @{ name = 'tabicl lr1e-5 ep6 20k';   limit = 10; script = 'finetune_tabicl.py';  a = @('--epochs', '6', '--lr', '1e-5', '--chunk', '4000', '--max-train', '20000', '--ctx-rows', '10000', '--val-rows', '8000', '--n-test', '10000', '--per-round', '3', '--patience', '2') },
  @{ name = 'tabicl lr5e-5 ep8 20k';   limit = 10; script = 'finetune_tabicl.py';  a = @('--arms', 'finetune', '--epochs', '8', '--lr', '5e-5', '--chunk', '4000', '--max-train', '20000', '--ctx-rows', '10000', '--val-rows', '8000', '--n-test', '10000', '--per-round', '3', '--patience', '3') }
)
$i = 0
foreach ($r in $runs) {
  $i++
  Say "start: $($r.name)"
  $runLog = "$b\logs\ft_suite_run$i.log"
  $args = '-X|utf8|' + "$b\scripts\fm\$($r.script)" + '|' + ($r.a -join '|')
  & powershell -NoProfile -ExecutionPolicy Bypass -File $guard -Exe $py -Log $runLog -LimitGB $r.limit -HeadroomGB 2 -ArgString $args
  $code = $LASTEXITCODE
  Say "end: $($r.name) (exit $code)"
  (Get-Content $runLog -ErrorAction SilentlyContinue | ForEach-Object { $_ -replace "`0",'' } | Select-String -Pattern ' LL |train rows|Traceback|Error' | Select-Object -Last 4) | ForEach-Object { Say "  $($_.Line)" }
}
Say 'suite done'
