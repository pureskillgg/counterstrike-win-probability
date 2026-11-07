# Fine-tune runs that survive a busy machine: waits until the 1 Hz status extraction has finished (so two heavy jobs of ours never compete for the
# machine's committed memory), then runs each fine-tune under scripts\fm\guarded_run.ps1 and RETRIES a run the guard killed (exit 99: the machine ran
# out of committed memory because other projects' jobs spiked) after a pause. Each run's GPU use is sampled every 20 s into logs\ft_gpu_<n>.csv, so
# "is it really on the GPU" has a record. Results: data\combined\exp_finetune.json (TabPFN v2), exp_finetune_tabicl.json (TabICL v2); log: logs\ft_retry.log
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\fm\run_ft_retry.ps1 [-NoWait] [-Which tabpfn,tabicl] [-Tries 4]
param([switch]$NoWait, [string]$Which = 'tabpfn,tabicl', [int]$Tries = 4, [int]$PauseMin = 6)
$b = 'G:\pureskill\csds-cs2'
$py = 'G:\pureskill\envs\tabicl\Scripts\python.exe'
$guard = "$b\scripts\fm\guarded_run.ps1"
$log = "$b\logs\ft_retry.log"
function Say($m) { try { "$(Get-Date -Format s) $m" | Add-Content -Encoding utf8 $log } catch { } }
if (-not $NoWait) {
  Say 'waiting for the final pipeline (scripts\run_final.ps1) to finish, so two heavy jobs of ours never compete for committed memory'
  $fl = "$b\logs\final.log"
  while ($true) {
    if ((Test-Path $fl) -and (Select-String -Path $fl -Pattern 'run_final done|did not finish' -Quiet)) { break }
    Start-Sleep -Seconds 60
  }
}
$runs = @{
  tabpfn = @{ limit = 14; script = 'finetune_tabpfn.py'; a = @('--version', 'v2', '--arms', 'finetune', '--epochs', '8', '--lr', '5e-5', '--ctx-query', '5000', '--ctx-rows', '10000', '--val-rows', '12000', '--n-test', '10000', '--per-round', '3', '--patience', '3') }
  tabicl = @{ limit = 16; script = 'finetune_tabicl.py'; a = @('--arms', 'finetune', '--epochs', '6', '--lr', '1e-5', '--chunk', '4000', '--max-train', '20000', '--ctx-rows', '10000', '--val-rows', '8000', '--n-test', '10000', '--per-round', '3', '--patience', '2') }
}
$n = 0
foreach ($name in ($Which -split ',')) {
  $r = $runs[$name]; if (-not $r) { continue }
  for ($try = 1; $try -le $Tries; $try++) {
    $n++
    $runLog = "$b\logs\ft_retry_${name}_$try.log"
    Say "start $name try $try"
    $gpuCsv = "$b\logs\ft_gpu_$n.csv"
    $sampler = Start-Process -FilePath powershell -WindowStyle Hidden -PassThru -ArgumentList @('-NoProfile', '-Command', "while(`$true){ (Get-Date -Format s) + ',' + (nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader) | Add-Content '$gpuCsv'; Start-Sleep 20 }")
    $argStr = '-X|utf8|' + "$b\scripts\fm\$($r.script)" + '|' + ($r.a -join '|')
    & powershell -NoProfile -ExecutionPolicy Bypass -File $guard -Exe $py -Log $runLog -LimitGB $r.limit -HeadroomGB 1.5 -ArgString $argStr
    Stop-Process -Id $sampler.Id -Force -ErrorAction SilentlyContinue
    $killed = (Get-Content "$runLog.guard" -ErrorAction SilentlyContinue | Select-String '\[guard\] KILL' -CaseSensitive | Select-Object -Last 1)
    $done = (Get-Content $runLog -ErrorAction SilentlyContinue | ForEach-Object { $_ -replace "`0", '' } | Select-String -Pattern 'fine-tuned' | Select-Object -Last 1)
    Say "end $name try $try; killed=$([bool]$killed); result line: $done"
    (Get-Content $runLog -ErrorAction SilentlyContinue | ForEach-Object { $_ -replace "`0", '' } | Select-String -Pattern ' LL |Traceback|Error' | Select-Object -Last 4) | ForEach-Object { Say "  $($_.Line)" }
    if (-not $killed) { break }
    Say "guard killed the run; pausing $PauseMin min before retry"
    Start-Sleep -Seconds ($PauseMin * 60)
  }
}
Say 'fine-tune retry script done'
