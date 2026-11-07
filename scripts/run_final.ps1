# Final pipeline on the enlarged store, step after step, every step under scripts\fm\guarded_run.ps1 (kills a step that holds too much committed memory or
# leaves the machine without commit headroom) and RETRIED after a guard kill. A failed step does not stop the later ones (their inputs may still exist);
# the summary at the end lists every step with its outcome. Log: logs\final.log; per-step logs: logs\final_<step>.log (+ .err, .guard).
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_final.ps1 [-From cache] [-Only export_snapshot,robustness] [-Name winprob_v2] [-Stride 2]
# Steps, in order: archive cache export_basic export_snapshot inspect baselines robustness families
# The new models are written as winprob_v2* (the first models, winprob_v1*, stay untouched as a comparison and as the fixtures of the mutation test).
param([string]$From = 'archive', [string]$Only = '', [string]$Name = 'winprob_v2', [int]$Stride = 2, [int]$Tries = 3, [int]$PauseMin = 5, [switch]$WaitForExtraction)
$b = 'G:\pureskill\csds-cs2'
$cpu = "$b\.pixi\envs\default\python.exe"
$gpu = 'C:\Users\13144\.pixi\workspace-envs\cs2gpu-15282324412032665829\envs\default\python.exe'
$guard = "$b\scripts\fm\guarded_run.ps1"
$log = "$b\logs\final.log"
$basic = "${Name}_basic"
function Say($m) { try { "$(Get-Date -Format s) $m" | Add-Content -Encoding utf8 $log } catch { } }

function Run-Step($label, $exe, $argList, $limit) {
  $runLog = "$b\logs\final_$label.log"
  for ($try = 1; $try -le $Tries; $try++) {
    Remove-Item "$runLog", "$runLog.err", "$runLog.guard" -Force -ErrorAction SilentlyContinue
    Say "start $label (try $try, limit $limit GB)"
    $t0 = Get-Date
    & powershell -NoProfile -ExecutionPolicy Bypass -File $guard -Exe $exe -Log $runLog -LimitGB $limit -HeadroomGB 1.5 -ArgString ($argList -join '|')
    $code = $LASTEXITCODE
    $killed = [bool](Get-Content "$runLog.guard" -ErrorAction SilentlyContinue | Select-String '\[guard\] KILL' -CaseSensitive)
    $secs = [int]((Get-Date) - $t0).TotalSeconds
    if (-not $killed -and $code -eq 0) { Say "ok    $label ($secs s)"; return 'ok' }
    $tail = (Get-Content "$runLog.err" -ErrorAction SilentlyContinue | Select-Object -Last 3) -join ' | '
    Say "FAIL  $label (try $try, exit $code, guard-killed $killed, $secs s) $tail"
    if (-not $killed) { return "failed (exit $code)" }   # a real error is not retried; a guard kill (machine out of memory) is
    Start-Sleep -Seconds ($PauseMin * 60)
  }
  return 'killed by the memory guard on every try'
}

if ($WaitForExtraction) {
  Say 'waiting for the 1 Hz status extraction (logs\extract_status_oct15.log.guard) to finish'
  while (-not ((Test-Path "$b\logs\extract_status_oct15.log.guard") -and (Select-String -Path "$b\logs\extract_status_oct15.log.guard" -Pattern '\[guard\] (finished|KILL)' -CaseSensitive -Quiet))) { Start-Sleep -Seconds 60 }
  $extractTail = (Get-Content "$b\logs\extract_status_oct15.log" -ErrorAction SilentlyContinue | Where-Object { $_ -match 'done in|FAILED|Traceback' } | Select-Object -Last 3) -join ' | '
  Say "extraction ended: $extractTail"
  if ($extractTail -notmatch 'done in') { Say 'the extraction did not finish (killed or failed): stopping; rerun extract_status_1hz.py (it resumes), then this script'; exit 1 }
}
$steps = @(
  @{ n = 'cache';           exe = $cpu; limit = 9;  a = @('-I', '-X', 'utf8', "$b\scripts\model\make_snapshot_cache.py", '--store', 'combined') },
  @{ n = 'export_basic';    exe = $gpu; limit = 13; a = @('-I', '-X', 'utf8', "$b\scripts\model\export_model.py", '--store', 'combined', '--sets', 'map', '--name', $basic, '--stride', $Stride) },
  @{ n = 'export_snapshot'; exe = $gpu; limit = 14; a = @('-I', '-X', 'utf8', "$b\scripts\model\export_model.py", '--store', 'combined', '--from-cache', '--sets', 'map,status', '--eras', '2026', '--name', $Name, '--stride', $Stride) },
  @{ n = 'inspect';         exe = $cpu; limit = 6;  a = @('-I', '-X', 'utf8', "$b\scripts\model\inspect_rounds.py", '--pred', "$b\data\combined\export_${Name}_test_predictions.parquet", '--store', 'combined', '--out', "$b\data\combined\round_inspection_$Name") },
  @{ n = 'baselines';       exe = $gpu; limit = 13; a = @('-I', '-X', 'utf8', "$b\scripts\model\run_baselines.py", '--store', 'combined', '--seeds', '0,1') },
  @{ n = 'robustness';      exe = $gpu; limit = 13; a = @('-I', '-X', 'utf8', "$b\scripts\model\exp_robustness.py", '--store', 'combined', '--stride', '3') },
  @{ n = 'temporal';        exe = $gpu; limit = 13; a = @('-I', '-X', 'utf8', "$b\scripts\model\exp_temporal.py", '--store', 'combined', '--cutoff', '2026/10/10') },
  @{ n = 'families';        exe = $gpu; limit = 13; a = @('-I', '-X', 'utf8', "$b\scripts\model\exp_model_families.py", '--store', 'combined', '--stride', '6') }
)
$order = @('archive') + ($steps | ForEach-Object { $_.n })
$start = [array]::IndexOf($order, $From)
if ($start -lt 0) { throw "unknown step $From" }
$want = if ($Only) { $Only -split ',' } else { $order[$start..($order.Count - 1)] }
$result = [ordered]@{}
Say "run_final: steps $($want -join ', ')"

if ($want -contains 'archive') {
  # keep the results of the first (6,475-match) store: the new runs overwrite baselines.json, exp_robustness.json, exp_model_families.json ... under the same names
  $d = "$b\data\combined"; $arc = "$d\archive_store6475"
  New-Item -ItemType Directory -Force $arc | Out-Null
  foreach ($f in 'baselines.json', 'exp_robustness.json', 'exp_model_families.json', 'exp_stack_2026.json', 'exp_stack_2026b.json', 'prep_report.json', 'grid_meta.json') {
    if ((Test-Path "$d\$f") -and -not (Test-Path "$arc\$f")) { Copy-Item "$d\$f" "$arc\$f" }
  }
  if ((Test-Path "$d\cache_snapshot_2026.parquet") -and -not (Test-Path "$arc\cache_snapshot_2026.parquet")) { Move-Item "$d\cache_snapshot_2026.parquet" "$arc\cache_snapshot_2026.parquet" }
  Say "archived the first store's results to $arc"
  $result['archive'] = 'ok'
}
foreach ($s in $steps) {
  if ($want -notcontains $s.n) { continue }
  $result[$s.n] = Run-Step $s.n $s.exe $s.a $s.limit
}
Say 'SUMMARY'
foreach ($k in $result.Keys) { Say ("  {0,-16} {1}" -f $k, $result[$k]) }
Say 'run_final done'
