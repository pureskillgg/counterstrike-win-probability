# Release B: fold the newly built tome sources (day 13 chunks, and later extensions) into the store and refresh everything the models read, LOCALLY (all steps are
# I/O or a few GB; the heavy GPU fits go to a pod afterwards with scripts\runpod\make_pod_bundle.py). Every step runs under scripts\fm\guarded_run.ps1 and the
# chain STOPS at the first failure (a later step on a half-updated store would mislead). Log: logs\release_b.log.
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_release_b.ps1 [-Sources 2026-10-13] [-From append]
# Steps: append prepare grid split extract cache
param([string]$Sources = '2026-10-13', [string]$From = 'append', [int]$Tries = 2)
$b = 'G:\pureskill\csds-cs2'
$cpu = "$b\.pixi\envs\default\python.exe"
$guard = "$b\scripts\fm\guarded_run.ps1"
$log = "$b\logs\release_b.log"
function Say($m) { try { "$(Get-Date -Format s) $m" | Add-Content -Encoding utf8 $log } catch { } }
function Run-Step($label, $argList, $limit) {
  $runLog = "$b\logs\release_b_$label.log"
  for ($try = 1; $try -le $Tries; $try++) {
    Remove-Item "$runLog", "$runLog.err", "$runLog.guard" -Force -ErrorAction SilentlyContinue
    Say "start $label (try $try, limit $limit GB)"
    $t0 = Get-Date
    & powershell -NoProfile -ExecutionPolicy Bypass -File $guard -Exe $cpu -Log $runLog -LimitGB $limit -HeadroomGB 1.5 -ArgString ($argList -join '|')
    $code = $LASTEXITCODE
    $killed = [bool](Get-Content "$runLog.guard" -ErrorAction SilentlyContinue | Select-String '\[guard\] KILL' -CaseSensitive)
    $secs = [int]((Get-Date) - $t0).TotalSeconds
    if (-not $killed -and $code -eq 0) { Say "ok    $label ($secs s)"; return $true }
    Say "FAIL  $label (try $try, exit $code, guard-killed $killed, $secs s) $((Get-Content "$runLog.err" -ErrorAction SilentlyContinue | Select-Object -Last 2) -join ' | ')"
    if (-not $killed) { return $false }
    Start-Sleep -Seconds 180
  }
  return $false
}
$steps = @(
  @{ n = 'append';  limit = 9;  a = @('-I', '-X', 'utf8', "$b\scripts\data\append_combined.py", '--sources', $Sources) },
  @{ n = 'prepare'; limit = 13; a = @('-I', '-X', 'utf8', "$b\scripts\data\prepare.py", '--store', 'combined') },
  @{ n = 'grid';    limit = 12; a = @('-I', '-X', 'utf8', "$b\scripts\data\make_grid.py", '--store', 'combined') },
  @{ n = 'split';   limit = 6;  a = @('-I', '-X', 'utf8', "$b\scripts\model\split_manifest.py", '--store', 'combined', '--extend') },
  @{ n = 'extract'; limit = 10; a = @('-I', '-X', 'utf8', "$b\scripts\data\extract_status_1hz.py", '--store', 'combined', '--threads', '4') },
  @{ n = 'cache';   limit = 9;  a = @('-I', '-X', 'utf8', "$b\scripts\model\make_snapshot_cache.py", '--store', 'combined', '--chunks', '12') }
)
$order = $steps | ForEach-Object { $_.n }
$start = [array]::IndexOf($order, $From)
if ($start -lt 0) { throw "unknown step $From" }
Say "release B: sources $Sources, steps $($order[$start..($order.Count - 1)] -join ', ')"
foreach ($s in $steps[$start..($steps.Count - 1)]) {
  if (-not (Run-Step $s.n $s.a $s.limit)) { Say "STOPPED at $($s.n): later steps would run on a half-updated store"; exit 1 }
}
Say 'release B local steps done: next, build the pod bundle (scripts\runpod\make_pod_bundle.py) and run the chain as winprob_v3'
