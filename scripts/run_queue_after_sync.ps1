# Run the incremental tome queue only when the machine is calm.
#
# Measured on 2026-10-14: (a) a build and a second build stream or a big `aws s3 sync` on the same USB drive slow each other badly when the drive is
# saturated; (b) when the machine is short of memory (free RAM under ~2 GB, committed memory near the limit, 8,000-10,000 page-ins per second) a chunk
# build stalls for an hour with almost no CPU. This script therefore waits until
#   * free RAM is at least 4 GB and the hard page-in rate stays under 2,000 per second for 3 consecutive minutes, and
#   * (with -WaitAws) no aws.exe is writing to E:,
# and then runs run_tome_queue.ps1 for each collection (create logs\STOP_QUEUE to stop after the chunk being built; delete it to allow runs again) in -Days, one after the other. run_tome_queue.ps1 is incremental (builds only matches that are not
# in a finished tome), so starting it late or twice is safe; a second queue never starts while one is running.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_queue_after_sync.ps1 [-Days 2026-10-13,2026-10-11,2026-10-10] [-WaitAws]
param([string[]]$Days = @('2026-10-13', '2026-10-11', '2026-10-10'), [switch]$WaitAws, [int]$MaxWaitMinutes = 720)
$base = 'G:\pureskill\csds-cs2'
$log = "$base\logs\after_sync.log"
function Say($m) { "$(Get-Date -Format s) $m" | Add-Content -Encoding utf8 $log }
function Calm {
  $free = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
  $pin = (Get-Counter '\Memory\Pages Input/sec' -SampleInterval 3 -MaxSamples 3).CounterSamples | Measure-Object CookedValue -Average
  $awsBusy = $false
  if ($WaitAws) { $awsBusy = [bool](Get-Process aws -ErrorAction SilentlyContinue) }
  # my own heavy experiments count as "not calm": a build next to a model fit or a foundation-model run is how the machine ended up paging
  $exp = [bool](Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python' -and $_.CommandLine -match 'exp_foundation|prep_fm_data|exp_tune_xgb|exp_model_families|exp_robustness|export_model' })
  return (($free -ge 4.0) -and ($pin.Average -lt 2000) -and -not $awsBusy -and -not $exp)
}
Say "waiting for a calm machine before building: $($Days -join ', ') (WaitAws=$WaitAws)"
$calm = 0; $t0 = Get-Date
while ($calm -lt 3) {
  if (((Get-Date) - $t0).TotalMinutes -gt $MaxWaitMinutes) { Say 'gave up waiting for a calm machine'; exit 1 }
  if (Calm) { $calm++ } else { $calm = 0 }
  if ($calm -lt 3) { Start-Sleep -Seconds 50 }
}
Say 'machine calm for 3 minutes'
foreach ($d in $Days) {
  if (Test-Path -LiteralPath "$base\logs\STOP_QUEUE") { Say 'logs\STOP_QUEUE exists: not starting the next collection'; break }
  # never two queues at once: wait while another run_tome_queue.ps1 is running
  while (@(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object { $_.CommandLine -match 'run_tome_queue\.ps1' -and $_.ProcessId -ne $PID }).Count -gt 0) { Start-Sleep -Seconds 20 }
  Say "queue start: $d"
  & powershell -NoProfile -ExecutionPolicy Bypass -File "$base\scripts\run_tome_queue.ps1" -Only $d *>> "$base\logs\tome_queue_after_sync.log"
  Say "queue end: $d (exit $LASTEXITCODE)"
}
Say 'all done'
