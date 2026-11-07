# Run a command with a memory guard: if the process (or any child) commits more than -LimitGB of private memory, or Windows' committed memory gets within
# -HeadroomGB of its limit, the process is killed and the reason is logged. On 2026-10-15 an unguarded TabICL fine-tune reached 24 GB of committed memory and
# took the whole machine to 78.7 of 79.3 GB (other projects' jobs were running too), so every long job of mine now goes through this guard.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\fm\guarded_run.ps1 -Exe <exe> -ArgString "<arg1>|<arg2>|..." -Log <file> [-LimitGB 12] [-HeadroomGB 3]
# Output: stdout/stderr go to <Log> and <Log>.err; the exit code is the program's, or 99 when the guard killed it.
param([Parameter(Mandatory)][string]$Exe, [string]$ArgString = "", [Parameter(Mandatory)][string]$Log, [double]$LimitGB = 12, [double]$HeadroomGB = 2)
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
$ArgList = @($ArgString -split '\|' | Where-Object { $_ })
$p = Start-Process -FilePath $Exe -ArgumentList $ArgList -RedirectStandardOutput $Log -RedirectStandardError "$Log.err" -WindowStyle Hidden -PassThru
$null = $p.Handle   # caches the process handle: without it ExitCode comes back empty after the process ended (every run used to report "exit code " and exit 0)
function Say($m) { "$(Get-Date -Format s) [guard] $m" | Add-Content -Encoding utf8 "$Log.guard" }
Say "started pid $($p.Id): $Exe $($ArgList -join ' ') (limit $LimitGB GB private, $HeadroomGB GB commit headroom)"
while (-not $p.HasExited) {
  Start-Sleep -Seconds 15
  $p.Refresh()
  if ($p.HasExited) { break }
  # the started process may be a launcher (a venv python shim): add up the whole process tree
  $ids = @($p.Id); $all = Get-CimInstance Win32_Process | Select-Object ProcessId, ParentProcessId
  for ($i = 0; $i -lt 4; $i++) { $ids = @($ids + ($all | Where-Object { $ids -contains $_.ParentProcessId } | ForEach-Object { $_.ProcessId }) | Select-Object -Unique) }
  $priv = ($ids | ForEach-Object { (Get-Process -Id $_ -ErrorAction SilentlyContinue).PrivateMemorySize64 } | Measure-Object -Sum).Sum / 1GB
  $os = Get-CimInstance Win32_OperatingSystem
  $left = ($os.TotalVirtualMemorySize - ($os.TotalVirtualMemorySize - $os.FreeVirtualMemory)) / 1MB   # GB of commit still available
  if ($priv -gt $LimitGB) { Say "KILL: private memory $([math]::Round($priv,1)) GB > limit"; Stop-Process -Id $p.Id -Force; exit 99 }
  # only a job that is itself big (> 3 GB) is killed for the machine running short: other projects' spikes must not kill a small job
  if ($left -lt $HeadroomGB -and $priv -gt 3) {
    Say "KILL: only $([math]::Round($left,1)) GB of commit left on the machine (process holds $([math]::Round($priv,1)) GB)"
    Stop-Process -Id $p.Id -Force
    exit 99
  }
}
Say "finished, exit code $($p.ExitCode)"
exit $p.ExitCode

