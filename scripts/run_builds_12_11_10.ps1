# Builds only (no download): tomes for S3 days 12, 11, 10 in chunks of 500 matches (day 09 is already built). Day 13 is downloaded separately
# (run_day13_download.ps1) AFTER the status extraction, so the two never compete for the E: USB disk.
# The user told us (2026-10-14 16:05) not to wait for other aws syncs: no gate here. Chunk builds are resumable (BUILD_OK per chunk).
#   powershell -File run_builds_12_11_10.ps1 [-WaitPid <pid of a queue instance that is already building>]
param([int]$WaitPid = 0)
$ErrorActionPreference = 'Continue'
$base = 'G:\pureskill\csds-cs2'
$log = "$base\logs\new_days.log"
function Log([string]$m) { "$(Get-Date -Format s) $m" | Add-Content -Path $log -Encoding utf8 }
if ($WaitPid -gt 0) {
  Log "builds v4: waiting for the queue instance $WaitPid that is already building"
  while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 20 }
}
foreach ($name in '2026-10-12', '2026-10-11', '2026-10-10') {
  Log "build $name"
  & powershell -NoProfile -ExecutionPolicy Bypass -File "$base\scripts\run_tome_queue.ps1" -Only $name *>> "$base\logs\tome_queue.log"
  Log "build $name finished (CHUNKS_DONE=$(Test-Path "$base\tomes\$name.CHUNKS_DONE"))"
}
Log "builds v4 done: days 12, 11, 10 (day 09 was built earlier). Next: build_combined -> prepare -> make_grid -> extract_status_1hz (E: reads), then the day-13 download."
