# New S3 days -> E: -> tomes. v3 (2026-10-14 15:40), strictly one E: job at a time.
#
# Why serial: E: is a USB HDD (WD Game Drive). A day-13 sync (123,068 small files) and a tome build (about 37 small reads per match) on
# it at the same time starve each other: the build sat 22 minutes with 6 s of CPU. So:
#   1. build day 12, 11, 10 in chunks of 500 matches (run_tome_queue.ps1 -Only ...; day 09 is already built)
#   2. then download S3 day 13 -> E:\csds_final\csds\2026\10\13 (copy only, no --delete; the user approved days 12 and 13, never 14)
#      completion is judged by comparing the S3 key list with the local files, never by waiting for the aws process (it can hang at exit)
#   3. only a VERIFIED day 13 is built (a half-synced day must never get a CHUNKS_DONE marker)
# Log: logs\new_days.log, sync log logs\aws_sync_13.log. Safe to rerun: sync and chunk builds resume.
$ErrorActionPreference = 'Continue'
$base = 'G:\pureskill\csds-cs2'
$bucket = 's3://pureskillgg-csgo-production-dataexchange-csds-0'
$e = 'E:\csds_final\csds'
$log = "$base\logs\new_days.log"
function Log([string]$m) { "$(Get-Date -Format s) $m" | Add-Content -Path $log -Encoding utf8 }

# E: is one USB disk: if ANY aws sync is running (the user runs their own in a terminal), wait for it instead of competing with it
function Wait-NoAws([string]$why) {
  $first = $true
  while (Get-Process -Name aws -ErrorAction SilentlyContinue) {
    if ($first) { Log "an aws process is running (not ours): waiting before $why"; $first = $false }
    Start-Sleep -Seconds 60
  }
  if (-not $first) { Log "aws is idle again: continuing with $why" }
}

Log "v3 start: waiting for $e"
while (-not (Test-Path -LiteralPath $e)) { Start-Sleep -Seconds 30 }
Log "E: is ready"

foreach ($name in '2026-10-12', '2026-10-11', '2026-10-10') {
  Wait-NoAws "build $name"
  Log "build $name"
  & powershell -NoProfile -ExecutionPolicy Bypass -File "$base\scripts\run_tome_queue.ps1" -Only $name *>> "$base\logs\tome_queue.log"
  Log "build $name step finished (CHUNKS_DONE=$(Test-Path "$base\tomes\$name.CHUNKS_DONE"))"
}

# ---- day 13: download, verify against S3, build
$day = '13'
$prefix = "csds/2026/10/$day/"
function S3-Keys { & aws s3 ls "$bucket/$prefix" --recursive --profile psgg --no-cli-pager 2>&1 | ForEach-Object { if ($_ -match '^\S+\s+\S+\s+\d+\s+(.+)$') { $matches[1] } } }
function Local-Keys { $root = 'E:\csds_final\'; $set = New-Object 'System.Collections.Generic.HashSet[string]'
  if (Test-Path -LiteralPath "$e\2026\10\$day") { foreach ($d in Get-ChildItem -LiteralPath "$e\2026\10\$day" -Directory) { foreach ($f in [System.IO.Directory]::GetFiles($d.FullName)) { [void]$set.Add($f.Substring($root.Length).Replace('\', '/')) } }
    foreach ($f in [System.IO.Directory]::GetFiles("$e\2026\10\$day")) { [void]$set.Add($f.Substring($root.Length).Replace('\', '/')) } }
  return $set }

Wait-NoAws "the day-13 download"
$s3 = @(S3-Keys)
Log "day 13: $($s3.Count) objects on S3"
$verified = $false
if ($s3.Count -eq 0) {
  Log "day 13: S3 listing empty; not downloading"
} elseif ((Get-PSDrive -Name E).Free -lt (74GB + 40GB)) {
  Log ("day 13: NOT downloading - E: has {0:N0} GB free" -f ((Get-PSDrive -Name E).Free / 1GB))
} else {
  for ($try = 1; $try -le 4 -and -not $verified; $try++) {
    # (no New-Item on E:: aws creates the destination folder itself, and that is the only write we make there)
    $p = Start-Process -FilePath 'aws' -ArgumentList @('s3', 'sync', "$bucket/$prefix", "$e\2026\10\$day\", '--profile', 'psgg', '--no-cli-pager', '--only-show-errors') `
      -RedirectStandardOutput "$base\logs\aws_sync_13.log" -RedirectStandardError "$base\logs\aws_sync_13.err" -PassThru -WindowStyle Hidden
    Log "day 13: sync attempt $try started (pid $($p.Id))"
    $t0 = Get-Date
    while ($true) {
      Start-Sleep -Seconds 180
      $alive = [bool](Get-Process -Id $p.Id -ErrorAction SilentlyContinue)
      $local = Local-Keys
      $missing = @($s3 | Where-Object { -not $local.Contains($_) }).Count
      Log "day 13: $($local.Count) files on E:, $missing of $($s3.Count) still missing, aws alive=$alive"
      if (-not $alive) { break }
      if ($missing -eq 0) { Log "day 13: all keys present, aws still lingering: stopping it"; Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue; break }
      if (((Get-Date) - $t0).TotalHours -gt 8) { Log "day 13: attempt $try timed out"; Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue; break }
    }
    $local = Local-Keys
    $missing = @($s3 | Where-Object { -not $local.Contains($_) })
    $s3set = New-Object 'System.Collections.Generic.HashSet[string]'; foreach ($k in $s3) { [void]$s3set.Add($k) }
    $extra = @($local | Where-Object { -not $s3set.Contains($_) })
    Log "day 13 check $try : missing $($missing.Count), extra local files $($extra.Count)"
    if ($extra.Count -gt 0) { $extra | Select-Object -First 20 | ForEach-Object { Log "  extra: $_" } }
    # extra files are leftover temp files of an aws run that was stopped mid-transfer; they are listed above, never deleted (E: is read-only for
    # us apart from the sync itself), and harmless: the builder reads only the channels each manifest names, and nothing is missing
    if ($missing.Count -eq 0) { $verified = $true }
  }
}
if ($verified) {
  Log "day 13 VERIFIED against S3; building"
  & powershell -NoProfile -ExecutionPolicy Bypass -File "$base\scripts\run_tome_queue.ps1" -Only 2026-10-13 *>> "$base\logs\tome_queue.log"
  Log "build 2026-10-13 step finished (CHUNKS_DONE=$(Test-Path "$base\tomes\2026-10-13.CHUNKS_DONE"))"
} else {
  Log "day 13 NOT verified: not built (rerun this script after fixing; missing or extra files are listed above)"
}
Log "run_new_days v3 done"
