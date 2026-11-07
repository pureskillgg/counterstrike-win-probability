# Download S3 day 13 -> E:\csds_final\csds\2026\10\13 (approved by the user on 2026-10-14; NEVER day 14; copy only, no --delete), verify it against the
# S3 key list (never by waiting for the aws process: it can hang at exit), and only then build it. Run AFTER the status extraction (E: is a USB disk).
#   powershell -File run_day13_download.ps1 [-Build]
param([switch]$Build)
$ErrorActionPreference = 'Continue'
$base = 'G:\pureskill\csds-cs2'
$bucket = 's3://pureskillgg-csgo-production-dataexchange-csds-0'
$e = 'E:\csds_final\csds'
$log = "$base\logs\new_days.log"
function Log([string]$m) { "$(Get-Date -Format s) $m" | Add-Content -Path $log -Encoding utf8 }
$day = '13'
$prefix = "csds/2026/10/$day/"
function S3-Keys { & aws s3 ls "$bucket/$prefix" --recursive --profile psgg --no-cli-pager 2>&1 | ForEach-Object { if ($_ -match '^\S+\s+\S+\s+\d+\s+(.+)$') { $matches[1] } } }
function Local-Keys { $root = 'E:\csds_final\'; $set = New-Object 'System.Collections.Generic.HashSet[string]'
  if (Test-Path -LiteralPath "$e\2026\10\$day") { foreach ($d in Get-ChildItem -LiteralPath "$e\2026\10\$day" -Directory) { foreach ($f in [System.IO.Directory]::GetFiles($d.FullName)) { [void]$set.Add($f.Substring($root.Length).Replace('\', '/')) } }
    foreach ($f in [System.IO.Directory]::GetFiles("$e\2026\10\$day")) { [void]$set.Add($f.Substring($root.Length).Replace('\', '/')) } }
  return $set }
$s3 = @(S3-Keys)
Log "day 13 download: $($s3.Count) objects on S3"
$verified = $false
if ($s3.Count -eq 0) { Log "day 13: S3 listing empty; nothing to do" }
elseif ((Get-PSDrive -Name E).Free -lt (74GB + 40GB)) { Log ("day 13: NOT downloading - E: has {0:N0} GB free" -f ((Get-PSDrive -Name E).Free / 1GB)) }
else {
  for ($try = 1; $try -le 4 -and -not $verified; $try++) {
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
    Log "day 13 check $try : missing $($missing.Count)"
    if ($missing.Count -eq 0) { $verified = $true }
  }
}
if ($verified -and $Build) {
  Log "day 13 VERIFIED against S3; building"
  & powershell -NoProfile -ExecutionPolicy Bypass -File "$base\scripts\run_tome_queue.ps1" -Only 2026-10-13 *>> "$base\logs\tome_queue.log"
  Log "build 2026-10-13 finished (CHUNKS_DONE=$(Test-Path "$base\tomes\2026-10-13.CHUNKS_DONE"))"
} elseif ($verified) { Log "day 13 VERIFIED against S3 (not built: run with -Build or run_tome_queue.ps1 -Only 2026-10-13)" }
else { Log "day 13 NOT verified: not built" }
