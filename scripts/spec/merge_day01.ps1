$base = 'G:\pureskill\csds-cs2'
$roots = @{
  A = 'E:\csds_extracted\csds\2024\02\01'
  C = 'E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02\01'
}
$dstRoot = "$base\merged_2024\csds\2024\02\01"
$rows = Import-Csv -LiteralPath "$base\notes\spec\merge_manifest_day01.csv" | Where-Object { $_.action -ne 'skip_duplicate' }
$copied = 0; $skipped = 0; $bad = New-Object System.Collections.Generic.List[string]
$i = 0
foreach ($r in $rows) {
  $i++
  $src = Join-Path $roots[$r.source_copy] $r.relative_path
  $dst = Join-Path $dstRoot $r.relative_path
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dst) | Out-Null
  if (Test-Path -LiteralPath $dst) { $skipped++ }
  else {
    try {
      [IO.File]::Copy($src, $dst, $false)
      $copied++
    } catch { $bad.Add("copy-fail: $($r.relative_path): $($_.Exception.Message)") }
  }
  if (Test-Path -LiteralPath $dst) {
    $len = (Get-Item -LiteralPath $dst).Length
    if ($len -ne [int64]$r.size_bytes) { $bad.Add("size-mismatch: $($r.relative_path) got $len want $($r.size_bytes)") }
  } else { $bad.Add("missing-after-copy: $($r.relative_path)") }
  if ($i % 500 -eq 0) { "progress $i/$($rows.Count) copied=$copied already_present=$skipped t=$(Get-Date -Format HH:mm:ss)" }
}
"done: rows=$($rows.Count) copied=$copied already_present=$skipped problems=$($bad.Count)"
$bad | Out-File -LiteralPath "$base\logs\merge_day01_problems.txt" -Encoding utf8
