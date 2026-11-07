# Build basic tomes for 2026-10-09 and 2026-10-10 straight from the E: match folders (read-only junctions, no copy).
# One tome collection per day so an interrupted build only restarts that day. Nothing is written under E:.
# Channel list: the builder's default (every manifest channel except the per-tick ones), same as the 2026-10-01_to_08 build.
param([string[]]$Days = @('09', '10'))
$base = 'G:\pureskill\csds-cs2'
$src = 'E:\csds_final\csds\2026\10'
foreach ($day in $Days) {
  $stage = "$base\staging_2026_10_$day\csds\2026\10\$day"
  $tome = "$base\tomes\2026-10-$day"
  $log = "$base\logs\build_2026_10_$day.log"
  if (Test-Path "$tome\tome\csds\header\dataframe_00000") { "day $day already built, skipping"; continue }
  New-Item -ItemType Directory -Force -Path $stage | Out-Null
  $n = 0
  foreach ($d in Get-ChildItem -LiteralPath "$src\$day" -Directory) {
    $link = Join-Path $stage $d.Name
    if (-not (Test-Path -LiteralPath $link)) { New-Item -ItemType Junction -Path $link -Target $d.FullName | Out-Null }
    $n++
  }
  "day $day staged $n junctions at $(Get-Date -Format s)"
  New-Item -ItemType Directory -Force -Path $tome | Out-Null
  $env:PURESKILLGG_TOME_DS_COLLECTION_PATH = "$base\staging_2026_10_$day"
  $env:PURESKILLGG_TOME_COLLECTION_PATH = $tome
  uv run 'G:\ngclab_multi_spectral\build_basic_tomes.py' *> $log
  "day $day build exit $LASTEXITCODE at $(Get-Date -Format s)"
}
