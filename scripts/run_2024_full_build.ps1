$base = 'G:\pureskill\csds-cs2'
$stage = "$base\staging_2024_full\csds\2024\02"
New-Item -ItemType Directory -Force -Path $stage | Out-Null
# day 01: every merged match folder (230)
$m = "$base\merged_2024\csds\2024\02\01"
New-Item -ItemType Directory -Force -Path "$stage\01" | Out-Null
foreach ($d in Get-ChildItem -LiteralPath $m -Directory) {
  $link = Join-Path "$stage\01" $d.Name
  if (-not (Test-Path -LiteralPath $link)) { New-Item -ItemType Junction -Path $link -Target $d.FullName | Out-Null }
}
# days 02-25: clean matches from the pre-scan (same selection as the first 2024 build)
$src = 'E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02'
$rows = Import-Csv -LiteralPath "$base\logs\prescan_2024.csv" | Where-Object { $_.status -eq 'ok' -and $_.day -ne '01' }
foreach ($g in $rows) {
  $dayDir = Join-Path $stage $g.day
  New-Item -ItemType Directory -Force -Path $dayDir | Out-Null
  $link = Join-Path $dayDir $g.match_folder
  if (-not (Test-Path -LiteralPath $link)) { New-Item -ItemType Junction -Path $link -Target (Join-Path $src "$($g.day)\$($g.match_folder)") | Out-Null }
}
"staged day01=" + @(Get-ChildItem -LiteralPath "$stage\01" -Directory).Count + " other days=" + @(Get-ChildItem -LiteralPath $stage -Directory | Where-Object { $_.Name -ne '01' } | ForEach-Object { Get-ChildItem -LiteralPath $_.FullName -Directory }).Count
$chan = ((Get-Content -LiteralPath "$base\scripts\channels_build_nobom.txt") | Where-Object { $_.Trim() }) -join ','
New-Item -ItemType Directory -Force -Path "$base\tomes\2024-02-all" | Out-Null
$env:PURESKILLGG_TOME_DS_COLLECTION_PATH = "$base\staging_2024_full"
$env:PURESKILLGG_TOME_COLLECTION_PATH = "$base\tomes\2024-02-all"
uv run 'G:\ngclab_multi_spectral\build_basic_tomes.py' --channels $chan *> "$base\logs\build_2024_02_all.log"
"build exit: $LASTEXITCODE"
