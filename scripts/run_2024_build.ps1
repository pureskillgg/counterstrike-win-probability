$base = 'G:\pureskill\csds-cs2'
$csv = "$base\logs\prescan_2024.csv"
while (-not (Test-Path -LiteralPath $csv)) { Start-Sleep -Seconds 20 }
Start-Sleep -Seconds 5
$rows = Import-Csv -LiteralPath $csv
$good = foreach ($r in $rows) {
  if ($r.status -eq 'ok') { $r; continue }
  if ($r.status -eq 'problem') {
    $items = $r.detail -split ';' | Where-Object { $_ -ne '' }
    $others = @($items | Where-Object { -not $_.StartsWith('player_inputs:') })
    if ($others.Count -eq 0) { $r }
  }
}
"prescan rows: $($rows.Count); usable for the build: $(@($good).Count)"
$stage = "$base\staging_2024\csds\2024\02"
$src = 'E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02'
foreach ($g in $good) {
  $dayDir = Join-Path $stage $g.day
  New-Item -ItemType Directory -Force -Path $dayDir | Out-Null
  $link = Join-Path $dayDir $g.match_folder
  if (-not (Test-Path -LiteralPath $link)) { New-Item -ItemType Junction -Path $link -Target (Join-Path $src "$($g.day)\$($g.match_folder)") | Out-Null }
}
"staged junctions: " + @(Get-ChildItem -LiteralPath $stage -Directory | ForEach-Object { Get-ChildItem -LiteralPath $_.FullName -Directory }).Count
$chan = ((Get-Content -LiteralPath "$base\scripts\channels_nonpertick.txt") | Where-Object { $_.Trim() -and $_.Trim() -ne 'player_inputs' }) -join ','
New-Item -ItemType Directory -Force -Path "$base\tomes\2024-02" | Out-Null
$env:PURESKILLGG_TOME_DS_COLLECTION_PATH = "$base\staging_2024"
$env:PURESKILLGG_TOME_COLLECTION_PATH = "$base\tomes\2024-02"
uv run 'G:\ngclab_multi_spectral\build_basic_tomes.py' --channels $chan *> "$base\logs\build_2024_02.log"
"build exit: $LASTEXITCODE"
