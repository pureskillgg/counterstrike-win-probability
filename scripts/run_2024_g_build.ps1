$base = 'G:\pureskill\csds-cs2'
$stage = "$base\staging_2024_g\csds\2024\02"
New-Item -ItemType Directory -Force -Path "$stage\01" | Out-Null
foreach ($d in Get-ChildItem -LiteralPath "$base\merged_2024\csds\2024\02\01" -Directory) { New-Item -ItemType Junction -Path (Join-Path "$stage\01" $d.Name) -Target $d.FullName | Out-Null }
$rows = Import-Csv -LiteralPath "$base\logs\prescan_2024.csv" | Where-Object { $_.status -eq 'ok' -and $_.day -ne '01' }
foreach ($g in $rows) {
  $dayDir = Join-Path $stage $g.day
  New-Item -ItemType Directory -Force -Path $dayDir | Out-Null
  New-Item -ItemType Junction -Path (Join-Path $dayDir $g.match_folder) -Target "$base\source_2024\csds\2024\02\$($g.day)\$($g.match_folder)" | Out-Null
}
"staged day01=" + @(Get-ChildItem -LiteralPath "$stage\01" -Directory).Count
$chan = ((Get-Content -LiteralPath "$base\scripts\channels_build_nobom.txt") | Where-Object { $_.Trim() }) -join ','
New-Item -ItemType Directory -Force -Path "$base\tomes\2024-02-g" | Out-Null
$env:PURESKILLGG_TOME_DS_COLLECTION_PATH = "$base\staging_2024_g"
$env:PURESKILLGG_TOME_COLLECTION_PATH = "$base\tomes\2024-02-g"
uv run 'G:\ngclab_multi_spectral\build_basic_tomes.py' --channels $chan *> "$base\logs\build_2024_02_g.log"
"build exit: $LASTEXITCODE"
