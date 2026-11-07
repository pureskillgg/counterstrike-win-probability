# Foundation-model head-to-head, one run after the other (never two heavy jobs at once). Waits for prep_fm_data.py to finish first.
# Results: data\combined\exp_foundation.json (one entry per run); log: logs\fm_suite.log
# The suite runs on the CPU: on 2026-10-14 the RTX 3080 went into an error state (nvidia-smi and PyTorch cannot reach it) and the CPU is shared with
# other jobs, so the runs start small (few ensemble members, 1,000 test seconds) and grow while they stay affordable. Add --device cuda runs back
# when the GPU is reset.
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\fm\run_fm_suite.ps1
$b = 'G:\pureskill\csds-cs2'
$log = "$b\logs\fm_suite.log"
function Say($m) { try { "$(Get-Date -Format s) $m" | Add-Content -Encoding utf8 $log } catch { } }
while (-not (Test-Path "$b\data\combined\fm\meta.json")) { Start-Sleep -Seconds 10 }
$tabicl = 'G:\pureskill\envs\tabicl\Scripts\python.exe'
$mitra = 'G:\pureskill\envs\mitra\Scripts\python.exe'
$runs = @(
  @{ py = $tabicl; a = @('--model', 'tabicl', '--ctx', '5000', '--device', 'cpu', '--n-test', '1000', '--n-estimators', '4') },
  @{ py = $tabicl; a = @('--model', 'tabicl', '--ctx', '10000', '--device', 'cpu', '--n-test', '1000', '--n-estimators', '4') },
  @{ py = $mitra;  a = @('--model', 'mitra', '--ctx', '5000', '--device', 'cpu', '--n-test', '1000') },
  @{ py = $tabicl; a = @('--model', 'tabicl', '--ctx', '10000', '--device', 'cpu', '--n-test', '4000', '--n-estimators', '8') },
  @{ py = $tabicl; a = @('--model', 'tabicl', '--ctx', '30000', '--device', 'cpu', '--n-test', '4000', '--n-estimators', '4') },
  @{ py = $mitra;  a = @('--model', 'mitra', '--ctx', '10000', '--device', 'cpu', '--n-test', '2000') }
)
foreach ($r in $runs) {
  Say "start: $($r.a -join ' ')"
  $env:MKL_THREADING_LAYER = 'SEQUENTIAL'
  & $r.py -X utf8 "$b\scripts\fm\exp_foundation.py" @($r.a) 2>&1 | ForEach-Object { Add-Content -Encoding utf8 "$b\logs\fm_suite_full.log" $_; if ($_ -match 'LL |Error|Traceback|rows/s|context|XGBoost') { Say "  $_" } }
  Say "end (exit $LASTEXITCODE)"
}
Say 'suite done'
