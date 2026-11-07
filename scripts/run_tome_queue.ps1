# Resumable, INCREMENTAL queue that builds basic tomes from E: match folders (read-only junctions, nothing written under E:).
#
# Data flow: the raw CSDS match folders live on E: (a big USB drive) and are only READ; the finished tomes are written to G:\pureskill\csds-cs2\tomes
# (the builder recommends a different disk for the output, and writing to the same USB drive would slow the reads that dominate the build).
#
# INCREMENTAL: before building a day the queue asks scripts\data\tome_ledger.py which matches are already inside a finished tome of that day
# (the header keysets of tomes\<name> and tomes\<name>_cNN that have a BUILD_OK marker) and builds ONLY the matches that are not. A day that
# is already complete costs one directory listing; a day that grew (new matches synced since the last run) gets new chunks numbered after the
# existing ones; the finished chunks are never touched or re-sliced. (The dsdk builder itself cannot do this: a complete header tome is a
# frozen match set and an interrupted build deletes its staging folder - so the incremental layer lives here, and append_combined.py does the
# same one level up for the combined store.)
#
# A day is built in CHUNKS of -ChunkSize matches (default 500). Each chunk is its own small tome collection `tomes\<name>_cNN` with its own
# BUILD_OK marker, so a crash, a reboot or a dropped drive costs at most one chunk (~8 minutes). When a day has no unbuilt match left,
# `tomes\<name>.CHUNKS_DONE` lists all its collections (the whole one, if any, and every chunk) and build_combined.py / append_combined.py
# merge them as sources. If E: is not ready the queue waits for it and retries a failed chunk 3 times.
#
# Usage: powershell -File run_tome_queue.ps1                       (default queue, in the order below)
#        powershell -File run_tome_queue.ps1 -Only 2026-10-13      (one collection)
#        powershell -File run_tome_queue.ps1 -DryRun               (print what would be built, build nothing)
param([string[]]$Only = @(), [int]$ChunkSize = 500, [switch]$DryRun)
[System.Diagnostics.Process]::GetCurrentProcess().PriorityClass = 'BelowNormal'
$base = 'G:\pureskill\csds-cs2'
$src = 'E:\csds_final\csds'
$py = "$base\.pixi\envs\default\python.exe"
$days30 = 1..30 | ForEach-Object { '{0:D2}' -f $_ }
# name -> year, month, days. Order requested by the user: 12 and 13 first, then backwards from 11 (09 and 2026-09 are already built).
$queue = [ordered]@{
  '2026-10-12' = @{ yyyy = '2026'; mm = '10'; days = @('12') }
  '2026-10-13' = @{ yyyy = '2026'; mm = '10'; days = @('13') }
  '2026-10-11' = @{ yyyy = '2026'; mm = '10'; days = @('11') }
  '2026-10-10' = @{ yyyy = '2026'; mm = '10'; days = @('10') }
  '2026-10-09' = @{ yyyy = '2026'; mm = '10'; days = @('09') }
  '2026-09'    = @{ yyyy = '2026'; mm = '09'; days = $days30 }
}

function Wait-Drive([string]$path, [int]$maxMinutes = 720) {
  $t0 = Get-Date
  while (-not (Test-Path -LiteralPath $path)) {
    if (((Get-Date) - $t0).TotalMinutes -gt $maxMinutes) { return $false }
    "$(Get-Date -Format s) waiting for $path (drive not ready)"
    Start-Sleep -Seconds 30
  }
  return $true
}

# Create (or repair) a directory junction. A junction left dangling by a dropped drive is removed with the NON-recursive
# Directory.Delete, which removes only the link and never touches the target's contents.
function Set-Junction([string]$link, [string]$target) {
  $it = Get-Item -LiteralPath $link -Force -ErrorAction SilentlyContinue
  if ($it) {
    if (@($it.Target) -contains $target) { return }
    [System.IO.Directory]::Delete($link)
  }
  New-Item -ItemType Junction -Path $link -Target $target | Out-Null
}

# One queue per collection name at a time: a second run would pick the same next chunk and delete the tome the first one is writing.
# The lock file holds the owner's process id; a lock whose process is gone is stale and is taken over.
function Enter-Lock([string]$name) {
  $lock = "$base\tomes\$name.LOCK"
  if (Test-Path -LiteralPath $lock) {
    $owner = 0; [void][int]::TryParse(((Get-Content -LiteralPath $lock -ErrorAction SilentlyContinue | Select-Object -First 1) -as [string]), [ref]$owner)
    if ($owner -gt 0 -and (Get-Process -Id $owner -ErrorAction SilentlyContinue)) { return $false }
    Remove-Item -LiteralPath $lock -Force -ErrorAction SilentlyContinue  # stale
  }
  try { $fs = [System.IO.File]::Open($lock, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::Read) } catch { return $false }
  $bytes = [System.Text.Encoding]::ASCII.GetBytes("$PID"); $fs.Write($bytes, 0, $bytes.Length); $fs.Close()
  return $true
}
function Exit-Lock([string]$name) { Remove-Item -LiteralPath "$base\tomes\$name.LOCK" -Force -ErrorAction SilentlyContinue }
# Create logs\STOP_QUEUE to make the queue stop after the chunk it is building (delete the file to allow runs again).
function Stop-Requested { return (Test-Path -LiteralPath "$base\logs\STOP_QUEUE") }
# tomes\<name>.CHUNKS_DONE lists every finished collection of the name (the whole one if it was built whole, and all chunks, in order). Written to a
# temporary file and moved into place, so a crash never leaves an empty or half-written marker (an empty one would drop the day from the combined store).
function Write-ChunksDone([string]$name) {
  $names = @()
  if (Test-Path "$base\tomes\$name\BUILD_OK") { $names += $name }
  $names += @(Get-ChildItem "$base\tomes" -Directory -Filter "${name}_c*" | Where-Object { $_.Name -match '_c\d+$' -and (Test-Path "$($_.FullName)\BUILD_OK") } | Sort-Object Name | ForEach-Object { $_.Name })
  if ($names.Count -eq 0) { return }
  $tmp = "$base\tomes\$name.CHUNKS_DONE.tmp"
  Set-Content -LiteralPath $tmp -Value ($names -join "`n")
  Move-Item -LiteralPath $tmp -Destination "$base\tomes\$name.CHUNKS_DONE" -Force
  "$name : all collections built ($($names -join ', ')) at $(Get-Date -Format s)"
}

foreach ($name in $queue.Keys) {
  if ($Only.Count -gt 0 -and $Only -notcontains $name) { continue }
  $q = $queue[$name]
  if (Stop-Requested) { "$name : logs\STOP_QUEUE exists; not starting"; break }
  if (-not $DryRun -and -not (Enter-Lock $name)) { "$name : another queue run holds tomes\$name.LOCK; skipping"; continue }
  try {
  if (-not (Wait-Drive $src)) { "$name : $src never became ready; stopping the queue"; break }

  # The ledger: matches already inside a finished tome of this name ("dd/id"), and the highest chunk number in use.
  $builtIds = @{}; $topChunk = 0
  $global:LASTEXITCODE = $null
  $ledger = @(& $py -I -X utf8 "$base\scripts\data\tome_ledger.py" --name $name)
  if ($LASTEXITCODE -ne 0) { "$name : tome_ledger.py failed (exit $LASTEXITCODE); skipping this collection"; continue }
  foreach ($ln in $ledger) {
    if ($ln -match '^#chunks (\d+)') { $topChunk = [int]$Matches[1] }
    elseif ($ln.Trim()) { $builtIds[$ln.Trim()] = $true }
  }

  $items = @()
  $gaps = 0   # matches or days that exist (or should) but cannot be built yet: while any, the day is NOT complete and gets no CHUNKS_DONE
  foreach ($dd in $q.days) {
    $dayPath = "$src\$($q.yyyy)\$($q.mm)\$dd"
    if (-not (Test-Path -LiteralPath $dayPath)) { "$name day $dd missing on E:"; $gaps++; continue }
    foreach ($d in Get-ChildItem -LiteralPath $dayPath -Directory) {
      if ($builtIds.ContainsKey("$dd/$($d.Name)")) { continue }  # already inside a finished tome: not even looked at
      # a half-synced folder (fewer than the 43 objects every complete match has) would abort the whole build; it is picked up by a later run
      if ([System.IO.Directory]::GetFiles($d.FullName).Count -lt 43) { Add-Content "$base\logs\skipped_incomplete.txt" "$name $dd $($d.Name)"; $gaps++; continue }
      $items += [pscustomobject]@{ dd = $dd; id = $d.Name; path = $d.FullName; zip = $false }
    }
    # S3 also holds an <id>.zip (43 entries csds/<yyyy>/<mm>/<dd>/<id>/<channel>) next to some match folders (7 on day 12, all of them
    # duplicates of complete folders). A zip is used ONLY when its id has no complete folder and is not built yet; it is unpacked into the G:
    # staging area below, never on E:.
    $have = @{}
    foreach ($it0 in $items) { if ($it0.dd -eq $dd) { $have[$it0.id] = $true } }
    foreach ($zf in Get-ChildItem -LiteralPath $dayPath -File -Filter '*.zip') {
      $zid = [System.IO.Path]::GetFileNameWithoutExtension($zf.Name)
      if ($builtIds.ContainsKey("$dd/$zid")) { continue }
      if (-not $have.ContainsKey($zid)) { $items +=[pscustomobject]@{ dd = $dd; id = $zid; path = $zf.FullName; zip = $true } }
    }
  }
  if ($items.Count -eq 0) {
    if ($builtIds.Count -gt 0) {
      "$name : nothing new ($($builtIds.Count) matches already in finished tomes)$(if ($gaps) { "; $gaps match(es)/day(s) still incomplete on E:" })"
      if ($gaps -eq 0 -and -not $DryRun) { Write-ChunksDone $name }  # the last chunk of an earlier run may have finished without the marker being written
    } else { "$name : no complete match folders found; skipping" }
    continue
  }
  $items = @($items | Sort-Object id)
  $nChunks = [int][math]::Ceiling($items.Count / $ChunkSize)
  "$name : $($items.Count) new matches ($($builtIds.Count) already built) in $nChunks chunk(s) of up to $ChunkSize, numbered from c$('{0:D2}' -f ($topChunk + 1)), at $(Get-Date -Format s)"
  if ($DryRun) { continue }
  $allOk = $true
  for ($c = 0; $c -lt $nChunks; $c++) {
    if (Stop-Requested) { "$name : logs\STOP_QUEUE exists; stopping before chunk $($c + 1) of $nChunks"; $allOk = $false; break }
    $cname = "{0}_c{1:D2}" -f $name, ($topChunk + $c + 1)
    $ctome = "$base\tomes\$cname"
    $clog = "$base\logs\build_$($cname -replace '-','_').log"
    if (Test-Path "$ctome\BUILD_OK") { "$cname already built, skipping"; continue }
    $slice = @($items | Select-Object -Skip ($c * $ChunkSize) -First $ChunkSize)
    $ok = $false
    for ($attempt = 1; $attempt -le 3 -and -not $ok; $attempt++) {
      if (-not (Wait-Drive $src)) { break }
      $stage = "$base\staging_$($cname -replace '-','_')"
      # A staging folder reused from an earlier attempt (different slice or ChunkSize) may hold junctions of matches that are not in this
      # slice, and the builder would build them too (then the next chunk builds them again). Remove such stale JUNCTIONS (non-recursive
      # Directory.Delete: only the link goes, never the target); a stale real folder (an unpacked zip) stops the chunk instead.
      $keep = @{}; foreach ($m in $slice) { $keep["$($m.dd)\$($m.id)"] = $true }
      $staleReal = $null
      if (Test-Path -LiteralPath "$stage\csds") {
        foreach ($dayDir in Get-ChildItem -LiteralPath "$stage\csds\$($q.yyyy)\$($q.mm)" -Directory -ErrorAction SilentlyContinue) {
          foreach ($entry in Get-ChildItem -LiteralPath $dayDir.FullName -Directory -Force -ErrorAction SilentlyContinue) {
            if ($keep.ContainsKey("$($dayDir.Name)\$($entry.Name)")) { continue }
            if ($entry.Attributes -band [System.IO.FileAttributes]::ReparsePoint) { [System.IO.Directory]::Delete($entry.FullName) }
            else { $staleReal = $entry.FullName }
          }
        }
      }
      if ($staleReal) { "$cname : stale real folder in staging, not in this slice ($staleReal); left untouched, stopping this chunk"; break }
      foreach ($m in $slice) {
        $stageDay = "$stage\csds\$($q.yyyy)\$($q.mm)\$($m.dd)"
        New-Item -ItemType Directory -Force -Path $stageDay | Out-Null
        if ($m.zip) {
          $dest = Join-Path $stageDay $m.id
          if (-not (Test-Path -LiteralPath $dest) -or ([System.IO.Directory]::GetFiles($dest).Count -lt 43)) {
            if (Test-Path -LiteralPath $dest) { Remove-Item -LiteralPath $dest -Recurse -Force }  # a real folder under G: staging (never a junction)
            Add-Type -AssemblyName System.IO.Compression.FileSystem
            [System.IO.Compression.ZipFile]::ExtractToDirectory($m.path, $stage)
          }
        } else {
          Set-Junction (Join-Path $stageDay $m.id) $m.path
        }
      }
      if (Test-Path $ctome) { Remove-Item -LiteralPath $ctome -Recurse -Force }  # half-built leftovers (G: only; no junctions inside a tome folder)
      New-Item -ItemType Directory -Force -Path $ctome | Out-Null
      # every match of the slice must be present in staging (a junction that failed to create would silently drop its match)
      $present = 0
      foreach ($m in $slice) { if (Test-Path -LiteralPath (Join-Path "$stage\csds\$($q.yyyy)\$($q.mm)\$($m.dd)" $m.id)) { $present++ } }
      if ($present -ne $slice.Count) { "$cname attempt ${attempt}: only $present of $($slice.Count) matches staged; retrying"; Start-Sleep -Seconds 20; continue }
      $env:PURESKILLGG_TOME_DS_COLLECTION_PATH = $stage
      $env:PURESKILLGG_TOME_COLLECTION_PATH = $ctome
      $global:LASTEXITCODE = $null   # an old 0 must never be mistaken for this build's result
      uv run 'G:\ngclab_multi_spectral\build_basic_tomes.py' *> $clog
      $code = $LASTEXITCODE
      # BUILD_OK only when the builder reports exactly the slice's match count in its own final log line
      $built = -1
      if (Test-Path -LiteralPath $clog) { $hit = Select-String -Path $clog -Pattern 'Build Basic Tomes Done\s+matches=(\d+)' | Select-Object -Last 1; if ($hit) { $built = [int]$hit.Matches[0].Groups[1].Value } }
      if ($code -eq 0 -and $built -eq $slice.Count) {
        Set-Content -LiteralPath "$ctome\BUILD_OK" -Value "matches_staged=$($slice.Count) matches_built=$built finished=$(Get-Date -Format s)"
        $ok = Test-Path -LiteralPath "$ctome\BUILD_OK"   # the chunk counts as built only when its marker is really on disk
      }
      "$cname attempt $attempt exit $code, built $built of $($slice.Count) matches at $(Get-Date -Format s)"
      if (-not $ok) { Start-Sleep -Seconds 20 }
    }
    if (-not $ok) { $allOk = $false; "$cname FAILED after retries; the next run resumes from this chunk"; break }
  }
  if ($allOk -and $gaps -eq 0) { Write-ChunksDone $name }
  elseif ($allOk) { "$name : this run's chunks are built, but $gaps match(es)/day(s) are still incomplete on E:; no CHUNKS_DONE yet (a later run builds them)" }
  } finally { if (-not $DryRun) { Exit-Lock $name } }
}
"queue finished at $(Get-Date -Format s)"
