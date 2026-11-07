# Download the 2026-10-25 and 2026-10-26 CSDS partitions from S3 to E:.
# Copy only: this script never uses --delete and does not start a tome build.
# Run only when E: is mounted and no other aws.exe job is using the drive:
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_days25_26_download.ps1

[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

$base = 'G:\pureskill\csds-cs2'
$bucket = 's3://pureskillgg-csgo-production-dataexchange-csds-0'
$destRoot = 'E:\csds_final'
$days = @('25', '26')
$logPath = Join-Path $base 'logs\days_25_26_download.log'
$maxAttempts = 4
$pollSeconds = 180
$maxHoursPerAttempt = 8
$safetyReserve = 40GB

if (-not (Test-Path -LiteralPath $base)) {
    throw "Project folder not found: $base"
}
New-Item -ItemType Directory -Path (Split-Path -Parent $logPath) -Force | Out-Null

function Write-Log([string]$Message) {
    "$(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss') $Message" |
        Add-Content -LiteralPath $logPath -Encoding UTF8
}

function Assert-AwsIdle {
    $active = @(Get-Process -Name aws -ErrorAction SilentlyContinue)
    if ($active.Count -gt 0) {
        throw "aws.exe is already running. Wait for that job to finish before using E:; no new sync was started."
    }
}

function Get-S3Objects([string]$Day) {
    $prefix = "csds/2026/10/$Day/"
    $listing = & aws s3 ls "$bucket/$prefix" --recursive --profile psgg --no-cli-pager 2>&1
    $awsExitCode = $LASTEXITCODE
    if ($awsExitCode -ne 0) {
        $detail = ($listing | ForEach-Object { [string]$_ }) -join ' | '
        throw "S3 listing failed for 2026-10-$Day (aws exit $awsExitCode): $detail"
    }

    $objects = @(
        foreach ($line in $listing) {
            $text = [string]$line
            if ($text -match '^\s*\S+\s+\S+\s+(\d+)\s+(.+?)\s*$') {
                $key = $matches[2].Trim()
                if (-not $key.StartsWith($prefix, [System.StringComparison]::Ordinal)) {
                    $key = "$prefix$key"
                }
                [pscustomobject]@{ Key = $key; Size = [int64]$matches[1] }
            }
        }
    )
    if ($objects.Count -eq 0) {
        throw "S3 returned no objects for 2026-10-$Day. Nothing was downloaded."
    }
    return $objects
}

function Get-LocalObjectSizes([string]$Day) {
    $dayRoot = Join-Path $destRoot "csds\2026\10\$Day"
    $rootPrefix = [System.IO.Path]::GetFullPath($destRoot).TrimEnd('\') + '\'
    $sizes = [System.Collections.Generic.Dictionary[string,long]]::new([System.StringComparer]::Ordinal)

    if (Test-Path -LiteralPath $dayRoot) {
        foreach ($file in Get-ChildItem -LiteralPath $dayRoot -File -Recurse -Force) {
            $key = $file.FullName.Substring($rootPrefix.Length).Replace('\', '/')
            $sizes[$key] = [int64]$file.Length
        }
    }
    return ,$sizes
}

function Get-PendingObjects([object[]]$S3Objects, $LocalSizes) {
    $pending = [System.Collections.Generic.List[object]]::new()
    foreach ($object in $S3Objects) {
        if (-not $LocalSizes.ContainsKey($object.Key) -or
            $LocalSizes[$object.Key] -ne [int64]$object.Size) {
            [void]$pending.Add($object)
        }
    }
    return $pending.ToArray()
}

function Get-TotalBytes([object[]]$Objects) {
    if ($Objects.Count -eq 0) { return [int64]0 }
    return [int64](($Objects | Measure-Object -Property Size -Sum).Sum)
}

$drive = Get-PSDrive -Name E -ErrorAction SilentlyContinue
if ($null -eq $drive -or -not (Test-Path -LiteralPath $destRoot)) {
    throw "E:\csds_final is not available. Mount E: before running this script."
}
Assert-AwsIdle

# Preflight both dates so insufficient space cannot leave only the first day downloaded.
$plan = @{}
$totalPendingBytes = [int64]0
foreach ($day in $days) {
    $objects = @(Get-S3Objects -Day $day)
    $localSizes = Get-LocalObjectSizes -Day $day
    $pending = @(Get-PendingObjects -S3Objects $objects -LocalSizes $localSizes)
    $pendingBytes = Get-TotalBytes -Objects $pending
    $plan[$day] = [pscustomobject]@{ Objects = $objects; PendingBytes = $pendingBytes }
    $totalPendingBytes += $pendingBytes
    Write-Log "preflight 2026-10-${day}: $($objects.Count) S3 objects; $($pending.Count) missing or wrong-sized; $([math]::Round($pendingBytes / 1GB, 2)) GB to fetch"
}

$drive = Get-PSDrive -Name E
$requiredBytes = $totalPendingBytes + $safetyReserve
if ($totalPendingBytes -gt 0 -and $drive.Free -lt $requiredBytes) {
    throw ("Not enough free space on E:. Need {0:N1} GB including the 40 GB reserve; have {1:N1} GB. No sync was started." -f
        ($requiredBytes / 1GB), ($drive.Free / 1GB))
}
Write-Log ("preflight passed: {0:N1} GB free; {1:N1} GB estimated download plus 40 GB reserve" -f
    ($drive.Free / 1GB), ($totalPendingBytes / 1GB))

foreach ($day in $days) {
    $prefix = "csds/2026/10/$day/"
    $destination = Join-Path $destRoot "csds\2026\10\$day"
    $objects = @($plan[$day].Objects)
    $verified = $false

    for ($attempt = 1; $attempt -le $maxAttempts -and -not $verified; $attempt++) {
        Assert-AwsIdle
        $localSizes = Get-LocalObjectSizes -Day $day
        $pending = @(Get-PendingObjects -S3Objects $objects -LocalSizes $localSizes)
        if ($pending.Count -eq 0) {
            $verified = $true
            Write-Log "2026-10-$day already complete; no sync needed"
            break
        }

        $drive = Get-PSDrive -Name E
        $pendingBytes = Get-TotalBytes -Objects $pending
        if ($drive.Free -lt ($pendingBytes + $safetyReserve)) {
            throw ("Not enough free space before 2026-10-${day}: need {0:N1} GB including reserve; have {1:N1} GB. No sync started for this day." -f
                (($pendingBytes + $safetyReserve) / 1GB), ($drive.Free / 1GB))
        }

        $syncLog = Join-Path $base "logs\aws_sync_2026-10-$day.log"
        $syncErr = Join-Path $base "logs\aws_sync_2026-10-$day.err"
        $syncArgs = @(
            's3', 'sync', "$bucket/$prefix", $destination,
            '--profile', 'psgg', '--no-cli-pager', '--only-show-errors'
        )
        $process = Start-Process -FilePath 'aws' -ArgumentList $syncArgs -RedirectStandardOutput $syncLog -RedirectStandardError $syncErr -PassThru -WindowStyle Hidden
        Write-Log "2026-10-$day sync attempt $attempt started (pid $($process.Id))"

        $started = Get-Date
        while ($true) {
            $localSizes = Get-LocalObjectSizes -Day $day
            $pending = @(Get-PendingObjects -S3Objects $objects -LocalSizes $localSizes)
            $running = [bool](Get-Process -Id $process.Id -ErrorAction SilentlyContinue)
            Write-Log "2026-10-${day} attempt ${attempt}: $($pending.Count) of $($objects.Count) objects still missing or wrong-sized; aws alive=$running"

            if ($pending.Count -eq 0) {
                if ($running) {
                    Write-Log "2026-10-$day objects verified; stopping lingering aws.exe"
                    Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
                }
                $verified = $true
                break
            }
            if (-not $running) { break }
            if (((Get-Date) - $started).TotalHours -gt $maxHoursPerAttempt) {
                Write-Log "2026-10-$day attempt $attempt timed out; stopping aws.exe"
                Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
                break
            }
            Start-Sleep -Seconds $pollSeconds
        }

        if (-not $verified) {
            $localSizes = Get-LocalObjectSizes -Day $day
            $pending = @(Get-PendingObjects -S3Objects $objects -LocalSizes $localSizes)
            Write-Log "2026-10-$day attempt $attempt ended; $($pending.Count) objects remain"
            $verified = ($pending.Count -eq 0)
        }
    }

    if (-not $verified) {
        throw "2026-10-$day did not verify after $maxAttempts attempts. Stopping before the next date."
    }
    Write-Log "2026-10-$day verified against the S3 object list and sizes"
}

Write-Log 'both requested dates are present and verified; no tome build was started'

