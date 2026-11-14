# Shared helpers for every script in scripts/. Dot-source it:
#   . "$PSScriptRoot\lib\common.ps1"
# Rules these helpers enforce:
#   * every native command's exit code is checked (PowerShell does not do this);
#   * every download is verified against a pinned sha256 before it is used;
#   * every git checkout is verified against a pinned commit.

Set-StrictMode -Version 3.0
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # Invoke-WebRequest's progress bar is very slow

$script:RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path

function Get-RepoRoot { return $script:RepoRoot }

function Get-Pins {
    return Import-PowerShellDataFile (Join-Path $script:RepoRoot 'scripts\pins.psd1')
}

function Format-ExitCode([int]$Code) {
    # Windows crash codes are easier to recognise in hex.
    $hex = '0x{0:X8}' -f $Code
    $known = @{
        '0xC0000135' = 'DLL not found'
        '0xC000001D' = 'illegal instruction (CPU lacks an instruction the binary uses)'
        '0xC0000005' = 'access violation'
        '0xC0000409' = 'stack buffer overrun / fast-fail (often an abort or failed check)'
        '0xC00000FD' = 'stack overflow'
    }
    if ($known.ContainsKey($hex)) { return "$Code ($hex, $($known[$hex]))" }
    return "$Code ($hex)"
}

function Invoke-Native {
    # Run a native program and throw if it fails. Output streams through.
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [string[]]$ArgumentList = @()
    )
    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "'$FilePath $($ArgumentList -join ' ')' failed with exit code $(Format-ExitCode $LASTEXITCODE)"
    }
}

function Invoke-NativeTimeout {
    # Run a native program with stdin closed and a hard time limit. Output goes
    # straight to this console. Returns the exit code; throws on timeout (after
    # killing the process tree) so a hung program can never stall a session.
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [string[]]$ArgumentList = @(),
        [Parameter(Mandatory)][int]$TimeoutSec
    )
    $psi = [System.Diagnostics.ProcessStartInfo]::new($FilePath)
    foreach ($a in $ArgumentList) { $psi.ArgumentList.Add($a) }
    $psi.UseShellExecute = $false
    $psi.RedirectStandardInput = $true   # then closed immediately: the program sees EOF, never waits for input
    $p = [System.Diagnostics.Process]::Start($psi)
    $p.StandardInput.Close()
    if (-not $p.WaitForExit($TimeoutSec * 1000)) {
        $p.Kill($true)
        throw "'$FilePath' did not finish within $TimeoutSec s and was killed"
    }
    $p.WaitForExit()   # flush asynchronous output handlers
    return $p.ExitCode
}

function Invoke-NativeCapture {
    # Like Invoke-NativeTimeout, but captures stdout and stderr (as text) instead
    # of streaming them. For tools whose output is parsed or archived.
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [string[]]$ArgumentList = @(),
        [Parameter(Mandatory)][int]$TimeoutSec,
        [hashtable]$Environment = @{},
        [string]$WorkingDirectory = ''  # the process's current directory (Push-Location does not set it)
    )
    $psi = [System.Diagnostics.ProcessStartInfo]::new($FilePath)
    foreach ($a in $ArgumentList) { $psi.ArgumentList.Add($a) }
    foreach ($k in $Environment.Keys) { $psi.Environment[$k] = [string]$Environment[$k] }
    if ($WorkingDirectory) { $psi.WorkingDirectory = $WorkingDirectory }
    $psi.UseShellExecute = $false
    $psi.RedirectStandardInput = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.StandardOutputEncoding = [System.Text.Encoding]::UTF8
    $psi.StandardErrorEncoding = [System.Text.Encoding]::UTF8
    $p = [System.Diagnostics.Process]::Start($psi)
    $p.StandardInput.Close()
    $out = $p.StandardOutput.ReadToEndAsync()
    $err = $p.StandardError.ReadToEndAsync()
    if (-not $p.WaitForExit($TimeoutSec * 1000)) {
        $p.Kill($true)
        throw "'$FilePath' did not finish within $TimeoutSec s and was killed"
    }
    $p.WaitForExit()
    return [pscustomobject]@{ ExitCode = $p.ExitCode; Stdout = $out.Result; Stderr = $err.Result }
}

function Get-Sha256([string]$Path) {
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

function Get-VerifiedFile {
    # Download $Url to $Destination unless an intact copy is already there.
    # Downloads go to "<dest>.part" first and are renamed only after the
    # sha256 matches, so an interrupted or corrupted download is never used.
    param(
        [Parameter(Mandatory)][string]$Url,
        [Parameter(Mandatory)][string]$Destination,
        [Parameter(Mandatory)][string]$Sha256
    )
    if (Test-Path -LiteralPath $Destination) {
        if ((Get-Sha256 $Destination) -eq $Sha256) {
            Write-Host "ok (cached)  $Destination"
            return
        }
        Write-Warning "sha256 mismatch for existing $Destination; downloading again"
        Remove-Item -LiteralPath $Destination
    }
    $dir = Split-Path -Parent $Destination
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $part = "$Destination.part"
    Write-Host "download     $Url"
    # -f: fail on HTTP errors; -L: follow redirects; -C -: resume a .part file.
    & curl.exe -fL --retry 5 --retry-delay 10 --retry-all-errors -C - -sS -o $part $Url
    if ($LASTEXITCODE -ne 0) { throw "download failed (curl exit $LASTEXITCODE): $Url" }
    $actual = Get-Sha256 $part
    if ($actual -ne $Sha256) {
        Remove-Item -LiteralPath $part
        throw "sha256 mismatch for $Url`n  expected $Sha256`n  actual   $actual"
    }
    Move-Item -Force -LiteralPath $part -Destination $Destination
    Write-Host "ok           $Destination"
}

function Sync-GitPin {
    # Shallow-clone $Url at $Tag into $Destination and verify the commit.
    # autocrlf is forced off so third-party trees are byte-identical to upstream.
    param(
        [Parameter(Mandatory)][string]$Url,
        [Parameter(Mandatory)][string]$Tag,
        [Parameter(Mandatory)][string]$Commit,
        [Parameter(Mandatory)][string]$Destination
    )
    if (Test-Path -LiteralPath (Join-Path $Destination '.git')) {
        $head = (& git -C $Destination rev-parse HEAD).Trim()
        if ($head -eq $Commit) {
            Write-Host "ok (cached)  $Destination @ $Tag"
            return
        }
        throw "$Destination is at $head but the pin is $Commit ($Tag). Delete the folder and re-run."
    }
    Write-Host "clone        $Url @ $Tag"
    Invoke-Native git @('-c', 'core.autocrlf=false', '-c', 'advice.detachedHead=false',
        'clone', '--quiet', '--depth', '1', '--branch', $Tag, $Url, $Destination)
    & git -C $Destination config core.autocrlf false
    $head = (& git -C $Destination rev-parse HEAD).Trim()
    if ($head -ne $Commit) {
        throw "$Url tag $Tag resolved to $head, expected $Commit. The tag may have moved upstream; investigate before changing the pin."
    }
    Write-Host "ok           $Destination @ $Tag ($Commit)"
}

# Whether a model file's matrices are K-quants (Q4_K_M and the like): its
# quantized gates use the bars of docs/adr/0022-k-quant-gates.md instead of
# ADR 0007's. The derived files are named by scripts/fetch-models.ps1.
function Test-KQuantModel([string]$Path) {
    return [System.IO.Path]::GetFileName($Path) -cmatch '-Q[2-6]_K(_[SML])?(-pure)?\.gguf$'
}

# Whether a model file is Qwen2.5 (named by scripts/fetch-models.ps1): its
# quantized gates use the bars of docs/adr/0023-qwen-gates.md. K-quant files
# take ADR 0022's (Test-KQuantModel first); no Qwen K-quant file has been measured.
function Test-QwenModel([string]$Path) {
    return [System.IO.Path]::GetFileName($Path) -cmatch '^Qwen2\.5-'
}

function Assert-FreeSpace {
    # Refuse to start a large write if it would leave less than the reserve free.
    param([string]$Drive = 'E', [double]$NeedGB = 0, [double]$ReserveGB = 8)
    $free = (Get-PSDrive -Name $Drive).Free / 1GB
    if ($free - $NeedGB -lt $ReserveGB) {
        throw ("Not enough disk space on {0}: {1:N1} GB free, need {2:N1} GB plus a {3:N1} GB reserve." -f $Drive, $free, $NeedGB, $ReserveGB)
    }
}

function Assert-GpuBudget {
    # The GPU-hang (TDR) budget across processes (docs/architecture/08-gpu-backend.md
    # section 4): hearth.exe and hearth_tests.exe append every real device loss,
    # GPU timeout and near-miss watchdog trip to build\<preset>\gpu-hang-ledger.txt.
    # GPU work is refused while any entry is younger than 60 s, or 2 entries fall
    # in the last 12 hours. Review the entries, then archive the file to continue.
    param([string]$What = 'GPU work')
    $now = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    $entries = @(Get-ChildItem -Path (Join-Path $script:RepoRoot 'build') -Filter 'gpu-hang-ledger.txt' -Recurse -Depth 1 -ErrorAction SilentlyContinue |
        ForEach-Object { $f = $_.FullName; Get-Content -LiteralPath $f | Where-Object { $_ } | ForEach-Object {
                $p = $_ -split "`t"
                if ($p.Count -lt 3 -or $p[1] -notmatch '^[0-9]+$') { throw "malformed line in ${f}: $_" }
                [pscustomobject]@{ File = $f; Seconds = [long]$p[1]; Line = $_ } } })
    $recent = @($entries | Where-Object { $now - $_.Seconds -lt 12 * 3600 })
    $fresh = @($entries | Where-Object { $now - $_.Seconds -lt 60 })
    if ($fresh.Count -gt 0 -or $recent.Count -ge 2) {
        $recent | ForEach-Object { Write-Host "  $($_.File): $($_.Line)" }
        throw ("${What} refused by the GPU hang budget: {0} ledger entries in the last 12 h, {1} in the last 60 s " +
            "(limit: fewer than 2, and none). Find the cause first; then move the ledger file(s) aside.") -f $recent.Count, $fresh.Count
    }
}

function Get-GpuProof {
    # Proof of execution for a GPU backend (docs/architecture/08-gpu-backend.md
    # section 4): the log lines in which hearth names the GPU device and counts
    # its submissions or dispatches. Throws if the gpu backend logged no
    # dispatch count, or only zero, so a run that never touched the GPU cannot
    # pass a GPU gate. Returns @() for the CPU backends.
    param([Parameter(Mandatory)][string]$Log, [Parameter(Mandatory)][string]$Backend)
    if ($Backend -ne 'gpu') { return @() }
    $lines = @($Log -split "`r?`n" | Where-Object { $_ -match '\bgpu: ' } | ForEach-Object { $_.Trim() })
    $counts = @($lines | ForEach-Object { if ($_ -match '(\d+) (GPU )?dispatches') { [long]$Matches[1] } })
    if ($counts.Count -eq 0) {
        throw "hearth --backend $Backend logged no GPU dispatch count; the run cannot count as GPU work"
    }
    if (($counts | Measure-Object -Maximum).Maximum -le 0) {
        throw "hearth --backend $Backend ran 0 GPU dispatches; the run cannot count as GPU work"
    }
    # A run that lost its GPU and continued on the CPU (08 doc section 4) is not GPU work either.
    if ($Log -match 'the GPU failed') { throw "hearth --backend $Backend lost its GPU during the run; it cannot count as GPU work" }
    return $lines
}
