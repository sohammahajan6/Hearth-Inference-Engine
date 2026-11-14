# The release zip: builds
# Release, runs the quick tests, checks that hearth.exe imports only Windows
# DLLs, zips the exe with its documents, then unpacks the zip into a fresh
# folder and tests it there. Fails at the first check that does not pass.
#
#   pwsh -File scripts\package.ps1 [-Model models\X.gguf] [-AllowDirty]
#
# Output: build\package\hearth-<version>-win-x64[-dirty].zip and
# build\package\package-report.txt. The test folder is build\package-test\.
[CmdletBinding()]
param(
    [string]$Model = 'models\Llama-3.2-1B-Instruct-Q8_0-pure.gguf',
    [switch]$AllowDirty
)
. "$PSScriptRoot\lib\common.ps1"
# Reports use one number format whatever the PC's region settings (this PC groups digits as 5,90,062).
[System.Threading.Thread]::CurrentThread.CurrentCulture = [cultureinfo]::InvariantCulture
. "$PSScriptRoot\lib\toolchain.ps1"

$root = Get-RepoRoot
$commit = (& git -C $root describe --always --dirty).Trim()
$dirty = $commit.EndsWith('-dirty')
if ($dirty -and -not $AllowDirty) { throw "package: the tree has uncommitted changes ($commit). Commit first, or pass -AllowDirty for a trial." }
$modelPath = Join-Path $root $Model
if (-not (Test-Path $modelPath)) { throw "package: missing $modelPath (the fresh-folder test runs it)" }

& "$PSScriptRoot\build.ps1" -Preset release
& "$PSScriptRoot\test.ps1" -Tier quick -Preset release -NoBuild
$exe = Join-Path $root 'build\release\cli\hearth.exe'
$report = [System.Collections.Generic.List[string]]::new()
function Check([string]$What, [bool]$Ok, [string]$Detail = '') {
    $line = ('{0} {1}{2}' -f ($(if ($Ok) { 'ok  ' } else { 'FAIL' })), $What, $(if ($Detail) { ": $Detail" } else { '' }))
    $report.Add($line)
    Write-Host "package: $line"
    if (-not $Ok) {
        [IO.File]::WriteAllText($script:reportFile, (($script:header + $report) -join "`n") + "`n")
        throw "package: check failed: $What"
    }
}

$version = ([regex]::Match((& $exe --version | Select-Object -First 1), '^hearth (\S+)$')).Groups[1].Value
if (-not $version) { throw 'package: hearth --version printed no version' }
$name = "hearth-$version-win-x64$(if ($dirty) { '-dirty' })"
$pkgDir = Join-Path $root 'build\package'
$stage = Join-Path $pkgDir $name
$zip = Join-Path $pkgDir "$name.zip"
$script:reportFile = Join-Path $pkgDir 'package-report.txt'
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
if (Test-Path $zip) { Remove-Item -Force $zip }
New-Item -ItemType Directory -Force -Path $stage | Out-Null
$script:header = @(
    "package: $name"
    "date:    $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
    "commit:  $commit"
    "command: pwsh -File scripts\package.ps1 -Model $Model$(if ($AllowDirty) { ' -AllowDirty' })"
    ''
)

# 3. Only Windows DLLs. A clean Windows PC has System32 and nothing of Visual Studio.
Enter-MsvcEnvironment
$dumpbin = (Get-Command dumpbin.exe -ErrorAction Stop).Source
$deps = @(& $dumpbin /nologo /dependents $exe | ForEach-Object { $_.Trim() } | Where-Object { $_ -match '\.dll$' })
if ($deps.Count -eq 0) { throw 'package: dumpbin listed no dependencies' }
$sys = Join-Path $env:SystemRoot 'System32'
$bad = @($deps | Where-Object {
        $_ -match '^(msvcp|vcruntime|concrt|vcomp)' -or -not ($_ -match '^api-ms-win-' -or (Test-Path (Join-Path $sys $_)))
    })
Check 'imports only Windows DLLs' ($bad.Count -eq 0) ($(if ($bad.Count) { "not part of Windows: $($bad -join ', ')" } else { ($deps -join ', ') }))

# 4. Stage and zip.
Copy-Item $exe $stage
foreach ($f in @('LICENSE', 'README.md', 'THIRD_PARTY_NOTICES.md', 'models\MODELS.md')) { Copy-Item (Join-Path $root $f) $stage }
$versionText = ((& $exe --version) -join "`n").Trim()
$exeSha = Get-Sha256 (Join-Path $stage 'hearth.exe')
[IO.File]::WriteAllText((Join-Path $stage 'BUILDINFO.txt'), "$versionText`nsha256 hearth.exe: $exeSha`n")
$files = @(Get-ChildItem $stage -File | Where-Object { $_.Name -ne 'SHA256SUMS.txt' } | Sort-Object Name)
[IO.File]::WriteAllText((Join-Path $stage 'SHA256SUMS.txt'), (($files | ForEach-Object { "$(Get-Sha256 $_.FullName)  $($_.Name)" }) -join "`n") + "`n")
Compress-Archive -Path $stage -DestinationPath $zip
$zipSha = Get-Sha256 $zip
$report.Add(("zip:  {0} ({1:N0} bytes) sha256 {2}" -f (Split-Path -Leaf $zip), (Get-Item $zip).Length, $zipSha))
$report.Add(("exe:  {0:N0} bytes, sha256 {1}" -f (Get-Item $exe).Length, $exeSha))

# 5. The fresh-folder test: only what the zip holds.
$test = Join-Path $root 'build\package-test'
if (Test-Path $test) { Remove-Item -Recurse -Force $test }
Expand-Archive -Path $zip -DestinationPath $test
$dir = Join-Path $test $name
$sums = Get-Content (Join-Path $dir 'SHA256SUMS.txt') | Where-Object { $_ }
$mism = @($sums | Where-Object { $h, $f = $_ -split '  ', 2; (Get-Sha256 (Join-Path $dir $f)) -ne $h })
Check 'unpacked files match SHA256SUMS.txt' ($mism.Count -eq 0 -and $sums.Count -eq $files.Count) "$($sums.Count) files"
$pexe = Join-Path $dir 'hearth.exe'
Check 'unpacked hearth.exe matches BUILDINFO.txt' ((Get-Sha256 $pexe) -eq $exeSha)
# Every unpacked run has the unpacked folder as its current directory.
$v = Invoke-NativeCapture -FilePath $pexe -ArgumentList @('--version') -TimeoutSec 60 -WorkingDirectory $dir
$vText = ($v.Stdout -replace "`r`n", "`n").Trim()  # a pipe gets CRLF line ends from the C runtime's text mode
Check 'hearth --version' ($v.ExitCode -eq 0 -and $vText -eq ($versionText -replace "`r`n", "`n")) (($vText -split "`n")[0..1] -join '; ')
$i = Invoke-NativeCapture -FilePath $pexe -ArgumentList @('info') -TimeoutSec 120 -WorkingDirectory $dir `
    -Environment @{ VK_LOADER_LAYERS_DISABLE = '~implicit~' }
Check 'hearth info' ($i.ExitCode -eq 0) "exit $($i.ExitCode)"
$runArgs = @('run', '--model', $modelPath, '--prompt', 'The capital of France is', '--n-predict', '16', '--temp', '0', '--ids')
$mine = Invoke-NativeCapture -FilePath $pexe -ArgumentList $runArgs -TimeoutSec 600 -WorkingDirectory $dir
$ref = Invoke-NativeCapture -FilePath $exe -ArgumentList $runArgs -TimeoutSec 600
$ids = { param($r) [regex]::Match($r.Stdout, '\[[0-9, ]+\]').Value }
$a = & $ids $mine
$b = & $ids $ref
Check 'hearth run: same token IDs as the build tree' ($mine.ExitCode -eq 0 -and $a -and $a -eq $b) $a
[IO.File]::WriteAllText($script:reportFile, (($script:header + $report) -join "`n") + "`n")
Write-Host "package: $zip"
Write-Host "package: report in $script:reportFile"
