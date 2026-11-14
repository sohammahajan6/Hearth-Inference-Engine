# Fetch every third-party input into third_party/ (gitignored), verified
# against scripts/pins.psd1. Safe to re-run: finished items are skipped.
#
#   pwsh -File scripts\fetch-deps.ps1
#
# Downloads only. Building the third-party tools is scripts\setup.ps1's job.
[CmdletBinding()]
param()
. "$PSScriptRoot\lib\common.ps1"

$root = Get-RepoRoot
$pins = Get-Pins
$tp = Join-Path $root 'third_party'
New-Item -ItemType Directory -Force -Path $tp | Out-Null
Assert-FreeSpace -NeedGB 2

# 1. llama.cpp official release zips (CPU and Vulkan baselines).
$dl = Join-Path $tp 'downloads'
foreach ($z in $pins.LlamaCpp.Zips) {
    $zipPath = Join-Path $dl $z.Name
    Get-VerifiedFile -Url "$($pins.LlamaCpp.ReleaseBase)/$($z.Name)" -Destination $zipPath -Sha256 $z.Sha256
    $dest = Join-Path $tp $z.Dir
    $stamp = Join-Path $dest '.extracted-from'
    if ((Test-Path $stamp) -and ((Get-Content -Raw $stamp).Trim() -eq $z.Sha256)) {
        Write-Host "ok (cached)  $dest"
        continue
    }
    if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    Invoke-Native tar.exe @('-xf', $zipPath, '-C', $dest)
    Set-Content -NoNewline -Path $stamp -Value $z.Sha256
    Write-Host "ok           extracted $($z.Name) -> $dest"
}

# 2. llama.cpp source at the same build number (oracle tools, tensor dumper).
Sync-GitPin -Url $pins.LlamaCpp.Url -Tag $pins.LlamaCpp.Tag -Commit $pins.LlamaCpp.Commit `
    -Destination (Join-Path $tp 'llama.cpp')

# 3. Vulkan headers, loader shim, shader compiler, validation layers, test framework.
foreach ($g in $pins.Git) {
    Sync-GitPin -Url $g.Url -Tag $g.Tag -Commit $g.Commit -Destination (Join-Path $tp $g.Name)
}

# 4. Unicode Character Database files (the tokenizer's tables are generated from them).
$ucd = Join-Path $tp "unicode\$($pins.Unicode.Version)"
foreach ($f in $pins.Unicode.Files) {
    Get-VerifiedFile -Url "$($pins.Unicode.Base)/$($f.Name)" -Destination (Join-Path $ucd $f.Name) -Sha256 $f.Sha256
}

Write-Host 'fetch-deps: all pinned dependencies present and verified.'
