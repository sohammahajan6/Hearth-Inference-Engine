# Format all first-party C++ with clang-format (from llvm-mingw, read-only use).
#
#   pwsh -File scripts\format.ps1          # rewrite files in place
#   pwsh -File scripts\format.ps1 -Check   # fail if any file is not formatted
#
# GLSL shaders are not formatted: clang-format mis-parses GLSL block syntax.
[CmdletBinding()]
param([switch]$Check)
. "$PSScriptRoot\lib\common.ps1"
. "$PSScriptRoot\lib\toolchain.ps1"

$root = Get-RepoRoot
$clangFormat = Join-Path (Get-LlvmMingwBin) 'clang-format.exe'
$files = & git -C $root ls-files --cached --others --exclude-standard -- '*.cpp' '*.h' |
    Where-Object { $_ -notmatch '^third_party/' } |
    ForEach-Object { Join-Path $root $_ }
if (-not $files) { Write-Host 'format: no files'; return }
if ($Check) {
    & $clangFormat --dry-run --Werror @files
    if ($LASTEXITCODE -ne 0) { throw 'format: files above need formatting. Run: pwsh -File scripts\format.ps1' }
    Write-Host "format: $(@($files).Count) files clean"
} else {
    Invoke-Native $clangFormat (@('-i') + $files)
    Write-Host "format: $(@($files).Count) files formatted"
}
