# Build hearth. This is the only supported way to build: it selects
# the pinned toolchain so results never depend on what happens to be on PATH.
#
#   pwsh -File scripts\build.ps1                  # MSVC Release (product + benchmarks)
#   pwsh -File scripts\build.ps1 -Preset debug    # MSVC Debug (debug CRT heap checks)
#   pwsh -File scripts\build.ps1 -Preset asan     # clang + AddressSanitizer + UBSan
#   pwsh -File scripts\build.ps1 -Preset analyze  # MSVC /analyze static analysis
#   pwsh -File scripts\build.ps1 -Target hearth_tests -Clean
[CmdletBinding()]
param(
    [ValidateSet('release', 'debug', 'asan', 'analyze')][string]$Preset = 'release',
    [string[]]$Target = @(),
    [switch]$Clean,
    [int]$Jobs = 0
)
. "$PSScriptRoot\lib\common.ps1"
. "$PSScriptRoot\lib\toolchain.ps1"

$root = Get-RepoRoot
$bdir = Join-Path $root "build\$Preset"
if ($Clean -and (Test-Path $bdir)) { Remove-Item -Recurse -Force $bdir }

$configure = @('-G', 'Ninja', '-S', $root, '-B', $bdir, "-DCMAKE_MAKE_PROGRAM=$(Get-NinjaExe)")
switch ($Preset) {
    'release' {
        Enter-MsvcEnvironment
        $configure += @('-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_C_COMPILER=cl', '-DCMAKE_CXX_COMPILER=cl')
    }
    'debug' {
        Enter-MsvcEnvironment
        $configure += @('-DCMAKE_BUILD_TYPE=Debug', '-DCMAKE_C_COMPILER=cl', '-DCMAKE_CXX_COMPILER=cl')
    }
    'analyze' {
        Enter-MsvcEnvironment
        $configure += @('-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_C_COMPILER=cl', '-DCMAKE_CXX_COMPILER=cl',
            '-DHEARTH_ANALYZE=ON')
    }
    'asan' {
        Enter-ClangEnvironment
        # Forward slashes: CMake writes these paths into generated .cmake files,
        # where a backslash starts an escape sequence.
        $bin = (Get-LlvmMingwBin) -replace '\\', '/'
        $configure += @('-DCMAKE_BUILD_TYPE=RelWithDebInfo',
            "-DCMAKE_C_COMPILER=$bin/clang.exe", "-DCMAKE_CXX_COMPILER=$bin/clang++.exe",
            "-DCMAKE_RC_COMPILER=$bin/x86_64-w64-mingw32-windres.exe", '-DHEARTH_SANITIZE=ON')
    }
}

Invoke-Native cmake $configure
$build = @('--build', $bdir)
if ($Target.Count -gt 0) { $build += '--target'; $build += $Target }
if ($Jobs -gt 0) { $build += @('--', "-j$Jobs") }
Invoke-Native cmake $build
Write-Host "build: $Preset ok -> $bdir"
