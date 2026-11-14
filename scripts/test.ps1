# Build and run the tests.
#
#   pwsh -File scripts\test.ps1                    # quick tier, MSVC Release
#   pwsh -File scripts\test.ps1 -Tier model        # tests that need the model files
#   pwsh -File scripts\test.ps1 -Tier full         # everything
#   pwsh -File scripts\test.ps1 -Preset asan       # same tests under ASan + UBSan
#   pwsh -File scripts\test.ps1 -Filter "*gguf*"   # only matching test cases
#   pwsh -File scripts\test.ps1 -Details          # also the tests' notes and the expected messages
#
# GPU tests run under the Khronos validation layer (built by setup.ps1) and
# fail if it reports anything. The output is only failures and the summary:
# tests that provoke a warning or error on purpose catch it and check its
# text, and any other warning or error fails the run (tests/test_main.cpp).
# -Details also prints the tests' MESSAGE notes (measurements, skips) and the
# caught lines, marked "(expected)".
[CmdletBinding()]
param(
    [ValidateSet('quick', 'model', 'full')][string]$Tier = 'quick',
    [ValidateSet('release', 'debug', 'asan')][string]$Preset = 'release',
    [string]$Filter = '',
    [switch]$NoBuild,
    [switch]$Details,
    [int]$TimeoutSec = 0
)
. "$PSScriptRoot\lib\common.ps1"
. "$PSScriptRoot\lib\toolchain.ps1"

$root = Get-RepoRoot
Assert-GpuBudget 'the tests (the GPU tests run in every tier)'
if (-not $NoBuild) {
    & "$PSScriptRoot\build.ps1" -Preset $Preset -Target @('hearth_tests', 'hearth')   # throws on failure
}
$exe = Join-Path $root "build\$Preset\tests\hearth_tests.exe"
if (-not (Test-Path $exe)) { throw "test executable not found: $exe" }

$env:VK_LOADER_LAYERS_DISABLE = '~implicit~'
$vvl = Join-Path $root 'third_party\install\vvl\bin'
if (Test-Path (Join-Path $vvl 'VkLayer_khronos_validation.json')) {
    $env:VK_ADD_LAYER_PATH = $vvl
    $env:HEARTH_TEST_VALIDATION = '1'
} else {
    Write-Warning 'Vulkan validation layer not built yet (scripts\setup.ps1 -Only vvl); GPU tests run without it.'
}
if ($Preset -eq 'asan') { Enter-ClangEnvironment }   # puts libc++/ASan runtime DLLs on PATH

$suites = switch ($Tier) {
    'quick' { 'quick' }
    'model' { 'model' }
    'full' { '*' }
}
$limits = @{ quick = 300; model = 1800; full = 7200 }
if ($TimeoutSec -le 0) { $TimeoutSec = $limits[$Tier] }
if ($Preset -eq 'asan') { $TimeoutSec *= 4 }

$testArgs = @("--test-suite=$suites", '--order-by=file', '--no-intro=true')
if ($Details) {
    $env:HEARTH_TEST_SHOW_EXPECTED = '1'
} else {
    $testArgs += '--reporters=no-notes'
    $env:HEARTH_TEST_SHOW_EXPECTED = '0'
}
if ($Filter) { $testArgs += "--test-case=$Filter" }
$t0 = Get-Date
$code = Invoke-NativeTimeout -FilePath $exe -ArgumentList $testArgs -TimeoutSec $TimeoutSec
$secs = ((Get-Date) - $t0).TotalSeconds
if ($code -ne 0) { throw ("tests failed: exit code {0} after {1:N1} s" -f (Format-ExitCode $code), $secs) }
Write-Host ("test: tier '{0}' ({1}) passed in {2:N1} s" -f $Tier, $Preset, $secs)
