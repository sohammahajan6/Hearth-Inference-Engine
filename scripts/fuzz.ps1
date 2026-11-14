# Run a fuzzer from the clang ASan/UBSan build with libFuzzer
# (docs/adr/0006-libfuzzer.md; docs/ACCEPTANCE.md: >= 10 minutes per fuzzer
# with no findings).
#
#   pwsh -File scripts\fuzz.ps1 -Target gguf|tokenizer|http -Seconds 600
#
# The tokenizer target loads the real vocabulary from
# models\Llama-3.2-1B-Instruct-Q8_0-pure.gguf (or $env:HEARTH_FUZZ_MODEL).
#
# The growing corpus lives in build\asan\fuzz-corpus\<target> and is reused by
# later runs; seeds come from tests\fuzz\seeds\<target>. A crash writes its
# input to build\asan\fuzz-artifacts\<target>\: minimise it, then add it to
# tests\fuzz\regressions\<target>\ so every test run replays it.
[CmdletBinding()]
param(
    [ValidateSet('gguf', 'tokenizer', 'http')][string]$Target = 'gguf',
    [int]$Seconds = 600,
    [long]$Seed = 1
)
. "$PSScriptRoot\lib\common.ps1"
. "$PSScriptRoot\lib\toolchain.ps1"

$root = Get-RepoRoot
& "$PSScriptRoot\build.ps1" -Preset asan -Target "hearth_fuzz_$Target"
Enter-ClangEnvironment
$exe = Join-Path $root "build\asan\tests\hearth_fuzz_$Target.exe"
$corpus = Join-Path $root "build\asan\fuzz-corpus\$Target"
$artifacts = Join-Path $root "build\asan\fuzz-artifacts\$Target"
$seeds = Join-Path $root "tests\fuzz\seeds\$Target"
New-Item -ItemType Directory -Force -Path $corpus, $artifacts | Out-Null
$fuzzArgs = @(
    "-max_total_time=$Seconds", "-seed=$Seed",
    '-timeout=2',           # an input taking > 2 s is a finding (quadratic behaviour)
    '-malloc_limit_mb=256', # an allocation > 256 MB is a finding (sizes trusted from the file)
    '-rss_limit_mb=2048',
    '-max_len=1048576',
    "-dict=$(Join-Path $root "tests\fuzz\$Target.dict")",
    "-artifact_prefix=$artifacts\",
    '-print_final_stats=1',
    $corpus, $seeds)
$code = Invoke-NativeTimeout -FilePath $exe -ArgumentList $fuzzArgs -TimeoutSec ($Seconds + 120)
if ($code -ne 0) {
    throw "fuzzer $Target found a problem: exit code $(Format-ExitCode $code); the input is in $artifacts"
}
Write-Host "fuzz: $Target ran $Seconds s (libFuzzer, ASan/UBSan) with no findings"
