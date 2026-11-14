# Phase 3 gate: hearth's forward pass against llama.cpp's, layer by layer
#.
#   1. The first -Tokens tokens of wikitext-2 (test set), tokenized by the
#      pinned llama-tokenize, BOS first.
#   2. The reference: hetero-oracle-dump (tools/oracle), llama.cpp b11960 with
#      an F32 KV cache; or, with -Reference cpu, hearth's own `cpu` backend,
#      the reference ACCEPTANCE names for the GPU backend.
#   3. hearth dump with -Backend.
#   4. tools/layer_diff.py, with -Gate. `identical` requires the same bits in
#      every tensor (it was the bar of the removed bit-identical backends,
#      docs/architecture/09-hetero.md).
#
#   pwsh -File scripts\oracle-check.ps1 [-Model models\X.gguf] [-Backend cpu-ref|cpu|gpu]
#                                       [-Reference llamacpp|cpu]
#                                       [-Gate fp32|quant|quant-short|quant-k|quant-k-short|quant-qwen|quant-qwen-short|identical|none]
#                                       [-Tokens 2048] [-Names l_out,result_output] [-ReportDir D] [-Keep]
#
# A GPU backend never falls back to the CPU in `dump` (08 doc section 5); the
# report also carries its log lines with the device and the dispatch count.
#
# The two runs are sequential (never two copies of a large model in RAM). The
# dumps go to bench/tmp/oracle/ (gitignored) and are deleted unless -Keep; the
# report, with timings and conditions, goes to bench/oracle/ (committed).
[CmdletBinding()]
param(
    [string]$Model = 'models\Llama-3.2-1B-Instruct-F32-pure.gguf',
    [ValidateSet('cpu-ref', 'cpu', 'gpu')][string]$Backend = 'cpu-ref',
    [ValidateSet('llamacpp', 'cpu')][string]$Reference = 'llamacpp',
    [ValidateSet('fp32', 'quant', 'quant-short', 'quant-k', 'quant-k-short', 'quant-qwen', 'quant-qwen-short', 'identical', 'none')][string]$Gate = 'fp32',
    [int]$Tokens = 2048,
    [string]$Names = 'l_out,result_output',
    [int]$Threads = 4,
    # Where the report goes. CI passes a folder under bench\tmp, so routine
    # runs do not rewrite the committed reports.
    [string]$ReportDir = 'bench\oracle',
    [switch]$Keep
)
. "$PSScriptRoot\lib\common.ps1"

$root = Get-RepoRoot
if ($Backend -eq 'gpu') { Assert-GpuBudget "oracle-check -Backend $Backend" }
$modelPath = Join-Path $root $Model
$text = Join-Path $root 'data\wikitext-2-raw\wiki.test.raw'
$tokenize = Join-Path $root 'third_party\llama-bin-cpu\llama-tokenize.exe'
$oracle = Join-Path $root 'third_party\install\oracle\bin\hetero-oracle-dump.exe'
$hearth = Join-Path $root 'build\release\cli\hearth.exe'
$py = Join-Path $root '.venv\Scripts\python.exe'
$need = [ordered]@{
    $modelPath = 'scripts\fetch-models.ps1'; $text = 'scripts\fetch-models.ps1'; $tokenize = 'scripts\fetch-deps.ps1'
    $oracle = 'scripts\setup.ps1 -Only oracle'; $hearth = 'scripts\build.ps1'; $py = 'scripts\setup.ps1 -Only venv'
}
if ($Reference -eq 'cpu') {
    $need.Remove($oracle)
    if ($Backend -eq 'cpu') { throw '-Reference cpu compares another backend with cpu; -Backend cpu would compare it with itself' }
}
foreach ($p in $need.Keys) { if (-not (Test-Path $p)) { throw "missing $p (run: pwsh -File $($need[$p]))" } }
$backendName = $Backend
$name = [System.IO.Path]::GetFileNameWithoutExtension($modelPath)
$tmp = Join-Path $root "bench\tmp\oracle\$name-$Tokens-$backendName-vs-$Reference"
$reportPath = Join-Path $root $ReportDir
New-Item -ItemType Directory -Force -Path $tmp, $reportPath | Out-Null
Assert-FreeSpace -NeedGB 3

# 1. Token ids. llama-tokenize prints them as one list: [128000, 220, ...].
$tok = Invoke-NativeCapture -FilePath $tokenize -ArgumentList @('-m', $modelPath, '-f', $text, '--ids', '--log-disable', '--no-escape', '--no-parse-special') `
    -TimeoutSec 600
if ($tok.ExitCode -ne 0) { throw "llama-tokenize failed: $(Format-ExitCode $tok.ExitCode)`n$($tok.Stderr)" }
$list = [regex]::Match($tok.Stdout, '\[([0-9, ]+)\]')
if (-not $list.Success) { throw 'llama-tokenize printed no id list' }
$ids = @($list.Groups[1].Value -split ',\s*' | Select-Object -First $Tokens)
if ($ids.Count -lt $Tokens) { throw "wikitext-2 test has only $($ids.Count) tokens" }
$tokFile = Join-Path $tmp 'tokens.txt'
[System.IO.File]::WriteAllText($tokFile, ($ids -join "`n") + "`n")

function Invoke-Timed([string]$What, [string]$Exe, [string[]]$ArgList, [int]$TimeoutSec) {
    Write-Host "oracle-check: $What"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $r = Invoke-NativeCapture -FilePath $Exe -ArgumentList $ArgList -TimeoutSec $TimeoutSec
    $sec = [math]::Round($sw.Elapsed.TotalSeconds, 1)
    if ($r.ExitCode -ne 0) { throw "$What failed: $(Format-ExitCode $r.ExitCode)`n$($r.Stdout)`n$($r.Stderr)" }
    return [pscustomobject]@{ Seconds = $sec; Log = $r.Stdout + $r.Stderr }
}

# 2 and 3: the two forward passes, one after the other.
$oDir = Join-Path $tmp "ref-$Reference"
$hDir = Join-Path $tmp $Backend
if ($Reference -eq 'llamacpp') {
    $refLine = 'llama.cpp: hetero-oracle-dump, one batch, F32 KV cache, flash attention off'
    $o = Invoke-Timed 'llama.cpp (hetero-oracle-dump, F32 KV cache)' $oracle @('--model', $modelPath, '--tokens-file', $tokFile,
        '--out', $oDir, '--names', $Names, '--positions', 'all', '--ctx', "$Tokens", '--f32-kv', '--threads', "$Threads") 7200
} else {
    $refLine = 'reference: hearth dump --backend cpu, F32 KV cache'
    $o = Invoke-Timed 'hearth dump --backend cpu (reference)' $hearth @('dump', '--model', $modelPath, '--tokens-file', $tokFile,
        '--out', $oDir, '--names', $Names, '--positions', 'all', '--ctx', "$Tokens", '--threads', "$Threads",
        '--backend', 'cpu') 14400
}
$h = Invoke-Timed "hearth dump --backend $Backend" $hearth (@('dump', '--model', $modelPath, '--tokens-file', $tokFile,
    '--out', $hDir, '--names', $Names, '--positions', 'all', '--ctx', "$Tokens", '--threads', "$Threads",
    '--backend', $Backend)) 14400

# 4. The comparison.
$gateArgs = if ($Gate -ne 'none') { @('--gate', $Gate) } else { @() }
$d = Invoke-NativeCapture -FilePath $py -ArgumentList (@('-I', (Join-Path $root 'tools\layer_diff.py'), $hDir, $oDir) + $gateArgs) `
    -TimeoutSec 3600
$commit = (& git -C $root describe --always --dirty).Trim()
$power = & { $s = Get-CimInstance Win32_Battery; if ($s.BatteryStatus -eq 2) { 'AC' } else { "battery status $($s.BatteryStatus)" } }
$header = @(
    "oracle-check: $name, backend $Backend, reference $Reference, gate $Gate"
    "date:      $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
    "commit:    $commit"
    "command:   pwsh -File scripts\oracle-check.ps1 -Model $Model -Backend $Backend -Reference $Reference -Gate $Gate -Tokens $Tokens -Names $Names -Threads $Threads -ReportDir $ReportDir"
    "tokens:    the first $Tokens of wikitext-2 test (llama-tokenize, b11960), teacher-forced"
    "$($refLine), $Threads threads: $($o.Seconds) s"
    "hearth:    hearth dump --backend $Backend, F32 KV cache, $Threads threads: $($h.Seconds) s"
)
# Proof of execution (08 doc section 4): the device and the dispatch count, from hearth's log.
$header += @(Get-GpuProof -Log $h.Log -Backend $Backend | ForEach-Object { "gpu log:   $_" })
$header += @(
    "machine:   $env:PROCESSOR_IDENTIFIER, $power"
    ''
)
$report = ($header + @($d.Stdout.TrimEnd())) -join "`n"
$reportFile = if ($Reference -eq 'llamacpp') { Join-Path $reportPath "$name-$backendName-$Tokens.txt" }
              else { Join-Path $reportPath "$name-$backendName-vs-$Reference-$Tokens.txt" }
[System.IO.File]::WriteAllText($reportFile, $report + "`n")
Write-Host $report
if ($d.ExitCode -ne 0) { throw "layer_diff: $(if ($Gate -ne 'none') { "gate $Gate failed" } else { 'failed' }) ($reportFile)`n$($d.Stderr)" }
if (-not $Keep) { Remove-Item -Recurse -Force $tmp }
Write-Host "oracle-check: report in $reportFile"
