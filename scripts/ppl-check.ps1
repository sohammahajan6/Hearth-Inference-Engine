# Perplexity gate: hearth ppl against llama-perplexity
# (pinned b11960 CPU zip) on wikitext-2 test, ctx 512, the same chunks, both
# with an F32 KV cache. |dPPL| / PPL must be <= 0.2 % for a quantized file
# and <= 0.05 % for the F32 file. With -Against cpu the baseline is hearth's
# own `cpu` backend (ACCEPTANCE: the GPU backends are measured against the
# CPU path), and a GPU backend must log its device and dispatch count.
#
# Resumable: each finished run's output is kept in bench\tmp\ppl\<run>\ until
# the report is written, and a run found there is not repeated. With
# -MaxSeconds the check stops after the run that crosses the budget (exit 3,
# "partial") and the same command continues it, so a 3B check (two runs of
# several minutes each) fits short foreground steps. A partial run left by
# another commit is refused, not mixed in.
#
#   pwsh -File scripts\ppl-check.ps1 [-Model models\X.gguf] [-Backend cpu|cpu-ref|gpu]
#                                    [-Against llamacpp|cpu] [-Chunks 40] [-MaxSeconds S] [-ReportDir bench\tmp\x]
[CmdletBinding()]
param(
    [string]$Model = 'models\Llama-3.2-1B-Instruct-Q8_0-pure.gguf',
    [ValidateSet('cpu', 'cpu-ref', 'gpu')][string]$Backend = 'cpu',
    [ValidateSet('llamacpp', 'cpu')][string]$Against = 'llamacpp',
    [int]$Chunks = 40,
    [int]$Threads = 4,
    [int]$MaxSeconds = 0,  # 0: no limit; else stop after the run that crosses it (exit 3) and resume on the next call
    [string]$ReportDir = 'bench\oracle'  # trial runs: a folder under bench\tmp
)
. "$PSScriptRoot\lib\common.ps1"

$root = Get-RepoRoot
if ($Backend -eq 'gpu') { Assert-GpuBudget "ppl-check -Backend $Backend" }
$backendName = $Backend
if ($Against -eq 'cpu' -and $Backend -eq 'cpu') { throw '-Against cpu compares another backend with cpu; -Backend cpu would compare it with itself' }
# The tree as it is when the run starts (the binary under test was built from it); edits made while a long run is going on do not change what was measured.
$commit = (& git -C $root describe --always --dirty).Trim()
$modelPath = Join-Path $root $Model
$text = Join-Path $root 'data\wikitext-2-raw\wiki.test.raw'
$hearth = Join-Path $root 'build\release\cli\hearth.exe'
$llama = Join-Path $root 'third_party\llama-bin-cpu\llama-perplexity.exe'
foreach ($p in @($modelPath, $text, $hearth, $llama)) { if (-not (Test-Path $p)) { throw "missing $p" } }
$name = [System.IO.Path]::GetFileNameWithoutExtension($modelPath)
$bar = ($Model -match 'F32') ? 0.0005 : 0.002
$reportPath = Join-Path $root $ReportDir
New-Item -ItemType Directory -Force -Path $reportPath | Out-Null

function Get-Ppl([string]$Log, [string]$What) {
    $m = [regex]::Match($Log, 'Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)')
    if (-not $m.Success) { throw "$What printed no final estimate" }
    return [pscustomobject]@{ Ppl = [double]$m.Groups[1].Value; Err = [double]$m.Groups[2].Value }
}

$tmp = Join-Path $root "bench\tmp\ppl\$name-$backendName-vs-$Against-$Chunks"
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
$commitFile = Join-Path $tmp 'commit.txt'
if (Test-Path $commitFile) {
    $was = (Get-Content -Raw $commitFile).Trim()
    if ($was -ne $commit) { throw "a partial run of $was (not $commit) is in $tmp; delete that folder to start again" }
} else {
    [IO.File]::WriteAllText($commitFile, "$commit`n")
}
$callStart = [System.Diagnostics.Stopwatch]::StartNew()
# Runs one side, or reads it back if an earlier call finished it. Returns its output and seconds.
function Invoke-Side([string]$Key, [string]$What, [string]$Exe, [string[]]$ArgList, [int]$TimeoutSec) {
    $logFile = Join-Path $tmp "$Key.log"
    $secFile = Join-Path $tmp "$Key.seconds"
    if ((Test-Path $logFile) -and (Test-Path $secFile)) {
        Write-Host "ppl-check: $What, finished in an earlier call"
        return [pscustomobject]@{ Log = (Get-Content -Raw $logFile); Seconds = [double](Get-Content -Raw $secFile) }
    }
    Write-Host "ppl-check: $What, $Chunks chunks"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $r = Invoke-NativeCapture -FilePath $Exe -ArgumentList $ArgList -TimeoutSec $TimeoutSec
    if ($r.ExitCode -ne 0) { throw "$What failed: $(Format-ExitCode $r.ExitCode)`n$($r.Stderr)" }
    $log = $r.Stdout + $r.Stderr
    [IO.File]::WriteAllText($logFile, $log)
    [IO.File]::WriteAllText($secFile, "$([int]$sw.Elapsed.TotalSeconds)`n")  # last: the run counts as finished only now
    return [pscustomobject]@{ Log = $log; Seconds = $sw.Elapsed.TotalSeconds }
}
function Stop-IfOverBudget {
    if ($MaxSeconds -gt 0 -and $callStart.Elapsed.TotalSeconds -ge $MaxSeconds) {
        Write-Host "ppl-check: partial, the baseline is done; run the same command again to continue"
        exit 3
    }
}
if ($Against -eq 'llamacpp') {
    $l = Invoke-Side 'baseline' 'llama-perplexity' $llama @('-m', $modelPath, '-f', $text, '-c', '512', '--chunks', "$Chunks",
        '-fa', 'off', '-ctk', 'f32', '-ctv', 'f32', '-t', "$Threads") 7200
    $ref = Get-Ppl $l.Log 'llama-perplexity'
    $refWhat = 'llama.cpp:'
    $refHow = "llama-perplexity b11960 CPU zip, -fa off -ctk f32 -ctv f32"
} else {
    $l = Invoke-Side 'baseline' 'hearth ppl --backend cpu (baseline)' $hearth @('ppl', '--model', $modelPath, '--file', $text,
        '--ctx', '512', '--chunks', "$Chunks", '--backend', 'cpu', '--threads', "$Threads") 43200
    $ref = Get-Ppl $l.Log 'hearth ppl --backend cpu'
    $refWhat = 'baseline: '
    $refHow = 'hearth ppl --backend cpu'
}
$tLlama = $l.Seconds
Stop-IfOverBudget
$h = Invoke-Side 'ours' "hearth ppl --backend $Backend" $hearth (@('ppl', '--model', $modelPath, '--file', $text, '--ctx', '512',
        '--chunks', "$Chunks", '--backend', $Backend, '--threads', "$Threads")) 43200
$tHearth = $h.Seconds
$ours = Get-Ppl $h.Log 'hearth ppl'
$proof = @(Get-GpuProof -Log $h.Log -Backend $Backend)
$rel = [math]::Abs($ours.Ppl - $ref.Ppl) / $ref.Ppl
$ok = $rel -le $bar
$report = @(
    "ppl-check: $name, wikitext-2 test, ctx 512, $Chunks chunks, F32 KV cache on both sides"
    "date:      $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
    "commit:    $commit"
    "command:   pwsh -File scripts\ppl-check.ps1 -Model $Model -Backend $Backend -Against $Against -Chunks $Chunks -Threads $Threads"
    ("{0} PPL = {1:0.0000} +/- {2:0.00000} ({3}, {4} threads: {5:N0} s)" -f $refWhat, $ref.Ppl, $ref.Err, $refHow, $Threads, $tLlama)
    ("hearth:    PPL = {0:0.0000} +/- {1:0.00000} (hearth ppl --backend {2}, {3} threads: {4:N0} s)" -f $ours.Ppl, $ours.Err, $Backend, $Threads, $tHearth)
) + @($proof | ForEach-Object { "gpu log:   $_" }) + @(
    ("gate {0} |dPPL| / PPL = {1:0.0000} % (bar {2:0.00} %)" -f ($(if ($ok) { 'ok  ' } else { 'FAIL' })), ($rel * 100), ($bar * 100))
)
$file = if ($Against -eq 'llamacpp') { Join-Path $reportPath "ppl-$name-$backendName-$Chunks.txt" }
        else { Join-Path $reportPath "ppl-$name-$backendName-vs-$Against-$Chunks.txt" }
[IO.File]::WriteAllText($file, ($report -join "`n") + "`n")
Remove-Item -Recurse -Force $tmp  # the report holds what is kept
$report | ForEach-Object { Write-Host $_ }
if (-not $ok) { throw "ppl-check: |dPPL| / PPL = $([math]::Round($rel * 100, 4)) % > $($bar * 100) % ($file)" }
