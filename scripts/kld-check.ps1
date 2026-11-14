# Quantized gate at >= 10k positions.
# The positions are llama-perplexity's: wikitext-2 test, chunks of 512 with BOS
# first if the vocabulary adds one (Llama 3: 128000; Qwen2.5 adds none, so its
# chunks keep their own first token), the logits at positions 256..510 of each
# chunk (40 chunks: 10,200).
# For every chunk, one after the other:
#   the baseline on the quantized file: llama.cpp, or with -Against cpu
#   hearth's own `cpu` backend (the baseline ACCEPTANCE names for the GPU
#   backends); llama.cpp on the reference file (F32; BF16 for the 3B, ADR 0011);
#   hearth dump --backend -Backend; all with an F32 KV cache;
#   tools/layer_diff.py --save keeps the per-position statistics; the dumps
#   are deleted.
# tools/kld_aggregate.py then applies the ADR 0007 bars over all positions.
# A GPU backend must log its device and dispatch count in every chunk.
#
# Resumable: the per-chunk statistics are kept in bench\tmp\kld\<run>\stats
# until the report is written, and chunks found there are skipped. With
# -MaxSeconds the run stops after the chunk that crosses the budget (exit 3,
# "partial") and the same command continues it; the report's time is the sum
# over the calls. This lets a long gate run as short foreground steps.
#
#   pwsh -File scripts\kld-check.ps1 [-Model models\X.gguf] [-Backend cpu|gpu] [-Against llamacpp|cpu]
#                                    [-Chunks 40] [-MaxSeconds S] [-ReportDir bench\tmp\x]
[CmdletBinding()]
param(
    [string]$Model = 'models\Llama-3.2-1B-Instruct-Q8_0-pure.gguf',
    [string]$Reference = 'models\Llama-3.2-1B-Instruct-F32-pure.gguf',
    [ValidateSet('cpu', 'gpu')][string]$Backend = 'cpu',
    [ValidateSet('llamacpp', 'cpu')][string]$Against = 'llamacpp',
    [int]$Chunks = 40,
    [int]$MaxSeconds = 0,  # 0: no limit; else stop after the chunk that crosses it (exit 3) and resume on the next call
    [int]$Threads = 4,
    [string]$ReportDir = 'bench\oracle'  # trial runs: a folder under bench\tmp
)
. "$PSScriptRoot\lib\common.ps1"

$root = Get-RepoRoot
if ($Backend -eq 'gpu') { Assert-GpuBudget "kld-check -Backend $Backend" }
$backendName = $Backend
if ($Against -eq 'cpu' -and $Backend -eq 'cpu') { throw '-Against cpu compares another backend with cpu; -Backend cpu would compare it with itself' }
# The tree as it is when the run starts (the binary under test was built from it); edits made while a long run is going on do not change what was measured.
$commit = (& git -C $root describe --always --dirty).Trim()
$modelPath = Join-Path $root $Model
$refPath = Join-Path $root $Reference
$text = Join-Path $root 'data\wikitext-2-raw\wiki.test.raw'
$hearth = Join-Path $root 'build\release\cli\hearth.exe'
$oracle = Join-Path $root 'third_party\install\oracle\bin\hetero-oracle-dump.exe'
$py = Join-Path $root '.venv\Scripts\python.exe'
foreach ($p in @($modelPath, $refPath, $text, $hearth, $oracle, $py)) { if (-not (Test-Path $p)) { throw "missing $p" } }
$name = [System.IO.Path]::GetFileNameWithoutExtension($modelPath)
$refName = [System.IO.Path]::GetFileNameWithoutExtension($refPath)
$tmp = Join-Path $root "bench\tmp\kld\$name-$backendName-vs-$Against"
$stats = Join-Path $tmp 'stats'
$reportPath = Join-Path $root $ReportDir
New-Item -ItemType Directory -Force -Path $stats, $reportPath | Out-Null
Assert-FreeSpace -NeedGB 2

# The whole file's tokens, as hearth ppl and llama-perplexity tokenize it.
$t = Invoke-NativeCapture -FilePath $hearth -ArgumentList @('tokenize', '--model', $modelPath, '--file', $text) -TimeoutSec 600
if ($t.ExitCode -ne 0) { throw "hearth tokenize failed: $(Format-ExitCode $t.ExitCode)" }
$all = @([regex]::Match($t.Stdout, '\[([0-9, ]+)\]').Groups[1].Value -split ',\s*')
if ($all.Count -lt $Chunks * 512) { throw "only $($all.Count) tokens for $Chunks chunks" }
# The BOS that llama-perplexity (and hearth ppl) put first in every chunk: only
# if the vocabulary adds one. Before 2026-11-14 this script always wrote Llama
# 3's 128000, an ordinary token in Qwen2.5's vocabulary.
$idsOf = {
    param([string[]]$extra)
    $r = Invoke-NativeCapture -FilePath $hearth -ArgumentList (@('tokenize', '--model', $modelPath, '--text', 'a') + $extra) -TimeoutSec 120
    if ($r.ExitCode -ne 0) { throw "hearth tokenize failed: $(Format-ExitCode $r.ExitCode)" }
    , @([regex]::Match($r.Stdout, '\[([0-9, ]+)\]').Groups[1].Value -split ',\s*')
}
$withBos = & $idsOf @()
$withoutBos = & $idsOf @('--no-bos')
$bos = if ($withBos.Count -eq $withoutBos.Count + 1) { $withBos[0] } elseif ($withBos.Count -eq $withoutBos.Count) { $null } else { throw "cannot tell the BOS from '$($withBos -join ',')' and '$($withoutBos -join ',')'" }
$baseName = if ($Against -eq 'llamacpp') { 'llama.cpp b11960' } else { 'hearth cpu' }
$proofFile = Join-Path $stats 'gpu-proof.txt'
$timeFile = Join-Path $stats 'seconds.txt'
$firstProof = @(if (Test-Path $proofFile) { Get-Content $proofFile })  # @(): an empty if-result would be $null
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$done = { param($k) @('agreement', 'quality-ours', 'quality-llama' | Where-Object { Test-Path (Join-Path $stats ("{0}-{1:00}.npz" -f $_, $k)) }).Count -eq 3 }
for ($i = 0; $i -lt $Chunks; ++$i) {
    if (& $done $i) { continue }  # resumed: this chunk's statistics exist
    $chunk = $all[($i * 512)..($i * 512 + 511)]
    if ($null -ne $bos) { $chunk[0] = $bos }  # BOS first, as llama-perplexity evaluates every chunk
    $dir = Join-Path $tmp "chunk"
    if (Test-Path $dir) { Remove-Item -Recurse -Force $dir }
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $tok = Join-Path $dir 'tokens.txt'
    [IO.File]::WriteAllText($tok, ($chunk -join "`n") + "`n")
    $common = @('--tokens-file', $tok, '--names', 'result_output', '--positions', '256-510', '--ctx', '512', '--threads', "$Threads")
    $baseline = if ($Against -eq 'llamacpp') { @($oracle, (@('--model', $modelPath, '--out', (Join-Path $dir 'llama'), '--f32-kv') + $common)) }
                else { @($hearth, (@('dump', '--model', $modelPath, '--out', (Join-Path $dir 'llama'), '--backend', 'cpu') + $common)) }
    foreach ($run in @(
            $baseline,
            @($oracle, (@('--model', $refPath, '--out', (Join-Path $dir 'f32'), '--f32-kv') + $common)),
            @($hearth, (@('dump', '--model', $modelPath, '--out', (Join-Path $dir 'ours'), '--backend', $Backend) + $common)))) {
        $r = Invoke-NativeCapture -FilePath $run[0] -ArgumentList $run[1] -TimeoutSec 3600
        if ($r.ExitCode -ne 0) { throw "chunk ${i}: $($run[0]) failed: $(Format-ExitCode $r.ExitCode)`n$($r.Stderr)" }
    }
    # The last run is ours: a GPU backend must prove it ran on the GPU, every chunk.
    $proof = @(Get-GpuProof -Log ($r.Stdout + $r.Stderr) -Backend $Backend)
    if ($firstProof.Count -eq 0 -and $proof.Count -gt 0) { $firstProof = $proof; [IO.File]::WriteAllLines($proofFile, [string[]]$proof) }
    foreach ($pair in @(@('ours', 'llama', 'agreement'), @('ours', 'f32', 'quality-ours'), @('llama', 'f32', 'quality-llama'))) {
        $d = Invoke-NativeCapture -FilePath $py -ArgumentList @('-I', (Join-Path $root 'tools\layer_diff.py'), (Join-Path $dir $pair[0]),
            (Join-Path $dir $pair[1]), '--save', (Join-Path $stats ("{0}-{1:00}.npz" -f $pair[2], $i))) -TimeoutSec 600
        if ($d.ExitCode -ne 0) { throw "chunk ${i}: layer_diff failed: $($d.Stderr)" }
    }
    Remove-Item -Recurse -Force $dir
    Write-Host ("kld-check: chunk {0}/{1} done ({2:N0} s in this call)" -f ($i + 1), $Chunks, $sw.Elapsed.TotalSeconds)
    if ($MaxSeconds -gt 0 -and $sw.Elapsed.TotalSeconds -ge $MaxSeconds -and $i + 1 -lt $Chunks) {
        Add-Content -Path $timeFile -Value ([int]$sw.Elapsed.TotalSeconds)
        Write-Host "kld-check: partial, $($i + 1) of $Chunks chunks done; run the same command again to continue"
        exit 3
    }
}
Add-Content -Path $timeFile -Value ([int]$sw.Elapsed.TotalSeconds)
$totalSeconds = (@(Get-Content $timeFile) | ForEach-Object { [int]$_ } | Measure-Object -Sum).Sum
$files = { param($kind) @(Get-ChildItem $stats -Filter "$kind-*.npz" | Sort-Object Name | ForEach-Object { $_.FullName }) }
$agg = Invoke-NativeCapture -FilePath $py -ArgumentList (@('-I', (Join-Path $root 'tools\kld_aggregate.py'), '--agreement') +
    (& $files 'agreement') + @('--quality-ours') + (& $files 'quality-ours') + @('--quality-llama') + (& $files 'quality-llama') +
    @('--reference-name', $refName, '--ours-name', "hearth $Backend", '--baseline-name', $baseName) +
    $(if (Test-KQuantModel $modelPath) { @('--bars', 'adr0022') } elseif (Test-QwenModel $modelPath) { @('--bars', 'adr0023') } else { @() })) -TimeoutSec 600
$header = @(
    "kld-check: $name, hearth --backend $Backend vs $baseName and $refName, $Chunks chunks x 255 positions (llama-perplexity's), F32 KV cache"
    "date:    $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
    "commit:  $commit"
    "bos:     $(if ($null -ne $bos) { "$bos first in every chunk" } else { 'none (the vocabulary adds none; each chunk keeps its first token)' })"
    "command: pwsh -File scripts\kld-check.ps1 -Model $Model -Reference $Reference -Backend $Backend -Against $Against -Chunks $Chunks -Threads $Threads"
    ("time:    {0:N0} s ({1} call(s))" -f $totalSeconds, @(Get-Content $timeFile).Count)
)
$header += @($firstProof | ForEach-Object { "gpu log (first chunk run): $_" })
$header += ''
$file = if ($Backend -eq 'cpu' -and $Against -eq 'llamacpp') { Join-Path $reportPath "kld-$name-$Chunks.txt" }
        else { Join-Path $reportPath "kld-$name-$backendName-vs-$Against-$Chunks.txt" }
[IO.File]::WriteAllText($file, (($header + @($agg.Stdout.TrimEnd())) -join "`n") + "`n")
Write-Host $agg.Stdout
if ($agg.ExitCode -ne 0) { throw "kld-check: a gate failed ($file)" }
Remove-Item -Recurse -Force $tmp
