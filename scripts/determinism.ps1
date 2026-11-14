# ACCEPTANCE "Determinism" for every backend, across processes and thread
# counts: the logits of the first 64 tokens of wiki.test.raw (BOS first) are
# dumped by separate `hearth dump` processes and compared by SHA-256.
#   - every backend: the same bits in every run and at every thread count;
#   - cpu: also compared with the hashes recorded on 2026-11-06, if given.
#
#   pwsh -File scripts\determinism.ps1 [-Models a.gguf,b.gguf] [-Backends cpu,gpu]
#                                      [-Threads 1,2,4,8] [-Report bench\release\determinism-YYYY-MM-DD.txt]
#
# About 2 minutes per 1B file and 4 per 3B file. Not a measurement.
[CmdletBinding()]
param(
    [string[]]$Models = @('models\Llama-3.2-1B-Instruct-Q8_0-pure.gguf', 'models\Llama-3.2-1B-Instruct-Q4_0-pure.gguf',
        'models\Llama-3.2-3B-Instruct-Q8_0-pure.gguf', 'models\Llama-3.2-3B-Instruct-Q4_0-pure.gguf'),
    [ValidateSet('cpu', 'gpu', 'cpu-ref')][string[]]$Backends = @('cpu', 'gpu'),
    [int[]]$Threads = @(1, 2, 4, 8),
    [string]$Report = ''
)
. "$PSScriptRoot\lib\common.ps1"

$root = Get-RepoRoot
if ($Backends | Where-Object { $_ -ne 'cpu' }) { Assert-GpuBudget 'the determinism check' }
$exe = Join-Path $root 'build\release\cli\hearth.exe'
if (-not (Test-Path $exe)) { throw "determinism: build Release first ($exe)" }
$env:VK_LOADER_LAYERS_DISABLE = '~implicit~'
if (-not $Report) { $Report = 'bench\release\determinism-' + (Get-Date -Format 'yyyy-MM-dd') + '.txt' }
$work = Join-Path $root 'bench\tmp\determinism'
New-Item -ItemType Directory -Force -Path $work | Out-Null
$wiki = Join-Path $root 'data\wikitext-2-raw\wiki.test.raw'
$noInput = Join-Path $work 'empty-stdin.txt'   # stdin closed: never interactive
Set-Content -Path $noInput -Value $null -NoNewline

# Hashes recorded for --backend cpu on 2026-11-06 (bench/cpu/determinism-*.txt).
$recorded = @{
    'Llama-3.2-1B-Instruct-Q8_0-pure' = '70be5931daf0ff037485672c60b0aa0e55505f022354c5c1ac9f8db777516df9'
    'Llama-3.2-1B-Instruct-Q4_0-pure' = '2a1ee0f68dee4c51f71002c3075f3769dacfbb844cf1bcf52b0227170ab7cece'
    'Llama-3.2-3B-Instruct-Q8_0-pure' = 'bca8fe3a33652cd286b5d9476479f34860c79afd73aa46624083c0dab5b68653'
    'Llama-3.2-3B-Instruct-Q4_0-pure' = '5f75928da6e1d6e9c0547d06f41ed5dd755318b17eaf4bfe3d2b6a9576bc0250'
}

function Invoke-Hearth([string[]]$a) {
    $out = Join-Path $work 'stdout.txt'
    $p = Start-Process -FilePath $exe -ArgumentList $a -RedirectStandardOutput $out -RedirectStandardError "$out.err" `
        -RedirectStandardInput $noInput -NoNewWindow -PassThru
    if (-not $p.WaitForExit(600 * 1000)) { $p.Kill(); throw "determinism: hearth $($a -join ' ') did not finish in 10 min" }
    if ($p.ExitCode -ne 0) { throw "determinism: hearth $($a -join ' ') failed ($($p.ExitCode)): $(Get-Content "$out.err" -Raw)" }
    Get-Content $out -Raw
}

$lines = [System.Collections.Generic.List[string]]::new()
$fails = 0
$commit = (& git -C $root describe --always --dirty --tags).Trim()
$lines.Add("Determinism of every backend across processes and thread counts (ACCEPTANCE `"Determinism`")")
$lines.Add("date:     $(Get-Date -Format 'yyyy-MM-dd HH:mm')")
$lines.Add("commit:   $commit")
$lines.Add("command:  pwsh -File scripts\determinism.ps1 -Backends $($Backends -join ',') -Threads $($Threads -join ',')")
$lines.Add("each run: hearth dump --model M --backend B --tokens-file <the first 64 tokens of wiki.test.raw, BOS first> --names result_output --positions all --ctx 64 --threads N; sha256 of result_output.bin (64 x n_vocab floats)")
$lines.Add('conditions: MSVC Release; AC')
$lines.Add('')

foreach ($m in $Models) {
    $model = Join-Path $root $m
    if (-not (Test-Path $model)) { $lines.Add("skip $m (not present)"); continue }
    $name = [System.IO.Path]::GetFileNameWithoutExtension($model)
    $ids = (Invoke-Hearth @('tokenize', '--model', $model, '--file', $wiki)).Trim().TrimStart('[').TrimEnd(']') -split ',\s*'
    $tokFile = Join-Path $work "$name-tokens.txt"
    $ids[0..63] | Set-Content -Path $tokFile -Encoding ascii
    $hashes = @{}
    foreach ($b in $Backends) {
        $runs = [System.Collections.Generic.List[object]]::new()
        foreach ($t in $Threads) { $runs.Add(@{ T = $t; Label = "threads $t run a" }) }
        foreach ($r in @('b', 'c')) { $runs.Add(@{ T = 4; Label = "threads 4 run $r" }) }
        $seen = @()
        foreach ($r in $runs) {
            $dir = Join-Path $work "$name-$b-$($r.T)"
            if (Test-Path $dir) { Remove-Item -Recurse -Force $dir }
            $a = @('dump', '--model', $model, '--backend', $b, '--tokens-file', $tokFile, '--names', 'result_output',
                '--positions', 'all', '--ctx', '64', '--threads', "$($r.T)", '--out', $dir)
            Invoke-Hearth $a | Out-Null
            $bin = Get-ChildItem -Path $dir -Recurse -Filter 'result_output*.bin' | Select-Object -First 1
            if (-not $bin) { throw "determinism: no result_output dump in $dir" }
            $h = (Get-FileHash -Algorithm SHA256 $bin.FullName).Hash.ToLowerInvariant()
            $seen += $h
            $lines.Add(("{0} {1,-6} {2}: {3}" -f $name, $b, $r.Label, $h))
            Remove-Item -Recurse -Force $dir
        }
        $distinct = @($seen | Select-Object -Unique)
        $hashes[$b] = $distinct[0]
        $ok = $distinct.Count -eq 1
        if (-not $ok) { $fails++ }
        $lines.Add(("gate {0}   {1} {2}: {3} runs, {4} distinct hash(es)" -f $(if ($ok) { 'ok  ' } else { 'FAIL' }), $name, $b, $seen.Count, $distinct.Count))
    }
    if ($hashes.ContainsKey('cpu') -and $recorded.ContainsKey($name)) {
        $same = $hashes['cpu'] -eq $recorded[$name]
        $lines.Add(("note       {0}: cpu hash {1} the one recorded on 2026-11-06" -f $name, $(if ($same) { 'equals' } else { 'differs from (the cpu backend changed since; not a gate)' })))
    }
    $lines.Add('')
}
$path = Join-Path $root $Report
$lines | Set-Content -Path $path -Encoding utf8
$lines | ForEach-Object { Write-Host $_ }
if ($fails -gt 0) { throw "determinism: $fails gate(s) failed (see $path)" }
Write-Host "determinism: all gates pass; report in $path"
