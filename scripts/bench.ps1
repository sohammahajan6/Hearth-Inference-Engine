# Speed against llama.cpp (docs/architecture/07-fast-cpu-backend.md §5,
# 08-gpu-backend.md §6). `hearth bench` and llama-bench on the same file,
# pp512 and tg128 by default, interleaved A B / B A over the rounds with a
# cool-down before every run, so that neither side always runs hot.
#   -Backend cpu: hearth's cpu backend vs the pinned b11960 win-cpu-x64 zip.
#   -Backend gpu: hearth's gpu backend vs the pinned b11960 win-vulkan-x64 zip
#     with every layer on the GPU (-ngl 99), its default F16 KV cache,
#     GGML_VK_VISIBLE_DEVICES=0 (pinned to the Iris Xe; b11960 lists it once,
#     2026-11-07, `llama-bench --list-devices`) and
#     VK_LOADER_LAYERS_DISABLE=~implicit~. llama.cpp's Vulkan backend copies
#     the weights into its own buffers; hearth's imports them without a copy.
# Refuses to run if a compiler, engine or oracle process is running
# (one heavy process at a time), or on battery (hearth bench checks).
#
#   pwsh -File scripts\bench.ps1 [-Model models\X.gguf] [-Backend cpu|gpu] [-Rounds 2] [-Reps 5] [-Prompt 512] [-Gen 128]
#                                [-Threads 4] [-CooldownSec 30] [-BandwidthGBs G] [-ReportDir bench\cpu|bench\gpu]
#
# Raw outputs and a summary go to <ReportDir>\<model>\ (trial runs: a folder under bench\tmp).
# The headline per test is the median over rounds of each run's median tok/s.
[CmdletBinding()]
param(
    [string]$Model = 'models\Llama-3.2-1B-Instruct-Q8_0-pure.gguf',
    [ValidateSet('cpu', 'gpu')][string]$Backend = 'cpu',
    [int]$Rounds = 2,
    [int]$Reps = 5,
    [int]$Prompt = 512,
    [int]$Gen = 128,
    [int]$Threads = 4,
    [int]$CooldownSec = 30,
    # The read bandwidth hearth probe measured for the device:
    # CPU 39.1, GPU 40 GB/s by default. hearth bench fails if decode claims
    # more than 1.05 times it.
    [double]$BandwidthGBs = 0,
    [string]$ReportDir = '',
    # The index of this call's first round, so a race split into one-round calls (each under the
    # 10-minute limit) keeps the A B / B A order: -Rounds 1 -RoundOffset 0, then -RoundOffset 1.
    [int]$RoundOffset = 0,
    [switch]$AllowBusy
)
. "$PSScriptRoot\lib\common.ps1"

$root = Get-RepoRoot
if ($Backend -eq 'gpu') { Assert-GpuBudget "bench -Backend $Backend" }
$commit = (& git -C $root describe --always --dirty).Trim()
if ($BandwidthGBs -le 0) { $BandwidthGBs = ($Backend -eq 'gpu') ? 40.0 : 39.1 }
if (-not $ReportDir) { $ReportDir = "bench\$Backend" }
$modelPath = Join-Path $root $Model
$hearth = Join-Path $root 'build\release\cli\hearth.exe'
$llamaDir = ($Backend -eq 'gpu') ? 'llama-bin-vulkan' : 'llama-bin-cpu'
$llama = Join-Path $root "third_party\$llamaDir\llama-bench.exe"
$llamaWhat = ($Backend -eq 'gpu') ? 'llama-bench (b11960 win-vulkan-x64 zip, -ngl 99, -fa off, its default F16 KV cache, GGML_VK_VISIBLE_DEVICES=0)' `
    : 'llama-bench (b11960 win-cpu-x64 zip, -fa off, its default F16 KV cache)'
# Both engines see the same Vulkan environment.
$env:VK_LOADER_LAYERS_DISABLE = '~implicit~'
if ($Backend -eq 'gpu') { $env:GGML_VK_VISIBLE_DEVICES = '0' }
foreach ($p in @($modelPath, $hearth, $llama)) { if (-not (Test-Path $p)) { throw "missing $p" } }
$name = [System.IO.Path]::GetFileNameWithoutExtension($modelPath)
$out = Join-Path $root (Join-Path $ReportDir $name)
New-Item -ItemType Directory -Force -Path $out | Out-Null

if (-not $AllowBusy) {
    $busy = @(Get-CimInstance Win32_Process | Where-Object {
            $_.Name -match '^(cl|link|ninja|clang|clang\+\+|ld\.lld|hearth|hearth_tests|hetero-oracle-dump|hetero-oracle-chat|llama-.*|hearth_fuzz_.*)\.exe$'
        } | ForEach-Object { "$($_.Name) ($($_.ProcessId))" })
    if ($busy.Count -gt 0) { throw "bench: heavy processes are running: $($busy -join ', ') (-AllowBusy measures anyway)" }
}

function Get-Stats([double[]]$Values) {
    $v = @($Values | Sort-Object)
    $at = { param($p) $pos = $p * ($v.Count - 1); $lo = [math]::Floor($pos); $hi = [math]::Min($lo + 1, $v.Count - 1)
        $v[$lo] + ($v[$hi] - $v[$lo]) * ($pos - $lo) }
    return [pscustomobject]@{ Median = (& $at 0.5); P25 = (& $at 0.25); P75 = (& $at 0.75) }
}

# One run of each engine; returns @{ pp = stats; tg = stats } of tok/s over the repetitions.
function Invoke-Hearth([string]$Tag) {
    $r = Invoke-NativeCapture -FilePath $hearth -ArgumentList @('bench', '--model', $modelPath, '--prompt', "$Prompt",
        '--gen', "$Gen", '--reps', "$Reps", '--threads', "$Threads", '--bandwidth', "$BandwidthGBs", '--backend', $Backend) -TimeoutSec 3600
    [IO.File]::WriteAllText((Join-Path $out "hearth-$Tag.txt"), $r.Stdout + $r.Stderr)
    if ($r.ExitCode -ne 0) { throw "hearth bench failed: $(Format-ExitCode $r.ExitCode)`n$($r.Stderr)" }
    # A gpu run must prove it ran on the GPU (08 doc section 4); never a silent CPU number.
    $script:gpuProof = @(Get-GpuProof -Log ($r.Stdout + $r.Stderr) -Backend $Backend)
    $res = @{}
    foreach ($m in [regex]::Matches($r.Stdout, '(?m)^(pp|tg)\d+\s+([0-9.]+)\s+([0-9.]+) \.\. ([0-9.]+)')) {
        $res[$m.Groups[1].Value] = [pscustomobject]@{ Median = [double]$m.Groups[2].Value; P25 = [double]$m.Groups[3].Value
            P75 = [double]$m.Groups[4].Value }
    }
    return $res
}

function Invoke-Llama([string]$Tag) {
    $gpuArgs = ($Backend -eq 'gpu') ? @('-ngl', '99') : @()
    $r = Invoke-NativeCapture -FilePath $llama -ArgumentList (@('-m', $modelPath, '-p', "$Prompt", '-n', "$Gen", '-r', "$Reps",
        '-t', "$Threads", '-fa', 'off', '-o', 'json') + $gpuArgs) -TimeoutSec 3600
    [IO.File]::WriteAllText((Join-Path $out "llama-bench-$Tag.json"), $r.Stdout)
    [IO.File]::WriteAllText((Join-Path $out "llama-bench-$Tag.log"), $r.Stderr)
    if ($r.ExitCode -ne 0) { throw "llama-bench failed: $(Format-ExitCode $r.ExitCode)`n$($r.Stderr)" }
    $res = @{}
    foreach ($t in ($r.Stdout | ConvertFrom-Json)) {
        if ($t.n_prompt -gt 0 -and $t.n_gen -eq 0) { $res['pp'] = Get-Stats ([double[]]$t.samples_ts) }
        elseif ($t.n_gen -gt 0 -and $t.n_prompt -eq 0) { $res['tg'] = Get-Stats ([double[]]$t.samples_ts) }
    }
    return $res
}

$script:gpuProof = @()
$runs = [System.Collections.Generic.List[object]]::new()
for ($i = 0; $i -lt $Rounds; ++$i) {
    $order = if (($i + $RoundOffset) % 2 -eq 0) { @('hearth', 'llama') } else { @('llama', 'hearth') }
    foreach ($engine in $order) {
        Write-Host "bench: round $($i + 1)/$Rounds, $engine (after a $CooldownSec s cool-down)"
        Start-Sleep -Seconds $CooldownSec
        # Files are named by the race's round, so the one-round calls of a split race do not overwrite each other.
        $tag = "r$($i + 1 + $RoundOffset)"
        $res = if ($engine -eq 'hearth') { Invoke-Hearth $tag } else { Invoke-Llama $tag }
        foreach ($want in @(@('pp', $Prompt), @('tg', $Gen))) {
            if ($want[1] -gt 0 -and -not $res.ContainsKey($want[0])) {
                throw "bench: no $($want[0])$($want[1]) result parsed from $engine's output (raw output in $out)"
            }
        }
        $runs.Add([pscustomobject]@{ Engine = $engine; Round = $i + 1; Result = $res })
    }
}

$lines = @(
    "bench: $name, hearth bench --backend $Backend vs $llamaWhat, $Threads threads"
    "date:    $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
    "commit:  $commit"
    "command: pwsh -File scripts\bench.ps1 -Model $Model -Backend $Backend -Rounds $Rounds -Reps $Reps -Prompt $Prompt -Gen $Gen -Threads $Threads -CooldownSec $CooldownSec$(if ($RoundOffset) { " -RoundOffset $RoundOffset" })"
    "order:   rounds alternate A B / B A, $CooldownSec s cool-down before every run, $Reps repetitions per run"
) + @($script:gpuProof | ForEach-Object { "gpu log (last hearth run): $_" }) + @(
    ''
    ('{0,-7} {1,-8} {2,-34} {3,10}' -f 'test', 'engine', 'per-round median tok/s (IQR)', 'median')
)
$headline = @{}
foreach ($test in @(@('pp', $Prompt), @('tg', $Gen))) {
    if ($test[1] -eq 0) { continue }
    foreach ($engine in @('hearth', 'llama')) {
        $per = @($runs | Where-Object { $_.Engine -eq $engine } | ForEach-Object { $_.Result[$test[0]] })
        $text = ($per | ForEach-Object { '{0:N1} ({1:N1}..{2:N1})' -f $_.Median, $_.P25, $_.P75 }) -join ', '
        $med = (Get-Stats ([double[]]($per | ForEach-Object { $_.Median }))).Median
        $headline["$($test[0])-$engine"] = $med
        $lines += ('{0,-7} {1,-8} {2,-34} {3,10:N1}' -f "$($test[0])$($test[1])", $engine, $text, $med)
    }
    $lines += ('{0,-7} hearth / llama.cpp = {1:N2}' -f "$($test[0])$($test[1])", ($headline["$($test[0])-hearth"] / $headline["$($test[0])-llama"]))
}
$summary = Join-Path $out $(if ($RoundOffset -gt 0) { "summary-from-r$($RoundOffset + 1).txt" } else { 'summary.txt' })
[IO.File]::WriteAllText($summary, ($lines -join "`n") + "`n")
$lines | ForEach-Object { Write-Host $_ }
Write-Host "bench: raw outputs and summary in $out"
