# Greedy-generation gate: 10 prompts x 128 tokens at
# temperature 0, hearth against llama.cpp from the same prompt IDs.
#   prompt IDs: hearth tokenize --chat (equal to llama.cpp's own template
#               rendering; scripts/chat-check.ps1)
#   llama.cpp:  hetero-oracle-dump --generate (F32 KV cache): its tokens and,
#               at every step, the top-2 probability gap of its distribution
#   hearth:     hearth run --tokens ... --temp 0 --ids
# The two streams (each ending with the end-of-generation token, if one was
# chosen) must be equal up to the first position where llama.cpp's top-2 gap
# is below -Gap: a divergence is allowed only at a near-tie.
# With -Against cpu the reference is hearth's own `cpu` backend (ACCEPTANCE:
# the GPU backends are measured against the CPU path): its stream from
# `hearth run --backend cpu`, and its top-2 gap at a divergence from a
# teacher-forced `hearth dump --backend cpu` of the shared prefix (the cpu
# backend's prefill equals its decode bit for bit, so that is the gap it
# decoded with). A GPU backend must log its device and dispatch count.
#
# The near-tie bar (-Gap, default): 0.005 for an F32 file and 0.05 otherwise
# (ACCEPTANCE). ADR 0016 (interim):
# for the gpu backend against cpu, a divergence where cpu's gap is at least
# 0.05 but below 0.10 also passes if KLD(cpu || gpu) at that position,
# teacher-forced, is at most 5e-3 (the measured floor of ADR 0007).
# ADR 0022: a K-quant file (Test-KQuantModel) against llama.cpp has the bar
# 0.10, and a divergence where llama.cpp's gap (the smaller of its decoding
# and teacher-forced ones) is at least 0.10 but below 0.20 also passes if
# KLD(llama.cpp || hearth) at that position, teacher-forced, is at most 2e-2:
# its Q8_K activations put the floor of correct implementations ~3.5x higher.
#
#   pwsh -File scripts\greedy-check.ps1 [-Model models\X.gguf] [-Backend cpu|cpu-ref|gpu]
#                                       [-Draft models\D.gguf] [-Against llamacpp|cpu] [-Gap G] [-Tokens 128]
#                                       [-ReportDir bench\tmp\x]
[CmdletBinding()]
param(
    [string]$Model = 'models\Llama-3.2-1B-Instruct-Q8_0-pure.gguf',
    [ValidateSet('cpu', 'cpu-ref', 'gpu')][string]$Backend = 'cpu',
    # Speculative decoding: the run under test drafts with this
    # model; greedy output must be the same as without it.
    [string]$Draft = '',
    [ValidateSet('llamacpp', 'cpu')][string]$Against = 'llamacpp',
    [double]$Gap = -1,  # default: see above (ACCEPTANCE, ADR 0016)
    [int]$Tokens = 128,
    # A subset of the 10 prompts, by index (default all), so a long run can be split into steps
    # that each fit a short time limit; the report name then carries the indices.
    # Text, not [int[]]: through `pwsh -File`, "0,1" arrives as one argument, and an int
    # parameter reads the comma as a thousands separator (0,1 -> 1, 0,1,2,3,4 -> 1234).
    [string[]]$Only = @(),  # not $Prompts: PowerShell names ignore case, and $prompts is the list below
    [int]$Threads = 4,
    [string]$ReportDir = 'bench\oracle'  # trial runs: a folder under bench\tmp
)
. "$PSScriptRoot\lib\common.ps1"

[int[]]$Only = @($Only | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ } |
        ForEach-Object { if ($_ -notmatch '^[0-9]+$') { throw "-Only: '$_' is not a prompt index" }; [int]$_ })
$root = Get-RepoRoot
if ($Backend -eq 'gpu') { Assert-GpuBudget "greedy-check -Backend $Backend" }
if ($Against -eq 'cpu' -and $Backend -eq 'cpu') { throw '-Against cpu compares another backend with cpu; -Backend cpu would compare it with itself' }
# The tree as it is when the run starts (the binary under test was built from it); edits made while a long run is going on do not change what was measured.
$commit = (& git -C $root describe --always --dirty).Trim()
$modelPath = Join-Path $root $Model
$hearth = Join-Path $root 'build\release\cli\hearth.exe'
$oracle = Join-Path $root 'third_party\install\oracle\bin\hetero-oracle-dump.exe'
foreach ($p in @($modelPath, $hearth, $oracle)) { if (-not (Test-Path $p)) { throw "missing $p" } }
$kQuant = (Test-KQuantModel $modelPath) -and $Against -eq 'llamacpp'
if ($Gap -lt 0) {
    $Gap = if ($Model -match 'F32') { 0.005 }
           elseif ($kQuant) { 0.10 }
           else { 0.05 }
}
# ADR 0016: the conditional band for the gpu backend against cpu (quantized
# files); ADR 0022: the one for K-quant files against llama.cpp.
$wideGap = if ($Against -eq 'cpu' -and $Backend -eq 'gpu' -and $Model -notmatch 'F32') { 0.10 } elseif ($kQuant) { 0.20 } else { 0.0 }
$wideKld = if ($kQuant) { 2e-2 } else { 5e-3 }
$bandAdr = if ($kQuant) { 'ADR 0022' } else { 'ADR 0016' }
$backendArgs = @('--backend', $Backend, '--no-fallback')
if ($Draft) { $backendArgs += @('--draft', (Join-Path $root $Draft)) }
$name = [System.IO.Path]::GetFileNameWithoutExtension($modelPath)
$backendName = $Backend
if ($Draft) { $backendName += '+draft' }
$tmp = Join-Path $root "bench\tmp\greedy\$name-$backendName-vs-$Against"
$refName = if ($Against -eq 'llamacpp') { 'llama.cpp' } else { 'cpu' }
$reportPath = Join-Path $root $ReportDir
New-Item -ItemType Directory -Force -Path $tmp, $reportPath | Out-Null
$date = '2026-11-04'
$prompts = @(
    'What is the capital of France? Answer in one sentence.'
    'Write a Python function that returns the n-th Fibonacci number.'
    'List three primary colors.'
    'Tell me a short story about a robot who learns to paint.'
    'What is 17 multiplied by 23? Show your steps.'
    "Translate into French: 'The weather is nice today.'"
    'Explain in two sentences why the sky is blue.'
    'Give me a haiku about autumn leaves.'
    'What are the main differences between a CPU and a GPU?'
    'Summarize the plot of Romeo and Juliet in three sentences.'
)

function Get-Ids([string]$Text, [string]$What) {
    $m = [regex]::Match($Text, '\[[0-9, ]*\]')
    if (-not $m.Success) { throw "$What printed no id list" }
    return @($m.Value.Trim('[', ']') -split ',\s*' | Where-Object { $_ -ne '' } | ForEach-Object { [int]$_ })
}

# The reference's top-2 probability gap for the token after `Ids`, from a
# teacher-forced batch evaluation: hetero-oracle-dump without --generate, or
# with -Against cpu, hearth dump --backend cpu.
function Get-TeacherForcedGap([int[]]$Ids, [int]$Ctx, [string]$OutDir) {
    $file = Join-Path $OutDir 'tokens.txt'
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    [IO.File]::WriteAllText($file, ($Ids -join "`n") + "`n")
    $common = @('--model', $modelPath, '--tokens-file', $file, '--out', $OutDir, '--names', 'result_output',
        '--positions', "$($Ids.Count - 1)", '--ctx', "$Ctx", '--threads', "$Threads")
    $r = if ($Against -eq 'llamacpp') { Invoke-NativeCapture -FilePath $oracle -ArgumentList ($common + @('--f32-kv')) -TimeoutSec 1800 }
         else { Invoke-NativeCapture -FilePath $hearth -ArgumentList (@('dump') + $common + @('--backend', 'cpu')) -TimeoutSec 1800 }
    if ($r.ExitCode -ne 0) { throw "teacher-forced reference dump failed: $(Format-ExitCode $r.ExitCode)`n$($r.Stderr)" }
    $bytes = [IO.File]::ReadAllBytes((Join-Path $OutDir 'result_output.bin'))
    $n = $bytes.Length / 4
    $max = [double]::NegativeInfinity
    for ($j = 0; $j -lt $n; ++$j) { $v = [BitConverter]::ToSingle($bytes, 4 * $j); if ($v -gt $max) { $max = $v } }
    $sum = 0.0; $p1 = 0.0; $p2 = 0.0
    for ($j = 0; $j -lt $n; ++$j) {
        $e = [math]::Exp([BitConverter]::ToSingle($bytes, 4 * $j) - $max)
        $sum += $e
        if ($e -gt $p1) { $p2 = $p1; $p1 = $e } elseif ($e -gt $p2) { $p2 = $e }
    }
    return ($p1 - $p2) / $sum
}

# log-softmax of a result_output.bin row (float32), in double.
function Read-LogSoftmax([string]$Path) {
    $bytes = [IO.File]::ReadAllBytes($Path)
    $n = $bytes.Length / 4
    $x = [double[]]::new($n)
    $max = [double]::NegativeInfinity
    for ($j = 0; $j -lt $n; ++$j) { $x[$j] = [BitConverter]::ToSingle($bytes, 4 * $j); if ($x[$j] -gt $max) { $max = $x[$j] } }
    $sum = 0.0
    for ($j = 0; $j -lt $n; ++$j) { $sum += [math]::Exp($x[$j] - $max) }
    $lse = $max + [math]::Log($sum)
    for ($j = 0; $j -lt $n; ++$j) { $x[$j] -= $lse }
    return , $x
}

# ADR 0016 / 0022: KLD(reference || backend under test) after `Ids`,
# teacher-forced; the reference's dump (cpu's, or llama.cpp's) is already in OutDir.
function Get-TeacherForcedKld([int[]]$Ids, [int]$Ctx, [string]$OutDir) {
    $gdir = Join-Path $OutDir $Backend
    $r = Invoke-NativeCapture -FilePath $hearth -ArgumentList @('dump', '--model', $modelPath, '--tokens-file',
        (Join-Path $OutDir 'tokens.txt'), '--out', $gdir, '--names', 'result_output', '--positions', "$($Ids.Count - 1)",
        '--ctx', "$Ctx", '--threads', "$Threads", '--backend', $Backend) -TimeoutSec 1800
    if ($r.ExitCode -ne 0) { throw "teacher-forced $Backend dump failed: $(Format-ExitCode $r.ExitCode)`n$($r.Stderr)" }
    if ($Backend -eq 'gpu') { $null = Get-GpuProof -Log ($r.Stdout + $r.Stderr) -Backend $Backend }
    $p = Read-LogSoftmax (Join-Path $OutDir 'result_output.bin')
    $q = Read-LogSoftmax (Join-Path $gdir 'result_output.bin')
    $kld = 0.0
    for ($j = 0; $j -lt $p.Length; ++$j) { $kld += [math]::Exp($p[$j]) * ($p[$j] - $q[$j]) }
    return $kld
}

$lines = @()
$firstProof = @()
$failed = 0
$identical = 0
$draftProposed = 0; $draftKept = 0
$which = if ($Only.Count -gt 0) { $Only } else { 0..($prompts.Count - 1) }
foreach ($ix in $which) { if ($ix -lt 0 -or $ix -ge $prompts.Count) { throw "-Only: index $ix is not in 0..$($prompts.Count - 1)" } }
foreach ($i in $which) {
    $t = Invoke-NativeCapture -FilePath $hearth -ArgumentList @('tokenize', '--model', $modelPath, '--chat', '--text',
        $prompts[$i], '--date', $date) -TimeoutSec 120
    if ($t.ExitCode -ne 0) { throw "hearth tokenize --chat failed: $(Format-ExitCode $t.ExitCode)`n$($t.Stderr)" }
    $prompt = Get-Ids $t.Stdout 'hearth tokenize'
    $ctx = $prompt.Count + $Tokens + 1
    $dir = Join-Path $tmp "prompt-$i"
    $tokFile = Join-Path $dir 'tokens.txt'
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    [IO.File]::WriteAllText($tokFile, ($prompt -join "`n") + "`n")
    if ($Against -eq 'llamacpp') {
        $o = Invoke-NativeCapture -FilePath $oracle -ArgumentList @('--model', $modelPath, '--tokens-file', $tokFile, '--out', $dir,
            '--ctx', "$ctx", '--generate', "$Tokens", '--f32-kv', '--threads', "$Threads") -TimeoutSec 3600
        if ($o.ExitCode -ne 0) { throw "hetero-oracle-dump --generate failed: $(Format-ExitCode $o.ExitCode)`n$($o.Stderr)" }
        $ref = Get-Content (Join-Path $dir 'generate.json') -Raw | ConvertFrom-Json
    } else {
        # The cpu backend's stream. Its decoding gaps are not printed; at a
        # divergence the teacher-forced gap below is the same number.
        $o = Invoke-NativeCapture -FilePath $hearth -ArgumentList @('run', '--model', $modelPath, '--tokens', ($prompt -join ','),
            '--temp', '0', '--n-predict', "$Tokens", '--ids', '--backend', 'cpu', '--ctx', "$ctx", '--threads', "$Threads") -TimeoutSec 7200
        if ($o.ExitCode -ne 0) { throw "hearth run --backend cpu failed: $(Format-ExitCode $o.ExitCode)`n$($o.Stderr)" }
        $gen = @(Get-Ids $o.Stdout 'hearth run --backend cpu')
        $ref = [pscustomobject]@{ generated = $gen; stop = $(if ($gen.Count -lt $Tokens) { 'end token' } else { 'n-predict' }); top2_gap = $null }
    }
    $h = Invoke-NativeCapture -FilePath $hearth -ArgumentList (@('run', '--model', $modelPath, '--tokens', ($prompt -join ','),
        '--temp', '0', '--n-predict', "$Tokens", '--ids', '--ctx', "$ctx", '--threads', "$Threads") + $backendArgs) -TimeoutSec 7200
    if ($h.ExitCode -ne 0) { throw "hearth run failed: $(Format-ExitCode $h.ExitCode)`n$($h.Stderr)" }
    $proof = @(Get-GpuProof -Log ($h.Stdout + $h.Stderr) -Backend $Backend)
    if ($Draft) {  # proof that the draft ran: hearth run logs what it proposed and kept
        $dm = [regex]::Match($h.Stdout + $h.Stderr, 'the draft proposed (\d+) tokens, (\d+) kept')
        if (-not $dm.Success) { throw "hearth run with --draft logged no draft statistics" }
        $draftProposed += [int]$dm.Groups[1].Value
        $draftKept += [int]$dm.Groups[2].Value
    }
    if ($firstProof.Count -eq 0) { $firstProof = $proof }
    $ours = Get-Ids $h.Stdout 'hearth run'
    $theirs = @($ref.generated)
    $k = 0
    while ($k -lt $ours.Count -and $k -lt $theirs.Count -and $ours[$k] -eq $theirs[$k]) { $k++ }
    if ($k -eq $ours.Count -and $k -eq $theirs.Count) {
        $identical++
        $line = "ok    prompt {0}: identical, {1} tokens (stop: {2})" -f $i, $theirs.Count, $ref.stop
    } else {
        # Both lists end with the end-of-generation token or at -Tokens, so a
        # difference always lies inside llama.cpp's list; anything else is a bug here.
        $prefix = if ($k -gt 0) { @($prompt + $theirs[0..($k - 1)]) } else { @($prompt) }
        if ($Against -eq 'llamacpp') {
            if ($k -ge $theirs.Count -or $k -ge @($ref.top2_gap).Count) { throw "prompt ${i}: hearth has $($ours.Count) tokens, llama.cpp $($theirs.Count), equal for $k" }
            $g = [double]$ref.top2_gap[$k]
            $ok = $g -lt $Gap
            $how = 'decoding'
            $also = ''
            if (-not $ok) {
                # ADR 0010: llama.cpp's other path, the shared prefix evaluated in one batch.
                $tf = Get-TeacherForcedGap $prefix $ctx (Join-Path $dir 'teacher-forced')
                $ok = $tf -lt $Gap
                $also = ', {0:0.0000} teacher-forced' -f $tf
                if (-not $ok -and [math]::Min($g, $tf) -lt $wideGap) {
                    # ADR 0022: inside the band, the deviation at this position must be within the floor.
                    $kld = Get-TeacherForcedKld $prefix $ctx (Join-Path $dir 'teacher-forced')
                    $ok = $kld -le $wideKld
                    $also += '; ADR 0022 band (below {0}): KLD(llama.cpp || hearth) there {1:0.000e+00}, bound {2:0e+00}' -f $wideGap, $kld, $wideKld
                }
            }
        } else {
            # Either stream may be the shorter one here (no end token in the
            # reference's printout): the gap is the reference's after the prefix.
            if ($k -ge $theirs.Count -and $k -ge $ours.Count) { throw "prompt ${i}: streams equal but not reported identical" }
            $g = Get-TeacherForcedGap $prefix $ctx (Join-Path $dir 'teacher-forced')
            $ok = $g -lt $Gap
            $how = 'teacher-forced (= its decoding)'
            $also = ''
            if (-not $ok -and $g -lt $wideGap) {
                # ADR 0016: inside the band, the deviation at this position must be within the floor.
                $kld = Get-TeacherForcedKld $prefix $ctx (Join-Path $dir 'teacher-forced')
                $ok = $kld -le $wideKld
                $also = '; ADR 0016 band (below {0}): KLD(cpu || gpu) there {1:0.000e+00}, bound {2:0e+00}' -f $wideGap, $kld, $wideKld
            }
        }
        if (-not $ok) { $failed++ }
        $line = "{0}  prompt {1}: equal for {2} tokens, then hearth {3} vs {4} {5}; {4}'s top-2 gap there {6:0.0000} {7}{8} ({9} {10})" -f `
            ($(if ($ok) { 'ok  ' } else { 'FAIL' })), $i, $k, ($(if ($k -lt $ours.Count) { $ours[$k] } else { 'end' })), $refName,
            ($(if ($k -lt $theirs.Count) { $theirs[$k] } else { 'end' })), $g, $how, $also, ($(if ($ok -and $g -lt $Gap) { 'a near-tie, below' } elseif ($ok -and $also -match 'band') { "within the $bandAdr band; plain bar" } elseif ($ok) { 'a near-tie teacher-forced, below' } else { 'not below' })), $Gap
    }
    Write-Host $line
    $lines += $line
}
$draftLine = if ($Draft) { @("draft:   $(Split-Path $Draft -Leaf): proposed $draftProposed tokens over the prompts, $draftKept kept") } else { @() }
$header = @(
    "greedy-check: $name, backend $Backend against $refName, $($which.Count) chat prompts$(if ($Only.Count -gt 0) { " (indices $($Only -join ','))" }) x $Tokens tokens, temperature 0, chat date $date"
    "date:    $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
    "commit:  $commit"
    "command: pwsh -File scripts\greedy-check.ps1 -Model $Model -Backend $Backend$(if ($Draft) { " -Draft $Draft" }) -Against $Against -Gap $Gap -Tokens $Tokens -Threads $Threads$(if ($Only.Count -gt 0) { " -Only $($Only -join ',')" })"
    "rule:    equal up to the first position where $refName's top-2 probability gap is below $Gap, decoding or teacher-forced (ACCEPTANCE, ADR 0010$(if ($kQuant) { ', ADR 0022' }))$(if ($wideGap -gt 0) { "; or below $wideGap with KLD($refName || $Backend) there <= $wideKld ($bandAdr)" })"
) + $draftLine + @($firstProof | ForEach-Object { "gpu log (prompt $($which[0])): $_" }) + @(
    ''
)
$summary = "greedy-check: $identical of $($which.Count) identical, $failed failed"
$part = if ($Only.Count -gt 0) { "-prompts-$($Only -join '_')" } else { '' }
$report = if ($Against -eq 'llamacpp') { Join-Path $reportPath "greedy-$name-$backendName$part.txt" }
          else { Join-Path $reportPath "greedy-$name-$backendName-vs-$Against$part.txt" }
[IO.File]::WriteAllText($report, (($header + $lines + @('', $summary)) -join "`n") + "`n")
Write-Host $summary
if ($failed) { throw "greedy-check: $failed prompt(s) diverged where $refName was not at a near-tie ($report)" }
if ($Draft -and $draftProposed -eq 0) { throw "greedy-check: the draft proposed no token, so speculative decoding was not tested ($report)" }
