# Tokenizer gate: hearth's
# token IDs must equal llama-tokenize's (pinned b11960 CPU zip, file read
# verbatim, special-token text not parsed) on the wikitext-2 files and on the
# adversarial corpus.
#
#   pwsh -File scripts\tokenizer-check.ps1 [-Model models\X.gguf] [-Files a,b] [-Report FILE]
#
# With the default file set, the report goes to bench\oracle\tokenizer-check.txt.
[CmdletBinding()]
param(
    [string]$Model = 'models\Llama-3.2-1B-Instruct-Q8_0-pure.gguf',
    [string[]]$Files = @('tests\fixtures\tokenizer\adversarial.txt', 'data\wikitext-2-raw\wiki.test.raw',
        'data\wikitext-2-raw\wiki.valid.raw', 'data\wikitext-2-raw\wiki.train.raw'),
    [string]$Report = ''  # default: bench\oracle\tokenizer-check.txt for the full set, none otherwise
)
. "$PSScriptRoot\lib\common.ps1"

$root = Get-RepoRoot
# The tree as it is when the run starts (the binary under test was built from it).
$commit = (& git -C $root describe --always --dirty).Trim()
$modelPath = Join-Path $root $Model
$hearth = Join-Path $root 'build\release\cli\hearth.exe'
$llama = Join-Path $root 'third_party\llama-bin-cpu\llama-tokenize.exe'
foreach ($p in @($modelPath, $hearth, $llama)) { if (-not (Test-Path $p)) { throw "missing $p" } }

function Get-IdList([string]$Output, [string]$What) {
    $m = [regex]::Match($Output, '\[[0-9, ]*\]')
    if (-not $m.Success) { throw "$What printed no id list" }
    return $m.Value
}

$failed = 0
$lines = @()
$total = 0
foreach ($rel in $Files) {
    $file = Join-Path $root $rel
    if (-not (Test-Path $file)) { throw "missing $file" }
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $h = Invoke-NativeCapture -FilePath $hearth -ArgumentList @('tokenize', '--model', $modelPath, '--file', $file) -TimeoutSec 600
    $tHearth = $sw.Elapsed.TotalSeconds
    if ($h.ExitCode -ne 0) { throw "hearth tokenize failed on ${rel}: $(Format-ExitCode $h.ExitCode)`n$($h.Stderr)" }
    $sw.Restart()
    $l = Invoke-NativeCapture -FilePath $llama -ArgumentList @('-m', $modelPath, '-f', $file, '--ids', '--log-disable',
        '--no-escape', '--no-parse-special') -TimeoutSec 600
    $tLlama = $sw.Elapsed.TotalSeconds
    if ($l.ExitCode -ne 0) { throw "llama-tokenize failed on ${rel}: $(Format-ExitCode $l.ExitCode)" }
    $a = Get-IdList $h.Stdout 'hearth tokenize'
    $b = Get-IdList $l.Stdout 'llama-tokenize'
    $count = ($b.Length -gt 2) ? ($b.Split(',').Count) : 0
    $bytes = (Get-Item $file).Length
    if ($a -eq $b) {
        $total += $count
        $line = "ok    {0}: {1} tokens identical ({2} bytes; hearth {3:N1} s, llama-tokenize {4:N1} s, each including the model load)" -f $rel, $count, $bytes, $tHearth, $tLlama
        Write-Host $line
        $lines += $line
        continue
    }
    $failed++
    $x = $a.Trim('[', ']').Split(', ')
    $y = $b.Trim('[', ']').Split(', ')
    $i = 0
    while ($i -lt $x.Count -and $i -lt $y.Count -and $x[$i] -eq $y[$i]) { $i++ }
    $lo = [math]::Max(0, $i - 3)
    $diff = @(
        ("DIFF  {0}: {1} vs {2} tokens; first difference at index {3}" -f $rel, $x.Count, $y.Count, $i)
        ("      hearth:         ... {0}" -f ($x[$lo..([math]::Min($x.Count - 1, $i + 5))] -join ', '))
        ("      llama-tokenize: ... {0}" -f ($y[$lo..([math]::Min($y.Count - 1, $i + 5))] -join ', ')))
    $diff | ForEach-Object { Write-Host $_ }
    $lines += $diff
}
$summary = if ($failed) { "tokenizer-check: $failed file(s) differ from llama-tokenize" } else {
    "tokenizer-check: every file tokenizes exactly as llama-tokenize does ($total tokens, with BOS)" }
$fullSet = -not $PSBoundParameters.ContainsKey('Files')
if (-not $Report -and $fullSet) { $Report = 'bench\oracle\tokenizer-check.txt' }
if ($Report) {
    $header = @(
        "tokenizer-check: hearth tokenize vs llama-tokenize (b11960 CPU zip, --ids --no-escape --no-parse-special), $([System.IO.Path]::GetFileName($modelPath))"
        "date:    $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
        "commit:  $commit"
        "command: pwsh -File scripts\tokenizer-check.ps1 -Model $Model$(if (-not $fullSet) { ' -Files ' + ($Files -join ',') })"
        ''
    )
    $path = Join-Path $root $Report
    New-Item -ItemType Directory -Force -Path (Split-Path $path) | Out-Null
    [IO.File]::WriteAllText($path, (($header + $lines + @('', $summary)) -join "`n") + "`n")
}
if ($failed) { throw $summary }
Write-Host $summary
