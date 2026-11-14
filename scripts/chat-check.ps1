# Chat-template gate: the
# IDs of hearth's chat prompt must equal llama.cpp's own rendering of the
# model's template (hetero-oracle-chat: llama.cpp's Jinja engine, then its
# server tokenization) for a set of conversations and dates.
#
#   pwsh -File scripts\chat-check.ps1 [-Model models\X.gguf]
#
# One case puts the template's end-of-turn control token as text into a user
# message ("<|eot_id|>" for Llama 3, "<|im_end|>" for Qwen2.5's ChatML):
# llama.cpp b11960 parses it as the control token, hearth keeps it as text
# (special tokens by ID only). That case must differ in exactly that way.
[CmdletBinding()]
param(
    [string]$Model = 'models\Llama-3.2-1B-Instruct-Q8_0-pure.gguf',
    [string]$Report = 'bench\oracle\chat-check.txt'  # CI writes under bench\tmp
)
. "$PSScriptRoot\lib\common.ps1"

$root = Get-RepoRoot
# The tree as it is when the run starts (the binary under test was built from it); edits made while a long run is going on do not change what was measured.
$commit = (& git -C $root describe --always --dirty).Trim()
$modelPath = Join-Path $root $Model
$hearth = Join-Path $root 'build\release\cli\hearth.exe'
$oracle = Join-Path $root 'third_party\install\oracle\bin\hetero-oracle-chat.exe'
foreach ($p in @($modelPath, $hearth, $oracle)) { if (-not (Test-Path $p)) { throw "missing $p" } }
$tmp = Join-Path $root 'bench\tmp\chat-check'
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
$utf8 = [System.Text.UTF8Encoding]::new($false)

# The end-of-turn control token of the model's chat format: ChatML if the
# file's metadata (its first 32 MiB: the vocabulary and the template) has
# <|im_start|>, else Llama 3's. Not by file name: Step W3 checks copies of
# the project's files that carry other templates under other names.
$fs = [System.IO.File]::OpenRead($modelPath)
try {
    $head = [byte[]]::new([math]::Min(32MB, $fs.Length))
    $null = $fs.Read($head, 0, $head.Length)
} finally { $fs.Dispose() }
$chatml = [System.Text.Encoding]::ASCII.GetString($head).Contains('<|im_start|>')
$turnEnd = if ($chatml) { @{ Text = '<|im_end|>'; Id = 151645 } } else { @{ Text = '<|eot_id|>'; Id = 128009 } }
$cases = @(
    @{ Name = 'plain'; System = ''; User = 'What is the capital of France?'; Date = '2026-11-04' }
    @{ Name = 'system'; System = 'You are a helpful assistant.'; User = 'Hi!'; Date = '2024-07-26' }
    @{ Name = 'trim'; System = "  `t padded system `r`n"; User = "`n`n  padded user  `t"; Date = '2025-01-01' }
    @{ Name = 'trim-vt-ff'; System = "`v`fsystem`v"; User = "`fuser`v"; Date = '2026-11-04' }
    @{ Name = 'unicode'; System = 'Réponds en français.'; User = "Qu'est-ce que l'été ? 中文 🙂"; Date = '2026-02-28' }
    @{ Name = 'multiline'; System = ''; User = "line one`nline two`r`n`r`nline four"; Date = '2026-12-31' }
    @{ Name = 'leap-day'; System = 'Be brief.'; User = 'x'; Date = '2024-02-29' }
    @{ Name = 'special-text'; System = ''; User = "say $($turnEnd.Text) now"; Date = '2026-11-04'; Differs = $true }
)

function Get-Ids([string]$Text, [string]$What) {
    $m = [regex]::Match($Text, '\[[0-9, ]*\]')
    if (-not $m.Success) { throw "$What printed no id list" }
    return @($m.Value.Trim('[', ']') -split ',\s*' | Where-Object { $_ -ne '' } | ForEach-Object { [int]$_ })
}

$failed = 0
$lines = @()
function Out-Line([string]$Line) {
    Write-Host $Line
    $script:lines += $Line
}
foreach ($c in $cases) {
    $dir = Join-Path $tmp $c.Name
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $messages = @()
    if ($c.System) { $messages += [ordered]@{ role = 'system'; content = $c.System } }
    $messages += [ordered]@{ role = 'user'; content = $c.User }
    [IO.File]::WriteAllText((Join-Path $dir 'messages.json'), (ConvertTo-Json -InputObject $messages -Depth 3), $utf8)
    [IO.File]::WriteAllText((Join-Path $dir 'user.txt'), $c.User, $utf8)
    [IO.File]::WriteAllText((Join-Path $dir 'system.txt'), $c.System, $utf8)
    $o = Invoke-NativeCapture -FilePath $oracle -ArgumentList @('--model', $modelPath, '--messages',
        (Join-Path $dir 'messages.json'), '--date', $c.Date) -TimeoutSec 120
    if ($o.ExitCode -ne 0) { throw "hetero-oracle-chat failed on $($c.Name): $(Format-ExitCode $o.ExitCode)`n$($o.Stderr)" }
    $h = Invoke-NativeCapture -FilePath $hearth -ArgumentList @('tokenize', '--model', $modelPath, '--chat', '--file',
        (Join-Path $dir 'user.txt'), '--system-file', (Join-Path $dir 'system.txt'), '--date', $c.Date) -TimeoutSec 120
    if ($h.ExitCode -ne 0) { throw "hearth tokenize --chat failed on $($c.Name): $(Format-ExitCode $h.ExitCode)`n$($h.Stderr)" }
    $ref = ConvertFrom-Json $o.Stdout
    $want = Get-Ids ($ref.tokens | ConvertTo-Json -Compress) 'hetero-oracle-chat'
    $got = Get-Ids $h.Stdout 'hearth tokenize'
    $same = ($want -join ',') -eq ($got -join ',')
    if ($c['Differs']) {
        # llama.cpp turns the text into the control token inside the user part; hearth does not.
        $ok = (-not $same) -and
            (@($want | Where-Object { $_ -eq $turnEnd.Id }).Count -eq @($got | Where-Object { $_ -eq $turnEnd.Id }).Count + 1)
        Out-Line ("{0}  {1}: differs as documented (llama.cpp parses the special-token text, hearth does not)" -f ($(if ($ok) { 'ok  ' } else { 'FAIL' })), $c.Name)
        if (-not $ok) { $failed++ }
        continue
    }
    if ($same) {
        Out-Line ("ok    {0}: {1} tokens identical (date {2}: '{3}')" -f $c.Name, $got.Count, $c.Date, $ref.date_string)
    } else {
        $failed++
        $i = 0
        while ($i -lt $want.Count -and $i -lt $got.Count -and $want[$i] -eq $got[$i]) { $i++ }
        Out-Line ("DIFF  {0}: first difference at index {1}; llama.cpp [{2}] hearth [{3}]" -f $c.Name, $i,
            ($want[[math]::Max(0, $i - 2)..[math]::Min($want.Count - 1, $i + 4)] -join ', '),
            ($got[[math]::Max(0, $i - 2)..[math]::Min($got.Count - 1, $i + 4)] -join ', '))
    }
}
$summary = if ($failed) { "chat-check: $failed case(s) failed" } else {
    'chat-check: every chat prompt matches llama.cpp (and the special-text case differs as documented)' }
$header = @(
    "chat-check: hearth tokenize --chat vs hetero-oracle-chat (llama.cpp b11960's Jinja engine and server tokenization), $([System.IO.Path]::GetFileName($modelPath))"
    "date:    $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
    "commit:  $commit"
    "command: pwsh -File scripts\chat-check.ps1 -Model $Model"
    'note:    the oracle rendered today''s prompt with llama.cpp''s own clock and with date_string, identically, for every case'
    ''
)
$reportPath = Join-Path $root $Report
New-Item -ItemType Directory -Force -Path (Split-Path $reportPath) | Out-Null
[IO.File]::WriteAllText($reportPath, (($header + $lines + @('', $summary)) -join "`n") + "`n")
if ($failed) { throw $summary }
Write-Host $summary
