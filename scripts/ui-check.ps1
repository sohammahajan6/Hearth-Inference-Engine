# `hearth ui` end to end: starts the
# real app without a browser, drives it over HTTP as the page does, and quits
# it. Fails at the first check that does not pass.
#
#   pwsh -File scripts\ui-check.ps1 [-Model models\X.gguf] [-NoBuild] [-Report F]
#
# Checks: the page needs the token; the API refuses a foreign Host, a foreign
# Origin and a missing token; the model loads on the CPU and a greedy chat
# reply names Paris and reports its timings; with a GPU, the same on the GPU;
# a benchmark of the CPU and then the GPU (or the CPU alone without a GPU)
# reports both speeds, or, on battery, is refused with the charger message;
# the result is in the history; Quit ends the process with exit code 0. About
# a minute on the 1B Q8_0.
[CmdletBinding()]
param(
    [string]$Model = 'models\Llama-3.2-1B-Instruct-Q8_0-pure.gguf',
    [switch]$NoBuild,
    [string]$Report = ''
)
. "$PSScriptRoot\lib\common.ps1"
. "$PSScriptRoot\lib\toolchain.ps1"
# One number format whatever the PC's region settings.
[System.Threading.Thread]::CurrentThread.CurrentCulture = [cultureinfo]::InvariantCulture

$root = Get-RepoRoot
$modelPath = Join-Path $root $Model
if (-not (Test-Path $modelPath)) { throw "ui-check: missing $modelPath (run scripts\fetch-models.ps1)" }
Assert-GpuBudget 'the ui check (the app uses the GPU)'
if (-not $NoBuild) { & "$PSScriptRoot\build.ps1" -Preset release -Target @('hearth') }
$exe = Join-Path $root 'build\release\cli\hearth.exe'
if (-not (Test-Path $exe)) { throw "ui-check: $exe not found" }

$work = Join-Path $root 'build\ui-check'
if (Test-Path $work) { Remove-Item -Recurse -Force $work }
New-Item -ItemType Directory -Force -Path $work | Out-Null
$urlFile = Join-Path $work 'url.txt'
$results = Join-Path $work 'results.jsonl'
$lines = [System.Collections.Generic.List[string]]::new()
function Check([string]$What, [bool]$Ok, [string]$Detail = '') {
    $line = ('{0} {1}{2}' -f $(if ($Ok) { 'ok  ' } else { 'FAIL' }), $What, $(if ($Detail) { ": $Detail" } else { '' }))
    $lines.Add($line)
    Write-Host "ui-check: $line"
    if (-not $Ok) { throw "ui-check: check failed: $What" }
}

# A raw request over TCP, so the Host and Origin headers are exactly what is
# written (HTTP clients set Host themselves). Returns status and body.
function Send-Raw([int]$Port, [string]$Text) {
    $c = [System.Net.Sockets.TcpClient]::new('127.0.0.1', $Port)
    try {
        $s = $c.GetStream()
        $s.ReadTimeout = 600000
        $b = [Text.Encoding]::UTF8.GetBytes($Text)
        $s.Write($b, 0, $b.Length)
        $ms = [IO.MemoryStream]::new()
        $s.CopyTo($ms)
        $all = [Text.Encoding]::UTF8.GetString($ms.ToArray())
    } finally { $c.Dispose() }
    $code = if ($all -match '^HTTP/1\.1 (\d{3})') { [int]$Matches[1] } else { 0 }
    $i = $all.IndexOf("`r`n`r`n")
    [pscustomobject]@{ Status = $code; Body = $(if ($i -ge 0) { $all.Substring($i + 4) } else { '' }) }
}
function Request([string]$Method, [string]$Path, [hashtable]$Form = $null, [hashtable]$Extra = @{}) {
    $h = [ordered]@{ Host = "127.0.0.1:$script:port"; 'X-Hearth-Token' = $script:token }
    foreach ($k in $Extra.Keys) { $h[$k] = $Extra[$k] }
    $body = ''
    if ($Form) {
        $body = ($Form.GetEnumerator() | ForEach-Object { [uri]::EscapeDataString($_.Key) + '=' + [uri]::EscapeDataString([string]$_.Value) }) -join '&'
        $h['Content-Type'] = 'application/x-www-form-urlencoded'
    }
    $h['Content-Length'] = [Text.Encoding]::UTF8.GetByteCount($body)
    $head = "$Method $Path HTTP/1.1`r`n" + (($h.GetEnumerator() | ForEach-Object { "$($_.Key): $($_.Value)" }) -join "`r`n") + "`r`n`r`n"
    Send-Raw $script:port ($head + $body)
}
# The events of a stream: name and parsed data.
function Get-Events([string]$Body) {
    @($Body -split "`n`n" | Where-Object { $_ -match '^event: (\S+)\ndata: (.*)$' } | ForEach-Object {
            [pscustomobject]@{ Name = $Matches[1]; Data = ($Matches[2] | ConvertFrom-Json) } })
}

$out = Join-Path $work 'stdout.txt'
$err = Join-Path $work 'stderr.txt'
$in = Join-Path $work 'stdin.txt'   # empty: the app never reads input
[IO.File]::WriteAllText($in, '')
$p = Start-Process -FilePath $exe -ArgumentList @('ui', '--no-browser', '--idle-minutes', '0', '--url-file', "`"$urlFile`"",
    '--results', "`"$results`"") -RedirectStandardOutput $out -RedirectStandardError $err -RedirectStandardInput $in -NoNewWindow -PassThru
try {
    for ($i = 0; $i -lt 300 -and -not (Test-Path $urlFile); $i++) { Start-Sleep -Milliseconds 100 }
    Check 'the app starts and listens' (Test-Path $urlFile)
    $url = (Get-Content $urlFile -Raw).Trim()
    if ($url -notmatch '^http://127\.0\.0\.1:(\d+)/\?t=([0-9a-f]{32})$') { throw "ui-check: unexpected address $url" }
    $script:port = [int]$Matches[1]
    $script:token = $Matches[2]
    Check 'it listens on 127.0.0.1 with a 128-bit key' $true "port $script:port"

    $r = Send-Raw $script:port "GET /?t=$script:token HTTP/1.1`r`nHost: 127.0.0.1:$script:port`r`n`r`n"
    Check 'the page is served with the key' ($r.Status -eq 200 -and $r.Body.Contains('<title>Hearth</title>')) "$($r.Body.Length) bytes"
    $r = Send-Raw $script:port "GET / HTTP/1.1`r`nHost: 127.0.0.1:$script:port`r`n`r`n"
    Check 'the page is refused without the key' ($r.Status -eq 403)
    $r = Send-Raw $script:port "GET /api/status HTTP/1.1`r`nHost: evil.example`r`nX-Hearth-Token: $script:token`r`n`r`n"
    Check 'a foreign Host is refused (DNS rebinding)' ($r.Status -eq 403)
    $r = Request 'POST' '/api/stop' @{} @{ Origin = 'http://evil.example' }
    Check 'a foreign Origin is refused (cross-site request)' ($r.Status -eq 403)
    $r = Send-Raw $script:port "GET /api/status HTTP/1.1`r`nHost: 127.0.0.1:$script:port`r`n`r`n"
    Check 'the API is refused without the key' ($r.Status -eq 403)

    $r = Request 'GET' '/api/status'
    $st = $r.Body | ConvertFrom-Json
    $found = @($st.models | Where-Object { $_.name -eq (Split-Path $modelPath -Leaf) })
    Check 'status lists the model' ($r.Status -eq 200 -and $found.Count -eq 1) "$($st.models.Count) model file(s); power: $($st.machine.power); GPU: $($st.machine.gpus -join ', ')"
    # -cin/-ceq: the page compares case-sensitively, and PowerShell's -eq does
    # not, so "AC" (which made the page warn "on battery" while plugged in)
    # once passed this script.
    Check 'status names the power source as the page reads it' ($st.machine.power -cin @('ac', 'battery', 'unknown')) "'$($st.machine.power)'"

    $haveGpu = @($st.machine.gpus).Count -gt 0
    function Test-Chat([string]$Mode) {
        $r = Request 'POST' '/api/load' @{ model = $found[0].path; mode = $Mode }
        $ld = ($r.Body | ConvertFrom-Json).loaded
        Check "the model loads on the $Mode" ($r.Status -eq 200 -and $null -ne $ld -and $ld.mode -ceq $Mode) "$($ld.label) [$($ld.backend)]"
        $r = Request 'POST' '/api/chat' @{ message = 'What is the capital of France? Answer in one word.'; temperature = '0'; max_tokens = '16' }
        $ev = Get-Events $r.Body
        $text = -join @($ev | Where-Object Name -eq 'token' | ForEach-Object { $_.Data.text })
        $done = @($ev | Where-Object Name -eq 'done')
        Check "a chat reply on the $Mode streams and names Paris" ($text -match 'Paris' -and $done.Count -eq 1 -and $done[0].Data.mode -ceq $Mode) ("'{0}'; read {1} tokens in {2:N3} s, wrote {3} in {4:N3} s, {5:N3} s in all" -f $text.Trim(), $done[0].Data.prompt_tokens, $done[0].Data.prefill_s, $done[0].Data.generated, $done[0].Data.decode_s, $done[0].Data.total_s)
    }
    if ($haveGpu) { Test-Chat 'gpu' }
    Test-Chat 'cpu'

    $which = if ($haveGpu) { 'both' } else { 'current' }
    $r = Request 'POST' '/api/bench' @{ backends = $which; prompt_tokens = '128'; gen_tokens = '32'; reps = '1' }
    $ev = Get-Events $r.Body
    $res = @($ev | Where-Object Name -eq 'result')
    $bad = @($ev | Where-Object Name -eq 'error')
    if ($st.machine.power -ceq 'ac') {
        Check "a benchmark ($which) runs on AC power" ($res.Count -eq 1) $(if ($bad.Count) { $bad[0].Data.message } else { '' })
        $d = $res[0].Data
        $want = if ($haveGpu) { 'cpu,gpu' } else { 'cpu' }
        $got = (@($d.runs) | ForEach-Object { $_.backend }) -join ','
        $positive = @(@($d.runs) | Where-Object { $_.pp_tps -gt 0 -and $_.tg_tps -gt 0 -and $_.decode_gbps -gt 0 }).Count -eq @($d.runs).Count
        Check 'the benchmark measured each engine' ($got -ceq $want -and $positive) ((@($d.runs) | ForEach-Object { '{0}: pp128 {1:N1} tok/s, tg32 {2:N1} tok/s, {3:N1} GB/s' -f $_.backend, $_.pp_tps, $_.tg_tps, $_.decode_gbps }) -join '; ')
        $st2 = (Request 'GET' '/api/status').Body | ConvertFrom-Json
        Check 'the model is loaded as before the benchmark' ($st2.loaded.mode -ceq 'cpu' -and $st2.loaded.context_used -eq 0) "$($st2.loaded.label)"
        $h = (Request 'GET' '/api/history').Body | ConvertFrom-Json
        Check 'the result is saved in the history' (@($h).Count -eq 1)
    } else {
        Check 'on battery, the benchmark is refused with the charger message' ($bad.Count -eq 1 -and $bad[0].Data.message -match 'charger') $bad[0].Data.message
    }

    $r = Request 'POST' '/api/quit' @{}
    Check 'Quit is accepted' ($r.Status -eq 200)
    $exited = $p.WaitForExit(30000)
    Check 'the process ends with exit code 0' ($exited -and $p.ExitCode -eq 0) $(if ($exited) { "exit code $($p.ExitCode)" } else { 'still running after 30 s' })
    $warn = @(Get-Content $err | Where-Object { $_ -match '^(warning|error): ' })
    Check 'no warning or error was logged' ($warn.Count -eq 0) ($warn -join ' | ')
} finally {
    if (-not $p.HasExited) { $p.Kill(); $p.WaitForExit(10000) | Out-Null }
    if ($Report) {
        $commit = (& git -C $root describe --always --dirty --tags).Trim()
        $file = Join-Path $root $Report
        New-Item -ItemType Directory -Force -Path (Split-Path $file) | Out-Null
        $hdr = @("ui-check: hearth ui end to end", "date:    $(Get-Date -Format 'yyyy-MM-dd HH:mm')", "commit:  $commit",
            "command: pwsh -File scripts\ui-check.ps1 -Model $Model$(if ($NoBuild) { ' -NoBuild' }) -Report $Report", '')
        [IO.File]::WriteAllText($file, (($hdr + $lines) -join "`n") + "`n")
    }
}
Write-Host 'ui-check: all checks passed'
