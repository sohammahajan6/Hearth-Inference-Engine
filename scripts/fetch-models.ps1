# Download the pinned models and evaluation text, then derive the exact
# model files the tests and benchmarks use. Safe to re-run.
#
#   pwsh -File scripts\fetch-models.ps1            # 1B model set + wikitext-2
#   pwsh -File scripts\fetch-models.ps1 -Set 3b    # 3B model set
#
# Derived files are made with llama-quantize from the pinned llama.cpp release,
# so the engine and llama.cpp are always compared on the same bytes. Most use
# --pure: every 2-D weight gets the requested type (no mixed "recipes"); the
# K-quant file is the Q4_K_M recipe itself.
[CmdletBinding()]
param(
    [ValidateSet('1b', '3b', 'qwen0.5b')][string]$Set = '1b',
    # The 3B BF16 and Qwen F16 sources are deleted after deriving unless this is given.
    [switch]$KeepSource
)
. "$PSScriptRoot\lib\common.ps1"

$root = Get-RepoRoot
$pins = Get-Pins
$models = Join-Path $root 'models'
$data = Join-Path $root 'data'
$quantize = Join-Path $root 'third_party\llama-bin-cpu\llama-quantize.exe'
if (-not (Test-Path $quantize)) { throw "llama-quantize not found at $quantize. Run scripts\fetch-deps.ps1 first." }

# Hugging Face cache stays on E: (C: holds the pagefile and TEMP).
$env:HF_HOME = Join-Path $root 'third_party\hf-home'

# Derived files per set. Their sha256 values are recorded in models/MODELS.md.
# A type with a recipe suffix (Q4_K_M) is llama-quantize's own mixture of
# types, made without --pure and named without "-pure": that is what K-quant
# files look like in practice (Q4_K_M: Q4_K, with Q6_K for token_embd and half
# of attn_v and ffn_down; docs/architecture/14-k-quants.md).
$derive = @{
    '1b' = @('F32', 'Q8_0', 'Q4_0', 'Q4_K_M')
    '3b' = @('Q8_0', 'Q4_0')
    'qwen0.5b' = @('F32', 'Q8_0', 'Q4_0')
}

$m = $pins.Models | Where-Object { $_.Set -eq $Set }
# A pin may name the file differently on disk (Local), e.g. in the project's naming.
$local = if ($m.ContainsKey('Local')) { $m.Local } else { $m.File }
$src = Join-Path $models $local
$needGB = switch ($Set) { '1b' { 2.4 + 7.1 + 0.8 } '3b' { 6.0 + 5.2 } default { 1.3 + 3.5 } }
if (-not (Test-Path $src)) { Assert-FreeSpace -NeedGB $needGB }
Get-VerifiedFile -Url "https://huggingface.co/$($m.Repo)/resolve/$($m.Revision)/$($m.File)" `
    -Destination $src -Sha256 $m.Sha256

$base = $local -replace '-(BF16|F16)\.gguf$', ''
foreach ($type in $derive[$Set]) {
    $pure = $type -cnotmatch '_K_[SML]$'
    $name = if ($pure) { "$base-$type-pure.gguf" } else { "$base-$type.gguf" }
    $out = Join-Path $models $name
    $expected = $pins.Derived[$name]
    if (-not (Test-Path $out)) {
        if (-not (Test-Path $src)) {
            throw "$out is missing and its source $src was deleted. Re-run with the source present (it will be downloaded again)."
        }
        $part = "$out.part"
        Write-Host "derive       $type -> $out"
        $qargs = @($src, $part, $type, '4')
        if ($pure) { $qargs = @('--pure') + $qargs }
        Invoke-Native $quantize $qargs
        Move-Item -Force $part $out
    }
    $actual = Get-Sha256 $out
    if (-not $expected) {
        Write-Warning "no pinned sha256 for $name yet: $actual (add it to Derived in scripts\pins.psd1)"
    } elseif ($actual -ne $expected) {
        throw "derived file $name has sha256 $actual, pinned $expected. The quantizer output changed; investigate before using it."
    }
    Write-Host ("ok           {0}  sha256={1}" -f $out, $actual)
}

if ($Set -ne '1b' -and -not $KeepSource -and (Test-Path $src)) {
    Remove-Item -LiteralPath $src
    Write-Host "removed      $src (derived files kept; pass -KeepSource to keep it)"
}

# wikitext-2 (raw) for perplexity.
$w = $pins.Wikitext
$zip = Join-Path $root "third_party\downloads\$($w.File)"
Get-VerifiedFile -Url "https://huggingface.co/datasets/$($w.Repo)/resolve/$($w.Revision)/$($w.File)" `
    -Destination $zip -Sha256 $w.Sha256
$wdir = Join-Path $data 'wikitext-2-raw'
if (-not (Test-Path (Join-Path $wdir 'wiki.test.raw'))) {
    New-Item -ItemType Directory -Force -Path $data | Out-Null
    Invoke-Native tar.exe @('-xf', $zip, '-C', $data)
}
Write-Host ("ok           {0}  sha256={1}" -f (Join-Path $wdir 'wiki.test.raw'), (Get-Sha256 (Join-Path $wdir 'wiki.test.raw')))
Write-Host "fetch-models: set '$Set' ready."
