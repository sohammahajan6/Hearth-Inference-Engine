# What the git pre-commit hook runs: formatting check + quick tests.
# Kept fast (incremental build + about 2 s of tests) so it is never skipped.
. "$PSScriptRoot\lib\common.ps1"
& "$PSScriptRoot\format.ps1" -Check
& "$PSScriptRoot\test.ps1" -Tier quick -Preset release
Write-Host 'pre-commit: ok'
