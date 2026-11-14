# Toolchain selection. Dot-source after common.ps1.
#
# Why this exists: a Windows PC often has several compilers and two different
# programs called link.exe on PATH (MSVC's linker and Git's coreutils "link").
# Every build goes through these functions, so they always use the same,
# known toolchain, whatever PATH the caller has.
#
# Overrides (environment variables):
#   HEARTH_MSVC_TOOLSET  MSVC toolset to require, e.g. 14.44 (default: the newest installed)
#   HEARTH_NINJA         full path of ninja.exe (default: ninja.exe on PATH)
#   HEARTH_LLVM_MINGW    llvm-mingw bin folder, for the sanitizer/lint builds only

$script:MsvcToolset = $env:HEARTH_MSVC_TOOLSET
$script:NinjaExe = $env:HEARTH_NINJA
$script:LlvmMingwBin = $env:HEARTH_LLVM_MINGW

function Get-NinjaExe {
    if (-not $script:NinjaExe) {
        $cmd = Get-Command ninja.exe -ErrorAction SilentlyContinue
        if (-not $cmd) { throw 'Ninja not found: put ninja.exe on PATH or set HEARTH_NINJA to its full path.' }
        $script:NinjaExe = $cmd.Source
    }
    if (-not (Test-Path $script:NinjaExe)) { throw "Ninja not found at $script:NinjaExe" }
    return $script:NinjaExe
}

function Get-LlvmMingwBin {
    if (-not $script:LlvmMingwBin) {
        $cmd = Get-Command clang++.exe -ErrorAction SilentlyContinue
        if (-not $cmd) { throw 'clang++ not found: set HEARTH_LLVM_MINGW to the llvm-mingw bin folder.' }
        $script:LlvmMingwBin = Split-Path $cmd.Source
    }
    if (-not (Test-Path (Join-Path $script:LlvmMingwBin 'clang++.exe'))) { throw "clang++ not found in $script:LlvmMingwBin" }
    return $script:LlvmMingwBin
}

function Enter-MsvcEnvironment {
    # Load the MSVC x64 developer environment into this PowerShell process.
    # The environment ends with the process, so the user's shell is not changed.
    if ($env:HEARTH_MSVC_ENTERED -eq '1') { return }
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path $vswhere)) { throw "vswhere not found at $vswhere" }
    $vs = & $vswhere -products * -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vs) { throw 'No Visual Studio installation with the MSVC x64 tools was found by vswhere.' }
    # VsDevCmd calls vswhere by name internally; without its folder on PATH it
    # prints a harmless "'vswhere.exe' is not recognized" line.
    $env:PATH = (Split-Path $vswhere) + ';' + $env:PATH
    $devShell = Join-Path $vs 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll'
    Import-Module $devShell
    $devArgs = '-arch=x64 -host_arch=x64 -no_logo'
    if ($script:MsvcToolset) { $devArgs += " -vcvars_ver=$script:MsvcToolset" }
    Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments $devArgs | Out-Null

    # Ninja goes first on PATH. Anaconda also ships a ninja, and it must not win.
    $env:PATH = (Split-Path (Get-NinjaExe)) + ';' + $env:PATH

    $cl = (Get-Command cl.exe -ErrorAction Stop).Source
    $link = (Get-Command link.exe -ErrorAction Stop).Source
    $msvcBin = Join-Path $vs "VC\Tools\MSVC\$script:MsvcToolset*\bin\Hostx64\x64"
    if ($cl -notlike "$msvcBin\cl.exe") { throw "cl.exe resolved to '$cl', expected MSVC $script:MsvcToolset under $vs" }
    if ($link -notlike "$msvcBin\link.exe") { throw "link.exe resolved to '$link' (not MSVC's linker). PATH order is wrong." }
    $env:HEARTH_MSVC_ENTERED = '1'
    $env:HEARTH_MSVC_CL = $cl
}

function Enter-ClangEnvironment {
    # llvm-mingw clang (x86_64-w64-windows-gnu target) for the sanitizer and lint
    # builds. Its bin folder goes first on PATH so the runtime DLLs it links
    # against (libc++, libunwind, ASan) are found when tests run.
    $bin = Get-LlvmMingwBin
    $env:PATH = $bin + ';' + (Split-Path (Get-NinjaExe)) + ';' + $env:PATH
}
