# One-time setup: fetch pinned dependencies and build the third-party tools
# this project needs, all inside third_party/. Nothing is installed system-wide
#. Safe to re-run.
#
#   pwsh -File scripts\setup.ps1                 # everything
#   pwsh -File scripts\setup.ps1 -Only glslang   # one item: hooks|deps|glslang|vvl|venv|oracle
[CmdletBinding()]
param(
    [ValidateSet('all', 'hooks', 'deps', 'glslang', 'vvl', 'venv', 'oracle')][string]$Only = 'all',
    # Parallel compile jobs for third-party builds. The validation layers have
    # very large generated files; 8 parallel cl.exe can exhaust 16 GB of RAM.
    [int]$Jobs = 4
)
. "$PSScriptRoot\lib\common.ps1"
. "$PSScriptRoot\lib\toolchain.ps1"

$root = Get-RepoRoot
$tp = Join-Path $root 'third_party'
$buildRoot = Join-Path $tp 'build'
$installRoot = Join-Path $tp 'install'

function Invoke-CMakeBuild {
    # Configure, build and install one third-party CMake project with MSVC + Ninja.
    param([string]$Name, [string]$Source, [string[]]$Options, [string[]]$Targets = @('install'))
    $bdir = Join-Path $buildRoot $Name
    $idir = Join-Path $installRoot $Name
    $common = @('-G', 'Ninja', '-S', $Source, '-B', $bdir,
        "-DCMAKE_MAKE_PROGRAM=$(Get-NinjaExe)",
        '-DCMAKE_BUILD_TYPE=Release',
        "-DCMAKE_INSTALL_PREFIX=$idir",
        # CMake 4 rejects projects declaring cmake_minimum_required < 3.5.
        '-DCMAKE_POLICY_VERSION_MINIMUM=3.5',
        # Hide the routine configure checks (compiler detection, "Looking for
        # pthread_create - not found" and the like); CMake's warnings and
        # errors still show.
        '--log-level=NOTICE')
    Invoke-Native cmake ($common + $Options)
    foreach ($t in $Targets) {
        Invoke-Native cmake @('--build', $bdir, '--target', $t, '--', "-j$Jobs")
    }
}

function Install-Glslang {
    $exe = Join-Path $installRoot 'glslang\bin\glslang.exe'
    if (Test-Path $exe) { Write-Host "ok (exists)  $exe"; return }
    Enter-MsvcEnvironment
    # ENABLE_OPT=OFF: no SPIRV-Tools optimizer. The Intel driver runs its own
    # compiler on the SPIR-V, and this keeps the build small and dependency-free.
    Invoke-CMakeBuild -Name 'glslang' -Source (Join-Path $tp 'glslang') -Options @(
        '-DENABLE_OPT=OFF', '-DGLSLANG_TESTS=OFF', '-DBUILD_EXTERNAL=OFF',
        '-DENABLE_GLSLANG_BINARIES=ON', '-DENABLE_HLSL=OFF', '-DBUILD_SHARED_LIBS=OFF')
    if (-not (Test-Path $exe)) { throw "glslang build finished but $exe is missing" }
    Write-Host "ok           $exe"
}

function Install-ValidationLayers {
    $json = Join-Path $installRoot 'vvl\bin\VkLayer_khronos_validation.json'
    if (Test-Path $json) { Write-Host "ok (exists)  $json"; return }
    Enter-MsvcEnvironment
    $src = Join-Path $tp 'Vulkan-ValidationLayers'
    $deps = Join-Path $buildRoot 'vvl-deps'
    # Step 1: the layer's own dependencies (SPIRV-Tools, SPIRV-Headers, ...),
    # at the commits in its scripts/known_good.json, which our pinned layer
    # commit fixes. --skip-existing-install makes re-runs cheap.
    $py = Get-Python312
    Invoke-Native $py @('-I', (Join-Path $src 'scripts\update_deps.py'), '--dir', $deps, '--config', 'release',
        '--arch', 'x64', '--generator', 'Ninja', '--skip-existing-install',
        '--cmake_var', 'CMAKE_POLICY_VERSION_MINIMUM=3.5')
    # Step 2: the layer itself, pointed at those dependencies through the
    # generated helper.cmake. (Letting CMake run step 1 itself with a custom
    # folder trips an IDE-only source_group() call in the layer's CMakeLists.)
    Invoke-CMakeBuild -Name 'vvl' -Source $src -Options @(
        '-C', (Join-Path $deps 'helper.cmake'), '-DUPDATE_DEPS=OFF', '-DBUILD_TESTS=OFF', '-DBUILD_WERROR=OFF')
    if (-not (Test-Path $json)) { throw "validation layer build finished but $json is missing" }
    Write-Host "ok           $json"
}

function Install-Oracle {
    # hetero-oracle-dump (tools/oracle): llama.cpp's forward pass with tensor
    # dumps, built from the pinned llama.cpp source (CPU backend, static).
    # Unlike the pinned tools above it is rebuilt on every run, because its
    # source lives in this repo; Ninja makes an up-to-date build a no-op.
    $exe = Join-Path $installRoot 'oracle\bin\hetero-oracle-dump.exe'
    $llama = Join-Path $tp 'llama.cpp'
    if (-not (Test-Path (Join-Path $llama 'include\llama.h'))) {
        throw "llama.cpp source not found at $llama. Run: pwsh -File scripts\setup.ps1 -Only deps"
    }
    $src = Join-Path $root 'tools\oracle'
    $bdir = Join-Path $buildRoot 'oracle'
    # CMake refuses a build folder configured from another source folder (for
    # example a git worktree of this repo), so such a folder is started afresh.
    $cache = Join-Path $bdir 'CMakeCache.txt'
    if (Test-Path $cache) {
        $configured = Select-String -LiteralPath $cache -Pattern '^CMAKE_HOME_DIRECTORY:INTERNAL=(.+)$' |
            Select-Object -First 1
        if (-not $configured -or ($configured.Matches[0].Groups[1].Value -ne ($src -replace '\\', '/'))) {
            Write-Host "clean        $bdir (configured from another source folder)"
            Remove-Item -Recurse -Force $bdir
        }
    }
    Enter-MsvcEnvironment
    # Forward slashes: CMake writes the path into generated files, where a
    # backslash starts an escape sequence.
    Invoke-CMakeBuild -Name 'oracle' -Source $src -Targets @('hetero-oracle-dump', 'hetero-oracle-chat') -Options @(
        '-DCMAKE_C_COMPILER=cl', '-DCMAKE_CXX_COMPILER=cl', "-DLLAMA_SOURCE_DIR=$($llama -replace '\\', '/')")
    # Only the executable: llama.cpp's own install rules (headers, libraries)
    # belong to other components.
    Invoke-Native cmake @('--install', $bdir, '--component', 'oracle')
    if (-not (Test-Path $exe)) { throw "oracle build finished but $exe is missing" }
    $run = Invoke-NativeCapture -FilePath $exe -ArgumentList @('--help') -TimeoutSec 60
    if ($run.ExitCode -ne 0) { throw "$exe --help failed with exit code $(Format-ExitCode $run.ExitCode)" }
    Write-Host "ok           $exe"
    # The chat tool has no --help; called without arguments it must print its usage and exit 2.
    $chat = Join-Path $installRoot 'oracle\bin\hetero-oracle-chat.exe'
    $run = Invoke-NativeCapture -FilePath $chat -ArgumentList @() -TimeoutSec 60
    if ($run.ExitCode -ne 2) { throw "$chat without arguments exited with $(Format-ExitCode $run.ExitCode), not 2" }
    Write-Host "ok           $chat"
}

function Get-Python312 {
    # HEARTH_PYTHON, else the py launcher's 3.12, else python.exe on PATH.
    if ($env:HEARTH_PYTHON) { return $env:HEARTH_PYTHON }
    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($launcher) {
        $p = (& $launcher.Source -3.12 -c 'import sys; print(sys.executable)' 2>$null)
        if ($LASTEXITCODE -eq 0 -and $p) { return $p.Trim() }
    }
    $cmd = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    throw 'Python 3.12 not found: install it, or set HEARTH_PYTHON to python.exe.'
}

function Install-Venv {
    $py = Get-Python312
    $venv = Join-Path $root '.venv'
    $vpy = Join-Path $venv 'Scripts\python.exe'
    if (-not (Test-Path $vpy)) {
        if (-not (Test-Path $py)) { throw "Python 3.12 not found at $py" }
        Invoke-Native $py @('-m', 'venv', $venv)
    }
    # --require-hashes: every wheel must match a hash in requirements.txt.
    Invoke-Native $vpy @('-m', 'pip', 'install', '--quiet', '--disable-pip-version-check',
        '--require-hashes', '--only-binary=:all:', '-r', (Join-Path $root 'requirements.txt'))
    Write-Host "ok           .venv ready"
}

function Install-Hooks {
    # Versioned hooks in scripts/hooks; this is a repo-local git setting.
    Invoke-Native git @('-C', $root, 'config', 'core.hooksPath', 'scripts/hooks')
    Write-Host 'ok           git hooks: scripts/hooks (pre-commit: format check + quick tests)'
}

switch ($Only) {
    'hooks' { Install-Hooks }
    'deps' { & "$PSScriptRoot\fetch-deps.ps1" }
    'glslang' { Install-Glslang }
    'vvl' { Install-ValidationLayers }
    'venv' { Install-Venv }
    'oracle' { Install-Oracle }
    'all' {
        Install-Hooks
        & "$PSScriptRoot\fetch-deps.ps1"
        Install-Venv
        Install-Glslang
        Install-ValidationLayers
        Install-Oracle
        Write-Host 'setup: complete.'
    }
}
