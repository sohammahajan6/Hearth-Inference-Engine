@echo off
rem Starts the Hearth app (`hearth ui`) from
rem the Release build. Double-click this file. Build first with:
rem   pwsh -File scripts\build.ps1
setlocal
set "EXE=%~dp0build\release\cli\hearth.exe"
if not exist "%EXE%" (
  echo Hearth is not built yet.
  echo Open PowerShell in this folder and run:  pwsh -File scripts\build.ps1
  pause
  exit /b 1
)
"%EXE%" ui %*
if errorlevel 1 pause
