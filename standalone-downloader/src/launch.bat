@echo off
rem Launcher for the portable (ZIP) build.
rem ASCII only on purpose: a .bat file mangles Hebrew in most code pages.
rem -ExecutionPolicy Bypass applies to THIS launch only; nothing on the
rem machine is reconfigured.
start "" powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%~dp0src\Downloader.ps1"
