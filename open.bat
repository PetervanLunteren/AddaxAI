@REM This script opens the AddaxAI program and allows for shortcut creation and execution from within Timelapse.
@REM Timelapse can also open it by executing the python command directly, but it is still here so that it mimics 
@REM the old method. In the old method we needed this batch script to perform some conda operations before opening
@REM the python script. With the Jan 2025 install update that is not necessary anymore. But this script is still 
@REM here so that Timelapse can run the same command as before and users don't have to update their Timelapse 
@REM version when using a new AddaxAI version. 
@REM Peter van Lunteren, 7 Mar 2025 (latest edit)

@REM set echo settings
echo off
@setlocal EnableDelayedExpansion

@REM set paths
for %%i in ("%~dp0..") do set "ADDAXAI_FILES_DIR=%%~fi"
set ADDAXAI_GUI="%ADDAXAI_FILES_DIR%\AddaxAI\AddaxAI_GUI.py"
set PYTHON_EXE="%ADDAXAI_FILES_DIR%\envs\env-base\python.exe"

@REM run script in either timelapse or normal mode
if "%1" == "timelapse" (
    %PYTHON_EXE% %ADDAXAI_GUI% --timelapse-path=%2
) else (
    %PYTHON_EXE% %ADDAXAI_GUI%
)

@REM keep terminal window open
echo.
echo In case of an error - copy-paste the above text and send it to the developer.
echo.
pause
