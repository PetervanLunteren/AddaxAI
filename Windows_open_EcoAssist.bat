@REM ### Windows commands to open the EcoAssist application https://github.com/PetervanLunteren/EcoAssist
@REM ### Peter van Lunteren, 2 Apr 2023 (latest edit)

@REM set echo settings
echo off
@setlocal EnableDelayedExpansion

@REM set variables
set LOCATION_ECOASSIST_FILES=%homedrive%%homepath%\EcoAssist_files
set PATH=%PATH%;"%LOCATION_ECOASSIST_FILES%"
set CONDA_DIRECTORY=%LOCATION_ECOASSIST_FILES%\miniconda
set ECOASSISTCONDAENV=%CONDA_DIRECTORY%\envs\ecoassistcondaenv
set PIP=%ECOASSISTCONDAENV%\Scripts\pip3
set HOMEBREW_DIR=%LOCATION_ECOASSIST_FILES%\homebrew
set GIT_DIRECTORY=%LOCATION_ECOASSIST_FILES%\git4windows
set GIT_PYTHON_GIT_EXECUTABLE=%GIT_DIRECTORY%\cmd\git.exe

@REM change directory
cd "%LOCATION_ECOASSIST_FILES%" || ( echo "Could not change directory to EcoAssist_files. Command could not be run. Installation was terminated. Please send an email to contact@pvanlunteren.com for assistance. Press any key to close this window." | wtee -a "%LOG_FILE%" & PAUSE>nul & EXIT )

@REM set log file and delete the last one
set LOG_FILE=%LOCATION_ECOASSIST_FILES%\EcoAssist\logfiles\session_log.txt
if exist "%LOG_FILE%" del /F "%LOG_FILE%"

@REM log the start of the session
set START_DATE=%date% %time%
echo EcoAssist session started at %START_DATE% > "%LOG_FILE%"

@REM add path to git to PATH
set PATH="%GIT_DIRECTORY%\cmd";%PATH%

@REM activate anaconda
set PATH="%CONDA_DIRECTORY%\Scripts";%PATH%
call "%CONDA_DIRECTORY%\Scripts\activate.bat" "%CONDA_DIRECTORY%"
echo Anaconda activated >> "%LOG_FILE%"

@REM activate environment
call conda activate %ECOASSISTCONDAENV%
echo conda environment activated >> "%LOG_FILE%"
call conda info --envs >> "%LOG_FILE%"

@REM add gits to PYTHONPATH
set PYTHONPATH=%LOCATION_ECOASSIST_FILES%;%LOCATION_ECOASSIST_FILES%\cameratraps;%LOCATION_ECOASSIST_FILES%\ai4eutils;%LOCATION_ECOASSIST_FILES%\yolov5;%LOCATION_ECOASSIST_FILES%\EcoAssist;%PYTHONPATH%
echo PYTHONPATH : %PYTHONPATH% >> "%LOG_FILE%"

@REM add python.exe to beginning of PATH
set PATH=%ECOASSISTCONDAENV%;%PATH%
echo PATH : %PATH% >> "%LOG_FILE%"

@REM check python version
python -V >> "%LOG_FILE%"
where python >> "%LOG_FILE%"

@REM run script
echo Opening EcoAssist now... >> "%LOG_FILE%"
python EcoAssist\EcoAssist_GUI.py 2>&1 >> "%LOG_FILE%"

@REM timestamp the end of session
set END_DATE=%date% %time%
echo EcoAssist session ended at %END_DATE% >> "%LOG_FILE%"
