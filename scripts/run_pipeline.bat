@echo off
setlocal

set "SCRIPTS_DIR=%~dp0"
set "REPO_ROOT=%SCRIPTS_DIR%..\"
set "VENV_PY=%REPO_ROOT%.venv\Scripts\python.exe"

if exist "%VENV_PY%" (
    set "PY_CMD=%VENV_PY%"
) else (
    set "PY_CMD=python"
)

if "%~1"=="" goto usage
if /I "%~1"=="help" goto usage
if /I "%~1"=="--help" goto usage
if /I "%~1"=="-h" goto usage

if /I "%~1"=="run" goto run
if /I "%~1"=="train" goto train
if /I "%~1"=="evaluate" goto evaluate
if /I "%~1"=="predict" goto predict
if /I "%~1"=="clean" goto clean

if /I "%~1"=="pipeline-run" goto pipeline_run
if /I "%~1"=="pipeline-train" goto pipeline_train
if /I "%~1"=="pipeline-eval" goto pipeline_eval
if /I "%~1"=="pipeline-predict" goto pipeline_predict

echo Unknown command: %~1
exit /b 1

:run
"%PY_CMD%" "%SCRIPTS_DIR%run_pipeline.py" run %2 %3 %4 %5 %6 %7 %8 %9
exit /b %errorlevel%

:train
"%PY_CMD%" "%SCRIPTS_DIR%run_pipeline.py" train %2 %3 %4 %5 %6 %7 %8 %9
exit /b %errorlevel%

:evaluate
"%PY_CMD%" "%SCRIPTS_DIR%run_pipeline.py" evaluate %2 %3 %4 %5 %6 %7 %8 %9
exit /b %errorlevel%

:predict
"%PY_CMD%" "%SCRIPTS_DIR%run_pipeline.py" predict %2 %3 %4 %5 %6 %7 %8 %9
exit /b %errorlevel%

:clean
"%PY_CMD%" "%SCRIPTS_DIR%run_pipeline.py" clean %2 %3 %4 %5 %6 %7 %8 %9
exit /b %errorlevel%

:pipeline_run
"%PY_CMD%" "%SCRIPTS_DIR%run_pipeline.py" run --input data/raw/CMOS_addtional_datasets %2 %3 %4 %5 %6 %7 %8 %9
exit /b %errorlevel%

:pipeline_train
"%PY_CMD%" "%SCRIPTS_DIR%run_pipeline.py" train --input data/raw/CMOS_addtional_datasets %2 %3 %4 %5 %6 %7 %8 %9
exit /b %errorlevel%

:pipeline_eval
"%PY_CMD%" "%SCRIPTS_DIR%run_pipeline.py" evaluate %2 %3 %4 %5 %6 %7 %8 %9
exit /b %errorlevel%

:pipeline_predict
"%PY_CMD%" "%SCRIPTS_DIR%run_pipeline.py" predict %2 %3 %4 %5 %6 %7 %8 %9
exit /b %errorlevel%

:usage
echo Usage:
echo   scripts\run_pipeline.bat run --input data/raw/CMOS_addtional_datasets [--config configs/default.yaml] [--skip-batch]
echo   scripts\run_pipeline.bat train --input data/raw/CMOS_addtional_datasets [--config configs/default.yaml]
echo   scripts\run_pipeline.bat evaluate [--config configs/default.yaml]
echo   scripts\run_pipeline.bat predict [--config configs/default.yaml]
echo   scripts\run_pipeline.bat clean
echo.
echo Shortcut aliases:
echo   scripts\run_pipeline.bat pipeline-run
echo   scripts\run_pipeline.bat pipeline-train
echo   scripts\run_pipeline.bat pipeline-eval
echo   scripts\run_pipeline.bat pipeline-predict
echo.
echo Notes:
echo   - Uses .venv\Scripts\python.exe if available, otherwise falls back to python on PATH.
exit /b 0
