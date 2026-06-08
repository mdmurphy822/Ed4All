@echo off
REM One-click launcher for the Ed4All control-plane GUI (Windows).
REM Double-click this file to set up a venv, install the GUI dependencies,
REM start the server, and open it in your browser. Any extra arguments are
REM forwarded to gui\launch.py (e.g. run-gui.bat --port 9000 --no-browser).

setlocal
set "SCRIPT_DIR=%~dp0"

REM Prefer the Python launcher (py); fall back to python on PATH.
where py >nul 2>nul
if %ERRORLEVEL%==0 (
    py "%SCRIPT_DIR%gui\launch.py" %*
    goto :done
)

where python >nul 2>nul
if %ERRORLEVEL%==0 (
    python "%SCRIPT_DIR%gui\launch.py" %*
    goto :done
)

echo ERROR: Could not find 'py' or 'python' on your PATH.
echo        Install Python 3.9+ from https://www.python.org/downloads/
echo        (tick "Add Python to PATH" during install) and re-run.
pause
exit /b 1

:done
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo The launcher exited with an error (code %ERRORLEVEL%).
    pause
)
endlocal
