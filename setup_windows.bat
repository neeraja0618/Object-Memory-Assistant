@echo off
cd /d "%~dp0"
py -3.11 --version >nul 2>&1
if errorlevel 1 (
  echo Python 3.11 was not found. Install it and run setup again.
  echo If supported by your Python Install Manager, use: py install 3.11
  pause
  exit /b 1
)
py -3.11 -m venv .venv
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto failed
".venv\Scripts\python.exe" install_runtime.py
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" doctor.py
if errorlevel 1 goto failed
echo Setup finished. Run run_windows.bat to open the Object Memory Assistant.
pause
exit /b 0
:failed
echo Setup failed. Read the message above. See README.md for troubleshooting.
pause
exit /b 1
