@echo off
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo Extract combined_demo_update inside your existing project folder first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" "%~dp0install_combined_demo.py"
if errorlevel 1 echo Installation failed. Keep this error visible and send it for help.
pause
