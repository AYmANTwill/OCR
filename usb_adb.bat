@echo off
cd /d "%~dp0"
title Phone display over USB (adb tunnel)
set "ADB=%~dp0platform-tools\adb.exe"

if not exist "%ADB%" (
  echo platform-tools not found next to this file.
  pause
  exit /b 1
)

echo ============================================================
echo   Phone display over the USB cable  (adb tunnel - no tethering)
echo ============================================================
echo.
echo   On the phone, USB debugging must be ON (Developer options).
echo   The first time, the phone shows "Allow USB debugging?" -
echo   tick "Always allow from this computer" and tap Allow.
echo.
pause

echo.
echo [1/3] Looking for the phone...
"%ADB%" devices
echo.
echo   Shows your phone with "device"  -^> good, continue.
echo   Shows "unauthorized"            -^> tap Allow on the phone, run this again.
echo   List is empty                   -^> USB debugging off, or charge-only cable.
echo.
pause

echo [2/3] Opening the cable tunnel (phone localhost:8765 -^> laptop)...
"%ADB%" reverse --remove-all >nul 2>&1
"%ADB%" reverse tcp:8765 tcp:8765
if errorlevel 1 (
  echo.
  echo   Tunnel failed - phone not authorized yet.
  echo   Allow USB debugging on the phone and run this again.
  echo.
  pause
  exit /b 1
)
echo   tunnel active:
"%ADB%" reverse --list

echo.
echo [3/3] Starting the display server (no window appears)...
start "" pythonw headless.py --quiet
echo.
echo ============================================================
echo   On the phone's browser open:
echo.
echo        http://localhost:8765
echo.
echo   Same address every time. Ctrl+Alt+Q on the laptop quits.
echo ============================================================
echo.
pause
