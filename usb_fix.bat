@echo off
cd /d "%~dp0"
title USB phone display fixer (auto / DHCP)

net session >nul 2>&1
if %errorlevel% neq 0 (
  echo.
  echo   This needs administrator rights.
  echo   Close this, then RIGHT-CLICK usb_fix.bat  ^>  "Run as administrator".
  echo.
  pause
  exit /b 1
)

echo ============================================================
echo   USB-C phone display fixer  (lets the phone assign the address)
echo ============================================================
echo.
echo   On the PHONE, make sure USB tethering is ON:
echo     Settings ^> Connections ^> Mobile Hotspot and Tethering
echo     ^> USB tethering = ON
echo   (If this is the first try after plugging in, toggle it OFF then ON.)
echo.
pause

echo.
echo [1/3] Making sure the USB adapter is on automatic (DHCP)...
netsh interface ip set address name="Ethernet 3" dhcp >nul 2>&1
echo [2/3] Asking the phone for an address (can take ~10 sec)...
ipconfig /renew "Ethernet 3" >nul 2>&1
echo.

set "USBIP="
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "$a=@(Get-NetIPAddress -InterfaceAlias 'Ethernet 3' -AddressFamily IPv4 -ErrorAction SilentlyContinue).Where({$_.IPAddress -notlike '169.254*'}); if($a){$a[0].IPAddress}"`) do set "USBIP=%%i"

if "%USBIP%"=="" (
  echo   The phone did not give an address.
  echo   Toggle USB tethering OFF then ON on the phone and run this file again.
  echo   ^(If it keeps failing, tell me and we switch to the adb-cable method.^)
  echo.
  pause
  exit /b 1
)

echo   Laptop USB address is: %USBIP%
echo.
echo [3/3] Starting the phone display server (no window will appear)...
start "" pythonw headless.py --quiet
echo.
echo ============================================================
echo   On the phone's browser open:
echo.
echo        http://%USBIP%:8765
echo.
echo   (Ctrl+Alt+Q on the laptop quits the server.)
echo ============================================================
echo.
pause
