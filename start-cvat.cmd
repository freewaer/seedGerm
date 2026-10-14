@echo off
REM Start CVAT. The keepalive process is essential: without a Windows process
REM attached, WSL terminates the VM and CVAT dies with it.

echo Holding the WSL VM open...
tasklist /FI "IMAGENAME eq wsl.exe" | find /I "wsl.exe" >nul || (
  start "" /B wsl.exe -d Ubuntu-24.04 -u root -e sleep infinity
)
timeout /t 3 /nobreak >nul

echo Starting CVAT containers...
wsl -d Ubuntu-24.04 -u root -e bash -lc "cd /mnt/d/Develope/seedGerm/cvat && docker compose up -d"

echo.
echo CVAT is booting - the server needs 60-90 seconds before it answers.
echo Waiting...

powershell -NoProfile -Command ^
  "$d=(Get-Date).AddMinutes(5); do { try { if((Invoke-WebRequest 'http://localhost:8080/api/server/about' -UseBasicParsing -TimeoutSec 5).StatusCode -eq 200){ Write-Host '' ; Write-Host 'CVAT is READY -> http://localhost:8080' -ForegroundColor Green; exit } } catch {}; Write-Host -NoNewline '.'; Start-Sleep 5 } while((Get-Date) -lt $d); Write-Host ''; Write-Host 'Timed out - check: docker compose ps' -ForegroundColor Yellow"

echo.
echo Login:  admin  /  SeedGerm2026!
pause
