@echo off
REM Stop CVAT and release its ~4GB of RAM. Annotation data is kept in docker
REM volumes, so nothing is lost - start-cvat.cmd brings it all back.

echo Stopping CVAT containers...
wsl -d Ubuntu-24.04 -u root -e bash -lc "cd /mnt/d/Develope/seedGerm/cvat && docker compose stop"

echo Releasing the WSL VM...
wsl --shutdown

echo.
echo CVAT stopped. Your data is safe in the docker volumes.
pause
