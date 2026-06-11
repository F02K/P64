@echo off
setlocal

set "ROOT=%~dp0"
set "PYTHONPATH=%ROOT%src"
set "DEFAULT_PROJECT=%ROOT%samples\FirstScene"

if /I "%~1"=="editor" goto editor
if /I "%~1"=="open" goto editor
if /I "%~1"=="hub" goto hub
if /I "%~1"=="run" goto run
if /I "%~1"=="game" goto run
if /I "%~1"=="build" goto build
if /I "%~1"=="build-hub" goto buildhub
if not "%~1"=="" goto hub_file

:hub
set "PROJECT=%~2"
if "%PROJECT%"=="" (
    python -m p64 hub
) else (
    python -m p64 hub "%PROJECT%"
)
goto done

:hub_file
python -m p64 hub "%~1"
goto done

:editor
set "PROJECT=%~2"
if "%PROJECT%"=="" set "PROJECT=%DEFAULT_PROJECT%"
python -m p64 editor "%PROJECT%"
goto done

:run
set "PROJECT=%~2"
if "%PROJECT%"=="" set "PROJECT=%DEFAULT_PROJECT%"
python -m p64 run "%PROJECT%"
goto done

:build
set "PROJECT=%~2"
if "%PROJECT%"=="" set "PROJECT=%DEFAULT_PROJECT%"
python -m p64 build "%PROJECT%"
goto done

:buildhub
python -m p64 build-hub
goto done

:usage
echo Usage:
echo   p64.bat
echo   p64.bat hub [project.p64]
echo   p64.bat editor [project-folder]
echo   p64.bat run [project-folder]
echo   p64.bat build [project-folder]
echo   p64.bat build-hub
echo.
echo Default project:
echo   %DEFAULT_PROJECT%
exit /b 1

:done
exit /b %ERRORLEVEL%
