@echo off
setlocal

set "ROOT=%~dp0"
echo build_all.bat is deprecated. Use build.bat all instead.
python "%ROOT%tools\build_tool\build_tool.py" all %*
exit /b %ERRORLEVEL%
