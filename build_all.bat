@echo off
setlocal

set "ROOT=%~dp0"
python "%ROOT%tools\build_tool\build_tool.py" %*
exit /b %ERRORLEVEL%
