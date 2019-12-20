@echo off
:start
set RETRY_COUNT=0
set TF_CPP_MIN_LOG_LEVEL=2

:run
call %* || goto :wait
pause
exit /B 0

:wait
echo %date% %time% Retry %RETRY_COUNT%
set /a RETRY_COUNT=%RETRY_COUNT%+1
timeout /t 3 /nobreak
goto :run