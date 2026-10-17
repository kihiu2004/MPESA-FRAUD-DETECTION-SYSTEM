@echo off
title M-Pesa Fraud Detection System

:menu
cls
echo ========================================
echo   M-Pesa Fraud Detection System
echo ========================================
echo.
echo Choose an option:
echo.
echo  [1] Install Dependencies
echo  [2] Run Full Pipeline (Train Models)
echo  [3] Run Dashboard (Web Interface)
echo  [4] Exit
echo.
set "choice="
set /p choice=Select option (1-4): 

if /i "%choice%"=="1" goto install
if /i "%choice%"=="2" goto pipeline
if /i "%choice%"=="3" goto dashboard
if /i "%choice%"=="4" goto exit
goto invalid

:install
call install_dependencies.bat
goto end

:pipeline
call run_pipeline.bat
goto end

:dashboard
call run_dashboard.bat
goto end

:invalid
echo Invalid selection. Please try again.
timeout /t 2 >nul
goto menu

:exit
exit

:end
pause
