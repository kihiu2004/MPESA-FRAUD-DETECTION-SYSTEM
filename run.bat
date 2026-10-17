@echo off
echo ========================================
echo M-Pesa Fraud Detection System
echo ========================================
echo.
echo Select an option:
echo.
echo 1. Install Dependencies
echo 2. Run Full Pipeline (Train Models)
echo 3. Run Dashboard (Web Interface)
echo 4. Exit
echo.

set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    call install_dependencies.bat
) else if "%choice%"=="2" (
    call run_pipeline.bat
) else if "%choice%"=="3" (
    call run_dashboard.bat
) else if "%choice%"=="4" (
    exit
) else (
    echo Invalid choice. Please try again.
    pause
    call run.bat
)
