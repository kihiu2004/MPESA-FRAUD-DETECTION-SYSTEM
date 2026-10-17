@echo off
echo ========================================
echo M-Pesa Fraud Detection System
echo ========================================
echo.
echo 1. Install Dependencies
echo 2. Run Full Pipeline (Train Models)
echo 3. Run Dashboard (Web Interface)
echo 4. Exit
echo.

choice /c 1234 /n /m "Enter your choice (1-4): "

if %errorlevel%==1 (
    call install_dependencies.bat
)
if %errorlevel%==2 (
    call run_pipeline.bat
)
if %errorlevel%==3 (
    call run_dashboard.bat
)
if %errorlevel%==4 (
    exit
)
