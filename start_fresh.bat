@echo off
echo ========================================
echo Fresh Start Snap Circuit Detection
echo ========================================
echo.

echo This will completely restart your snap circuit component detection process.
echo.
echo Steps:
echo 1. Setup environment and validate data
echo 2. Annotate images (if needed)
echo 3. Train YOLOv8 model
echo 4. Test detection on sample images
echo.

set /p choice="Do you want to continue? (y/n): "
if /i "%choice%" neq "y" (
    echo Cancelled.
    pause
    exit /b
)

echo.
echo Starting fresh start pipeline...
echo.

python fresh_start_main.py

echo.
echo Pipeline complete!
echo Check the output directory for results.
echo.
pause 