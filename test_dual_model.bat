@echo off
echo ====================================================
echo         Testing New Dual Board Model
echo ====================================================
echo.
echo This script will test your newly trained dual board model:
echo   Model: dual_board_dual_board_1758049257.pt
echo.
echo Features:
echo   ✓ Live camera with real-time image splitting
echo   ✓ Graph generation for connectivity analysis
echo   ✓ Component detection and scoring
echo   ✓ Circuit visualization with PNG outputs
echo.

:MENU
echo Choose test mode:
echo.
echo 1. Standard Test (2 second processing intervals)
echo 2. Fast Test (1 second processing intervals)
echo 3. Display Only (no file saving)
echo 4. Custom Settings
echo 5. Exit
echo.

set /p choice=Enter your choice (1-5): 

if "%choice%"=="1" goto STANDARD
if "%choice%"=="2" goto FAST
if "%choice%"=="3" goto DISPLAY_ONLY
if "%choice%"=="4" goto CUSTOM
if "%choice%"=="5" goto EXIT

echo Invalid choice! Please try again.
goto MENU

:STANDARD
echo.
echo ====== STANDARD TEST MODE ======
echo Starting dual board test with standard settings...
echo.
python test_new_dual_board_model.py --model models/weights/dual_board_dual_board_1758049257.pt
goto END

:FAST
echo.
echo ====== FAST TEST MODE ======
echo Starting dual board test with fast processing...
echo.
python test_new_dual_board_model.py --model models/weights/dual_board_dual_board_1758049257.pt --fast-mode
goto END

:DISPLAY_ONLY
echo.
echo ====== DISPLAY ONLY MODE ======
echo Starting dual board test with display only (no saving)...
echo.
python test_new_dual_board_model.py --model models/weights/dual_board_dual_board_1758049257.pt --no-save
goto END

:CUSTOM
echo.
echo ====== CUSTOM SETTINGS ======
echo.
set /p camera=Enter camera ID (default 0): 
if "%camera%"=="" set camera=0

set /p split=Enter split ratio (default 0.5): 
if "%split%"=="" set split=0.5

set /p interval=Enter processing interval in seconds (default 2.0): 
if "%interval%"=="" set interval=2.0

echo.
echo Starting dual board test with custom settings:
echo   Camera: %camera%
echo   Split ratio: %split%
echo   Interval: %interval%s
echo.
python test_new_dual_board_model.py --model models/weights/dual_board_dual_board_1758049257.pt --camera %camera% --split-ratio %split% --interval %interval%
goto END

:EXIT
echo.
echo Exiting dual board model tester.
goto END

:END
echo.
echo ====================================================
echo           Test Session Complete
echo ====================================================
echo.
echo Outputs saved to:
echo   • dual_board_output/left/     (Left board results)
echo   • dual_board_output/right/    (Right board results) 
echo   • dual_board_output/combined/ (Combined analysis)
echo.
echo Circuit visualizations:
echo   • latest_circuit_visual.png files for live viewing
echo   • Timestamped PNG files for historical analysis
echo.
pause
