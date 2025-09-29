@echo off
echo ====================================================
echo    Photos Model Training Status Checker
echo ====================================================
echo.
echo Checking training progress...
echo.

python monitor_photos_training.py

echo.
echo ====================================================
echo Choose an option:
echo.
echo 1. Check training progress again
echo 2. Test model (when training is complete)
echo 3. View training directory
echo 4. Exit
echo.

set /p choice=Enter your choice (1-4): 

if "%choice%"=="1" goto START
if "%choice%"=="2" goto TEST
if "%choice%"=="3" goto VIEW
if "%choice%"=="4" goto EXIT

echo Invalid choice! Please try again.
pause
goto START

:START
cls
goto BEGIN

:TEST
echo.
echo ====== TESTING TRAINED MODEL ======
python test_photos_model.py
echo.
pause
goto START

:VIEW
echo.
echo ====== OPENING TRAINING DIRECTORY ======
start dual_board_training\photos_model_fixed
echo.
pause
goto START

:EXIT
echo.
echo Goodbye!
pause

:BEGIN
