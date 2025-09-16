@echo off
echo ====================================================
echo    Dual Board Snap Circuit Detection System
echo ====================================================
echo.
echo This script helps you get started with the dual board
echo detection system for snap circuits.
echo.

:MENU
echo Choose an option:
echo.
echo 1. Check System Status
echo 2. Complete Workflow (Annotation → Training → Live Detection)  
echo 3. Manual Annotation Only
echo 4. Data Augmentation Only
echo 5. Model Training Only
echo 6. Live Detection Only
echo 7. Exit
echo.

set /p choice=Enter your choice (1-7): 

if "%choice%"=="1" goto STATUS
if "%choice%"=="2" goto COMPLETE
if "%choice%"=="3" goto ANNOTATE
if "%choice%"=="4" goto AUGMENT
if "%choice%"=="5" goto TRAIN
if "%choice%"=="6" goto LIVE
if "%choice%"=="7" goto EXIT

echo Invalid choice! Please try again.
goto MENU

:STATUS
echo.
echo ====== CHECKING SYSTEM STATUS ======
python setup_dual_board_system.py --status
echo.
pause
goto MENU

:COMPLETE
echo.
echo ====== RUNNING COMPLETE WORKFLOW ======
echo This will run the entire pipeline from annotation to live detection.
echo Make sure your images are in the new_images/ folder.
echo.
set /p confirm=Continue? (y/n): 
if /i "%confirm%"=="y" (
    python setup_dual_board_system.py --complete-workflow --images new_images/
) else (
    echo Cancelled.
)
echo.
pause
goto MENU

:ANNOTATE
echo.
echo ====== MANUAL ANNOTATION ======
echo This will start the dual board annotation tool.
echo.
echo Controls:
echo - Mouse drag: Draw bounding box
echo - Numbers 0-9: Select component class
echo - TAB: Switch between left/right side
echo - SPACE: Toggle green tape detection
echo - 's': Save and continue to next image
echo - 'q': Quit
echo.
set /p confirm=Start annotation? (y/n): 
if /i "%confirm%"=="y" (
    python setup_dual_board_system.py --step annotate --images new_images/
) else (
    echo Cancelled.
)
echo.
pause
goto MENU

:AUGMENT
echo.
echo ====== DATA AUGMENTATION ======
echo This will create augmented training data from your annotations.
echo.
set /p confirm=Run data augmentation? (y/n): 
if /i "%confirm%"=="y" (
    python setup_dual_board_system.py --step augment --images new_images/ --annotations dual_board_annotations/
) else (
    echo Cancelled.
)
echo.
pause
goto MENU

:TRAIN
echo.
echo ====== MODEL TRAINING ======
echo This will train a new dual board detection model.
echo Training may take 30 minutes to several hours depending on your hardware.
echo.
echo GPU Requirements:
echo - NVIDIA GPU with 6GB+ VRAM recommended
echo - CUDA-compatible PyTorch installation
echo.
set /p epochs=Enter number of epochs (default 200): 
if "%epochs%"=="" set epochs=200

set /p confirm=Start training with %epochs% epochs? (y/n): 
if /i "%confirm%"=="y" (
    python setup_dual_board_system.py --step train --data dual_board_augmented_dataset/data.yaml --epochs %epochs%
) else (
    echo Cancelled.
)
echo.
pause
goto MENU

:LIVE
echo.
echo ====== LIVE DETECTION SYSTEM ======
echo This will start the live dual board detection system.
echo.
echo Make sure:
echo - Camera is connected and working
echo - Two circuit boards are positioned side by side
echo - Green tape is visible on both boards
echo - Good lighting conditions
echo.
echo Controls during live detection:
echo - 'q': Quit system
echo - 's': Save current frame
echo - 'p': Pause/Resume processing
echo - 't': Toggle green tape detection overlay
echo - SPACE: Force process current frame
echo.

set /p camera=Enter camera ID (default 0): 
if "%camera%"=="" set camera=0

set /p model=Enter model path (leave empty for default): 

set /p confirm=Start live detection? (y/n): 
if /i "%confirm%"=="y" (
    if "%model%"=="" (
        python setup_dual_board_system.py --step live --camera %camera%
    ) else (
        python setup_dual_board_system.py --step live --camera %camera% --model "%model%"
    )
) else (
    echo Cancelled.
)
echo.
pause
goto MENU

:EXIT
echo.
echo Thanks for using the Dual Board Detection System!
echo.
echo Outputs are saved in:
echo - dual_board_annotations/     (annotations)
echo - dual_board_augmented_dataset/  (training data)
echo - models/weights/             (trained models)
echo - dual_board_output/          (live detection results)
echo.
echo For more information, see DUAL_BOARD_SYSTEM_README.md
echo.
pause
exit

:ERROR
echo.
echo An error occurred. Please check:
echo 1. Python is installed and in PATH
echo 2. Required packages are installed (pip install ultralytics opencv-python)
echo 3. Project files are in the correct location
echo.
pause
goto MENU
