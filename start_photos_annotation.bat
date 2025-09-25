@echo off
echo ====================================================
echo    Photos Annotation for Dual Board System
echo ====================================================
echo.
echo This script will help you annotate all photos in the
echo photos folder for training a new dual board model.
echo.
echo Features:
echo - Dual board split-screen annotation
echo - Navigate all 15 classes easily
echo - YOLO format output including green tape detection
echo - Enhanced keyboard controls
echo.
echo Classes include:
echo 0: Wire                    8: U_2 red alarm circuit
echo 1: Battery Holder         9: U_3 green space war circuit
echo 2: LED_1 (Yellow)        10: Speaker
echo 3: LED_2 (Red)           11: Slide switch
echo 4: Resistor              12: Press switch
echo 5: Lamp                  13: Whistle chip
echo 6: Photoresistor         14: Green tape
echo 7: U_1 blue music circuit
echo.
echo Controls:
echo - Mouse: Click + drag to draw bounding box
echo - Numbers 0-9: Direct class selection (0-9)
echo - UP/DOWN arrows: Cycle through ALL 15 classes
echo - +/- keys: Next/Previous class
echo - TAB: Switch between left/right side
echo - 'c': Show all classes
echo - 's': Save and continue
echo - 'q': Quit
echo.
pause
echo.
echo Starting annotation system...
python annotate_photos.py
echo.
echo Annotation session complete!
pause
