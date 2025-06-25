@echo off
echo 🎯 SNAP CIRCUIT BATCH ANNOTATION
echo ================================
echo.
echo Available circuit images to annotate:
echo 1. circuit_01.jpg (val)
echo 2. circuit_02.jpg (train) 
echo 3. circuit_03.png (train)
echo 4. circuit_04.png (train)
echo 5. circuit_05.png (train)
echo 6. circuit_06.png (val)
echo 7. circuit_07.png (train)
echo 8. snap_circuit_image.jpg (already done)
echo.
echo OPTIONS:
echo A) Annotate ALL images in sequence
echo B) Annotate specific image by name
echo.
echo ANNOTATION GUIDELINES REMINDER:
echo • Label COMPLETE components, not individual parts
echo • battery_holder (3): ONE box around entire battery compartment  
echo • switch (1): ONE box around complete switch assembly
echo • led (4): ONE box around LED housing
echo • connection_node (9): ONE large box covering wire segments
echo.
set /p choice="Enter choice (A for all, or specific image name): "

if /i "%choice%"=="A" (
    echo Starting batch annotation of all images...
    python batch_annotate.py all
) else (
    echo Annotating specific image: %choice%
    python batch_annotate.py %choice%
)

echo.
echo Press any key to exit...
pause 