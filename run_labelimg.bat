@echo off
echo Starting Custom Annotation Tool for Snap Circuit Components...
echo.
echo CONTROLS:
echo - Click and drag to draw bounding box
echo - Press number keys (0-9) to change class:
echo   0=wire, 1=switch, 2=button, 3=battery_holder, 4=led...
echo - Press 's' to save annotations
echo - Press 'r' to reset current image  
echo - Press 'q' to quit
echo.
echo REMEMBER: Follow ANNOTATION_GUIDELINES.md 
echo - Label COMPLETE components, not individual parts!
echo.
python manual_annotate.py snap_circuit_image.jpg classes.txt 