#!/bin/bash

echo "===================================================="
echo "         Testing New Dual Board Model"
echo "===================================================="
echo ""
echo "This script will test your newly trained dual board model:"
echo "  Model: dual_board_dual_board_1758049257.pt"
echo ""
echo "Features:"
echo "  ✓ Live camera with real-time image splitting"
echo "  ✓ Graph generation for connectivity analysis"
echo "  ✓ Component detection and scoring"
echo "  ✓ Circuit visualization with PNG outputs"
echo ""

while true; do
    echo "Choose test mode:"
    echo ""
    echo "1. Standard Test (2 second processing intervals)"
    echo "2. Fast Test (1 second processing intervals)"
    echo "3. Display Only (no file saving)"
    echo "4. Custom Settings"
    echo "5. Exit"
    echo ""
    
    read -p "Enter your choice (1-5): " choice
    
    case $choice in
        1)
            echo ""
            echo "====== STANDARD TEST MODE ======"
            echo "Starting dual board test with standard settings..."
            echo ""
            python test_new_dual_board_model.py --model models/weights/dual_board_dual_board_1758049257.pt
            break
            ;;
        2)
            echo ""
            echo "====== FAST TEST MODE ======"
            echo "Starting dual board test with fast processing..."
            echo ""
            python test_new_dual_board_model.py --model models/weights/dual_board_dual_board_1758049257.pt --fast-mode
            break
            ;;
        3)
            echo ""
            echo "====== DISPLAY ONLY MODE ======"
            echo "Starting dual board test with display only (no saving)..."
            echo ""
            python test_new_dual_board_model.py --model models/weights/dual_board_dual_board_1758049257.pt --no-save
            break
            ;;
        4)
            echo ""
            echo "====== CUSTOM SETTINGS ======"
            echo ""
            read -p "Enter camera ID (default 0): " camera
            camera=${camera:-0}
            
            read -p "Enter split ratio (default 0.5): " split
            split=${split:-0.5}
            
            read -p "Enter processing interval in seconds (default 2.0): " interval
            interval=${interval:-2.0}
            
            echo ""
            echo "Starting dual board test with custom settings:"
            echo "  Camera: $camera"
            echo "  Split ratio: $split"
            echo "  Interval: ${interval}s"
            echo ""
            python test_new_dual_board_model.py --model models/weights/dual_board_dual_board_1758049257.pt --camera $camera --split-ratio $split --interval $interval
            break
            ;;
        5)
            echo ""
            echo "Exiting dual board model tester."
            exit 0
            ;;
        *)
            echo "Invalid choice! Please try again."
            echo ""
            ;;
    esac
done

echo ""
echo "===================================================="
echo "           Test Session Complete"
echo "===================================================="
echo ""
echo "Outputs saved to:"
echo "  • dual_board_output/left/     (Left board results)"
echo "  • dual_board_output/right/    (Right board results)" 
echo "  • dual_board_output/combined/ (Combined analysis)"
echo ""
echo "Circuit visualizations:"
echo "  • latest_circuit_visual.png files for live viewing"
echo "  • Timestamped PNG files for historical analysis"
echo ""
