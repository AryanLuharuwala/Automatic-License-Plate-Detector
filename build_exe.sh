#!/bin/bash
echo "Building ALPR System Windows Executable..."
echo ""

# Create dist folder if it doesn't exist
mkdir -p dist

# Run PyInstaller
pyinstaller --onefile \
    --windowed \
    --name "ALPR_TollPlaza" \
    --icon=icon.ico \
    --add-data "config.json:." \
    --add-data "models:models" \
    --hidden-import=torch \
    --hidden-import=cv2 \
    --hidden-import=easyocr \
    --hidden-import=ultralytics \
    --hidden-import=psycopg2 \
    --collect-all torch \
    --collect-all easyocr \
    --collect-all ultralytics \
    main_gui.py

echo ""
echo "Build complete! Executable is in dist/ALPR_TollPlaza.exe"
