@echo off
echo Building ALPR System Windows Executable...
echo.

REM Create dist folder if it doesn't exist
if not exist "dist" mkdir dist

REM Log file for build output
set LOGFILE=build_log.txt
echo Logging build output to %LOGFILE%
echo. > %LOGFILE%

REM Run PyInstaller and show real-time output while logging
REM Redirect both stdout and stderr to the log
pyinstaller --onefile ^
    --windowed ^
    --name "ALPR_TollPlaza" ^
    --icon=icon.ico ^
    --add-data "config.json;." ^
    --add-data "models;models" ^
    --hidden-import=torch ^
    --hidden-import=cv2 ^
    --hidden-import=easyocr ^
    --hidden-import=ultralytics ^
    --hidden-import=psycopg2 ^
    --collect-all torch ^
    --collect-all easyocr ^
    --collect-all ultralytics ^
    main_gui.py > %LOGFILE% 2>&1

REM Also print log content in real time
type %LOGFILE%

echo.
echo Build complete! Executable is in dist\ALPR_TollPlaza.exe
echo Check %LOGFILE% for detailed errors and warnings.
pause
