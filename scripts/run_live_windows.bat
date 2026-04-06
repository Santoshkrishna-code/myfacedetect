@echo off
REM Run live face detection on Windows (uses virtualenv if present)
REM Usage: run_live_windows.bat [cam_index] [width] [height]
SETLOCAL

REM Resolve project root relative to this script (%~dp0 is scripts\ folder)
SET SCRIPTDIR=%~dp0
SET PROJROOT=%SCRIPTDIR%..\
SET VENV=%PROJROOT%facedetect_env

IF EXIST "%VENV%\Scripts\activate.bat" (
  echo Activating virtualenv at %VENV%
  call "%VENV%\Scripts\activate.bat"
) ELSE (
  echo No virtualenv found at %VENV%. Using system Python.
)

REM Prefer realtime_camera.py if present (accepts --cam/--width/--height)
IF EXIST "%PROJROOT%examples\realtime_camera.py" (
  SET SCRIPT=%PROJROOT%examples\realtime_camera.py
) ELSE IF EXIST "%PROJROOT%examples\detect_faces_live.py" (
  SET SCRIPT=%PROJROOT%examples\detect_faces_live.py
) ELSE (
  echo No live demo script found in examples\. Expected realtime_camera.py or detect_faces_live.py
  pause
  goto :eof
)

SET CAM=%1
IF "%CAM%"=="" SET CAM=0
SET WIDTH=%2
IF "%WIDTH%"=="" SET WIDTH=640
SET HEIGHT=%3
IF "%HEIGHT%"=="" SET HEIGHT=480

echo Running live demo: %SCRIPT% --cam %CAM% --width %WIDTH% --height %HEIGHT%
python "%SCRIPT%" --cam %CAM% --width %WIDTH% --height %HEIGHT%

echo Demo finished.
pause
