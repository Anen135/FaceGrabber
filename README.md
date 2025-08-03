# install
pip install -r requirements.txt

# how compile it?
pyinstaller   --onefile   --paths=venv/Lib/site-packages   --collect-all=cv2   --hidden-import=cv2   --add-binary="venv/Lib/site-packages/cv2/opencv_videoio_ffmpeg4110_64.dll;cv2"   --icon="facegraber.ico"   --add-data="venv/Lib/site-packages/mediapipe/modules/face_detection/face_detection_full_range_cpu.binarypb;mediapipe/modules/face_detection"   main.py

## примечание:
со временем библитеки могут обновится, тогда нужно будет заменить аргументы --add-binary и --add-data на актуальные

# run
/dist/main.exe