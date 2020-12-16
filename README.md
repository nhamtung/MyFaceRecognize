# MyFaceRecognize

# Require
- Jetson TX2 and camera
- Screen
- Install:
    + $pip3 install dlib==19.8.1
    + $pip3 install opencv-python
--------------------------------------------------------------------------------

# Open windown for stream camera
- Run with onboard camera: $python3 streamCamera.py
- Run with usb camera: 
    + Check usb camera port: $ls /dev/video* (example: /dev/video0)
    + Run: $python3 streamCamera.py --usb --vid=0
--------------------------------------------------------------------------------

# Test face recognize with jetson TX2
- Run: $python3 faceRecognize.py
--------------------------------------------------------------------------------

# Camture image and save to folder imageCapture with jetson TX2
# folderName is name of folder to save image of person capture (simple folderName is name of persion)
# press key 'C' to capture image
# Capture five image to auto encode image
- Run: $python3 captureImage.py --name='folderName'
--------------------------------------------------------------------------------

# Encode image by manual
# folderName is name of folder to saved image whith encoding
- Run: $python3 faceEncode.py --name='folderName'
--------------------------------------------------------------------------------