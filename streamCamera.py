#!/usr/bin/env python 
# run: $python3 streamCamera.py 

import sys
import argparse
import subprocess
import cameraJetsonTx2

from threading import Thread
import threading

import cv2
cv2.__version__

WINDOW_NAME = "Stream Camera"

full_scrn = False
help_text = '"Esc" to Quit, "F" to Toggle Fullscreen'

def parse_args():
    # Parse input arguments
    desc = 'Capture and display live camera video on Jetson TX2/TX1'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--rtsp', dest='use_rtsp',
                        help='use IP CAM (remember to also set --uri)',
                        action='store_true')
    parser.add_argument('--uri', dest='rtsp_uri',
                        help='RTSP URI, e.g. rtsp://192.168.1.64:554',
                        default=None, type=str)
    parser.add_argument('--latency', dest='rtsp_latency',
                        help='latency in ms for RTSP [200]',
                        default=200, type=int)
    parser.add_argument('--usb', dest='use_usb',
                        help='use USB webcam (remember to also set --vid)',
                        action='store_true')
    parser.add_argument('--vid', dest='video_dev',
                        help='device # of USB webcam (/dev/video?) [1]',
                        default=1, type=int)
    parser.add_argument('--width', dest='image_width',
                        help='image width [1920]',
                        default=1920, type=int)
    parser.add_argument('--height', dest='image_height',
                        help='image height [1080]',
                        default=1080, type=int)
    args = parser.parse_args()
    return args

def open_window(width, height):
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, width, height)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, 'Camera Demo for Jetson TX2/TX1')

def keyControl():
    global full_scrn
    key = cv2.waitKey(10)
    if key == 27: # ESC key: quit program
        return True
    elif key == ord('F') or key == ord('f'): # toggle fullscreen
        full_scrn = not full_scrn
        if full_scrn:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    return False

def read_cam(cap):
    global help_text
    font = cv2.FONT_HERSHEY_PLAIN
    while True:
        if keyControl():
            break
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            # Check to see if the user has closed the window
            # If yes, terminate the program
            break
        _, img_ = cap.read() # grab the next image frame from camera
        img_ = cv2.flip(img_, 0)

        cv2.putText(img_, help_text, (11, 20), font, 1.0, (32, 32, 32), 4, cv2.LINE_AA)
        cv2.putText(img_, help_text, (10, 20), font, 1.0, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, img_)


def main():
    args = parse_args()
    print('Called with args: ', args)
    print('OpenCV version: {}'.format(cv2.__version__))

    if args.use_rtsp:
        cap = cameraJetsonTx2.open_cam_rtsp(args.rtsp_uri, args.image_width, args.image_height, args.rtsp_latency)
    elif args.use_usb:
        cap = cameraJetsonTx2.open_cam_usb(args.video_dev, args.image_width, args.image_height)
    else: # by default, use the Jetson onboard camera
        cap = cameraJetsonTx2.open_cam_onboard(args.image_width, args.image_height)

    if not cap.isOpened():
        sys.exit('Failed to open camera!')

    open_window(args.image_width, args.image_height)
    
    # read_cam(cap)
    try:
        t1 = threading.Thread(target=read_cam, args=(cap,))
        t1.start()
        t1.join()
    except:
        print ("threading is error")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()