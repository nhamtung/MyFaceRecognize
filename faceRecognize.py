#!/usr/bin/env python 
# run: $python3 faceRecognize.py 

import sys
import argparse
import subprocess
import time
import faceRecoginitionLib
import cameraJetsonTx2

from threading import Thread
import threading

import cv2
cv2.__version__

WINDOW_NAME = "Camera"
nameRecognized = ""
pre_time_ = 0

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
    elif key == ord('H') or key == ord('h'): # toggle help message
        show_help = not show_help
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
        img_ = getFaceLocation(img_)

        cv2.putText(img_, help_text, (11, 20), font, 1.0, (32, 32, 32), 4, cv2.LINE_AA)
        cv2.putText(img_, help_text, (10, 20), font, 1.0, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, img_)


def getFaceLocation(img_):
    global nameRecognized
    location_ = faceRecoginitionLib.get_face_location(img_)
    if location_ != None:
        start_point_rec_ = (location_.left(), location_.top()) 
        end_point_ = (location_.right(), location_.bottom()) 
        color_rec_ = (0, 255, 0)
        thickness_rec_ = 5
        cv2.rectangle(img_, start_point_rec_, end_point_, color_rec_, thickness_rec_)
        
        start_point_text_ = (location_.left(), location_.top()-20) 
        font_text_ = 20
        fontScale_text_ = 1
        color_text_ = (0, 0, 255)
        thickness_text_ = 1
        cv2.putText(img_, nameRecognized, start_point_text_, font_text_, fontScale_text_, color_text_, thickness_text_, cv2.LINE_AA)
    return img_
            
def faceRecognize(cap):
    global pre_time_, nameRecognized
    while True:
        if keyControl():
            break
        curr_time_ = time.clock()
        time_ = curr_time_ - pre_time_
        if time_ > 0.5:
            _, img_ = cap.read() # grab the next image frame from camera
            img_ = cv2.flip(img_, 0)
            emb_ = faceRecoginitionLib.get_face_encode(img_)
            # print("testCameraJetsonTX2.py - emb_: ", emb_)
            nameRecognized = faceRecoginitionLib.read_image_encode(emb_)
            print("testCameraJetsonTX2.py - nameRecognized: ", nameRecognized)
            pre_time_ = curr_time_

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
    
    # cameraJetsonTx2.read_cam(cap)
    try:
        t1 = threading.Thread(target=read_cam, args=(cap,))
        t2 = threading.Thread(target=faceRecognize, args=(cap,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
    except:
        print ("threading is error")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()