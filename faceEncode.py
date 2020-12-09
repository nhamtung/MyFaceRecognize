#!/usr/bin/env python 
# run: $python3 faceEncode.py --name='NhamVanTung'

import sys
import argparse
import faceRecoginitionLib

import cv2
cv2.__version__

def parse_args():
    # Parse input arguments
    desc = 'encoder Image'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--name', dest='nameFolderPersion', help='Name of folder contains image of Persion', default=None, type=str)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print('Called with args: ', args)
    print('OpenCV version: {}'.format(cv2.__version__))
    faceRecoginitionLib.faceEncode(args.nameFolderPersion)

if __name__ == '__main__':
    main()
