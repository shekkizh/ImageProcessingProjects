__author__ = 'Charlie'

import numpy as np
import cv2
import sys, inspect, os
import argparse
import collections

cmd_subfolder = os.path.realpath(
    os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..", "Image_Lib")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import image_utils as utils

ap = argparse.ArgumentParser("Lucas Kanade flow tracking")
ap.add_argument("-v", "--video", required=False, help="Path to video file. Default - Camera")
ap.add_argument("-n", "--max_corners", required=False, type=int, help="Max no. of corners to track. Default 100")
args = vars(ap.parse_args())

if not args.get("max_corners", None):
    max_corners = 100
else:
    max_corners = args["max_corners"]

if not args.get("video", None):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args["video"])

feature_params = dict(maxCorners=max_corners, qualityLevel=0.3, minDistance=15, blockSize=7)

termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=termination)

face_cascade = cv2.CascadeClassifier('Image_Lib/Face_Data/haarcascade_frontalface_default.xml')

p0 = []
prev_frame = None

while (True):
    grabbed, frame = camera.read()
    if not grabbed:
        print "Camera read failed!"
        break

    frame = utils.image_resize(frame)
    curr_frame_gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    info = ""

    if prev_frame is None or len(p0) <= 3:
        info = "Detecting..."
        face = utils.detect_face(face_cascade, frame)
        if face is not None:
            prev_frame = curr_frame_gray
            x,y,w,h = face
            roi = np.zeros(prev_frame.shape, dtype=np.uint8)
            roi[y:y+h, x:x+w] = 255
            p0 = cv2.goodFeaturesToTrack(prev_frame, mask=roi, **feature_params)

    else:
        p1,st,err = cv2.calcOpticalFlowPyrLK(prev_frame, curr_frame_gray, p0, None, **lk_params)
        # Update points being tracked to new good set
        p0 = p1[st==1].reshape(-1,1,2)
        info = "Tracking: %d" % len(p0)
        for pt in p1:
            a,b = pt.ravel()
            cv2.circle(frame, (a,b), 3, (0,255,0), -1)
    prev_frame

    utils.add_text(frame, info)
    cv2.imshow("Output", frame)
    key = cv2.waitKey(1) & 0xFF
    if key== ord('q'):
        break

    elif key == ord('r'):
        p0 = []

camera.release()
cv2.destroyAllWindows()