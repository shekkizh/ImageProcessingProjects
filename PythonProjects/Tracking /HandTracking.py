__author__ = 'Charlie'
# Hand Tracking using skin calibration

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


ap = argparse.ArgumentParser("Track and blur faces in video input")
ap.add_argument("-v", "--video", help="Path to video file. Defaults to webcam video")

args = vars(ap.parse_args())

camera = cv2.VideoCapture(0)

calibrated = False

grabbed, frame = camera.read()
if not grabbed:
    raise ValueError("Camera read failed!")
bg = utils.image_resize(frame, height=600).astype(np.float32)

while True:
    grabbed, frame = camera.read()
    if not grabbed:
        print "Camera read failed"
        break

    frame = utils.image_resize(frame, height=600)
    height, width, channels = frame.shape

    if not calibrated:
        # Sample hand color
        utils.add_text(frame, "Press space after covering rectangle with hand. Hit SPACE when ready")
        x, y, w, h = width / 4, height / 2, 50, 50

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Calibration", frame)
        if cv2.waitKey(2) & 0xFF == ord(' '):
            roi = frame[y:y + h, x:x + w, :]
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            min_value = np.amin(roi_hsv, (0, 1))
            max_value = np.amax(roi_hsv, (0, 1))
            cv2.destroyWindow("Calibration")
            calibrated = True

    else:
        cv2.accumulateWeighted(frame, bg, 0.01)
        frame ^= cv2.convertScaleAbs(bg)

        # frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # hand_mask = cv2.inRange(frame_hsv, min_value, max_value)
        #
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        # hand_mask = cv2.erode(hand_mask, kernel, iterations=2)
        # hand_mask = cv2.dilate(hand_mask, kernel, iterations=2)
        #
        # # hand_mask = cv2.GaussianBlur(hand_mask, (7, 7), 0).astype(np.uint8)
        #
        # cnts = utils.get_contours(hand_mask, 255)
        # cnt = max(cnts, key=cv2.contourArea)
        # # hand = cv2.bitwise_and(frame, frame, mask=hand_mask)
        # rect = cv2.minAreaRect(cnt)
        # box = np.int0(cv2.boxPoints(rect))
        # hull = cv2.convexHull(cnt)
        # cv2.drawContours(frame, [hull], 0, (255,0,0), 3)

        cv2.imshow("Output", frame)
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

camera.release()
cv2.destroyAllWindows()
