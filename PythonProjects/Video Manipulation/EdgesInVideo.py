__author__ = 'Charlie'
import argparse, inspect, sys, os
import cv2
import numpy as np

cmd_subfolder = os.path.realpath(
    os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..", "Image_Lib")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import image_utils as utils

ap = argparse.ArgumentParser("Real time edge finding on video")
ap.add_argument("-v", "--video", help="Path to video file. Defaults to camera if not provided")
args = vars(ap.parse_args())

if not args.get("video", False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args["video"])

while True:
    grabbed, frame = camera.read()
    if not grabbed:
        break

    imageGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = cv2.GaussianBlur(imageGray, (5, 5), 0)
    result = cv2.Canny(result, 50, 175)

    (_, cnts, _) = cv2.findContours(result, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = sorted(cnts, key=cv2.contourArea)
    cv2.drawContours(frame, cnts, -1, (0, 0, 0), 1)
    cv2.imshow("Edge", utils.image_resize(frame, height=500))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
