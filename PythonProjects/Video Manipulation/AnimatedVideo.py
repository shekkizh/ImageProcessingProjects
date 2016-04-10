__author__ = 'Charlie'
import numpy as np
import cv2
import sys, inspect, os
import argparse

cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0],"..","Image_Lib")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import image_utils as utils

ap = argparse.ArgumentParser("Real time edge finding on video")
ap.add_argument("-v", "--video", help="Path to video file. Defaults to camera if not provided")
ap.add_argument("-n", "--N", help = "No of colors")
args = vars(ap.parse_args())

if not args.get("video", False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args["video"])

if not args.get("N", False):
    n =10
else:
    n = args["N"]

termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    grabbed, frame = camera.read()
    if not grabbed:
        break

    # imageGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ret, lbl, centers = cv2.kmeans(frame.reshape((-1, 3)).astype(np.float32), n, None, termination, 3, cv2.KMEANS_PP_CENTERS)
    #
    # centers = centers.astype("uint8")
    # frame = centers[lbl.flatten()].reshape(frame.shape)

    # frame = 8*(frame/8)
    imageGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = cv2.GaussianBlur(imageGray, (5, 5), 0)
    result = utils.autoCanny(result)

    (_, cnts, _) = cv2.findContours(result, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = sorted(cnts, key=cv2.contourArea)
    # frame = cv2.pyrMeanShiftFiltering(frame,21, 51)
    frame = 8*(frame/8)
    cv2.drawContours(frame, cnts, -1, (0, 0, 0), 1)
    cv2.imshow("Video", utils.image_resize(frame, height=500))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
