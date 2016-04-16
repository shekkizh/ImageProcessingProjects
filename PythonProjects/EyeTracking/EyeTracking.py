__author__ = 'Charlie'

import numpy as np
import cv2
import argparse
import os, sys, inspect

cmd_subfolder = os.path.realpath(
    os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..", "..", "Image_Lib")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import image_utils as utils
import EyeTrackingLib as tracker

ap = argparse.ArgumentParser("Finds pupil location in eyes")
ap.add_argument("-i", "--image", required=True, help="Path to image file")
ap.add_argument("-m", "--mode", required=False, help="Process after detecting face Y/N. Default = Y")
args = vars(ap.parse_args())

face_cascade = cv2.CascadeClassifier('Image_Lib/Face_Data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('Image_Lib/Face_Data/haarcascade_eye.xml')
if not args.get("mode", None):
    detect_face = True
else:
    detect_face = False

image = cv2.imread(args["image"])
print image.shape

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
tracker.showImage = True

face_box = None
faces = face_cascade.detectMultiScale(gray, 1.1, 3)
if len(faces) > 0:
    face_box = max(faces, key=lambda item: item[2] * item[3])

if detect_face and face_box is not None:
    x, y, w, h = face_box
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 3)
    roi_x = int(x + w * 0.14)
    roi_y = int(y + h * 0.25)
    roi_w = int(w * 0.3)
    roi_h = int(h * 0.3)

    cv2.rectangle(image, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)
    row, col = tracker.find_eye_center(gray[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w])
    print row, col

    cv2.circle(image, (roi_x + col, roi_y + row), 10, (0, 255, 0), -1)

    roi_x = x + w - roi_w - int(w * 0.13)
    cv2.rectangle(image, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 255, 0), 2)
    row, col = tracker.find_eye_center(gray[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w])
    print row, col

    cv2.circle(image, (roi_x + col, roi_y + row), 10, (0, 255, 0), -1)

else:
    eyes = eye_cascade.detectMultiScale(gray, 1.2, 3)
    if len(eyes) == 2:
        x, y, w, h = eyes[0]
        row, col = tracker.find_eye_center(gray[y:y + h, x:x + w])
        print row, col
        cv2.circle(image, (x + col, y + row), 10, (255, 0, 0), -1)

        x, y, w, h = eyes[1]
        row, col = tracker.find_eye_center(gray[y:y + h, x:x + w])
        print row, col

        cv2.circle(image, (x + col, y + row), 10, (255, 0, 0), -1)

    else:
        row, col = tracker.find_eye_center(gray[:, 0:image.shape[1] / 2])
        print row, col

        cv2.circle(image, (col, row), 10, (0, 0, 255), -1)

        row, col = tracker.find_eye_center(gray[:, image.shape[1] / 2:])
        print row, col

        cv2.circle(image, (image.shape[1] / 2 + col, row), 10, (0, 0, 255), -1)

cv2.imshow("Output", utils.image_resize(image, height=600))
cv2.waitKey()
cv2.destroyAllWindows()
