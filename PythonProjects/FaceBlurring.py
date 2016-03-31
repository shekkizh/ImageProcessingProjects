__author__ = 'Charlie'
import numpy as np
import cv2
import sys, inspect, os
import argparse

cmd_subfolder = os.path.realpath(
    os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..", "Image_Lib")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import image_utils as utils

ap = argparse.ArgumentParser("Track and blur faces in video input")
ap.add_argument("-v", "--video", help="Path to video file. Defaults to webcam video")

args = vars(ap.parse_args())

if not args.get("video", False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args["video"])

face_cascade = cv2.CascadeClassifier('Image_Lib/haarcascade_frontalface_alt2.xml')

while (True):
    grabbed, frame = camera.read()
    if not grabbed:
        print "Camera read failed!"
        break
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = utils.image_resize(frame, height=600)

    faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)
    # print len(faces)
    # cv2.drawContours(frame, map(cv2.boxPoints, faces), -1, (0, 255, 0), 2)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0),2)
        roi = frame[y:y+h, x:x+w,:]
        frame[y:y+h, x:x+w,:] = 0
        # frame[y:y+h, x:x+w,:] = cv2.GaussianBlur(roi, (3,3),0)

    cv2.imshow("Output", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
