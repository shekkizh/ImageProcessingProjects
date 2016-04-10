__author__ = 'Charlie'
import numpy as np
import cv2
import sys, inspect, os
import argparse

cmd_subfolder = os.path.realpath(
    os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..","..", "Image_Lib")))
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

face_cascade = cv2.CascadeClassifier('Image_Lib/Face_Data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('Image_Lib/Face_Data/haarcascade_eye.xml')

while True:
    grabbed, frame = camera.read()
    if not grabbed:
        print "Camera read failed!"
        break

    frame = utils.image_resize(frame, height=600)
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            gray_roi = gray_image[y:y + h, x:x + w]
            color_roi = frame[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(gray_roi, 1.1, 4)
            print len(eyes)
            if len(eyes) > 0:
                for ex, ey, ew, eh in eyes:
                    cv2.rectangle(color_roi, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)
                    gray_eye_roi = gray_roi[ey:ey + eh, ex:ex + ew]
                    color_eye_roi = color_roi[ey:ey + eh, ex:ex + ew]
                    circles = cv2.HoughCircles(gray_eye_roi, cv2.HOUGH_GRADIENT, 1, 20, minRadius=0)
                    if not circles:
                        try:
                            circles = np.uint16(np.around(circles))
                            for i in circles[0, :]:
                                cv2.circle(color_eye_roi, (i[0], i[1], i[2]), (0, 0, 255), 2)
                                cv2.circle(color_eye_roi, (i[0], i[1]), 2, (0, 255, 255), 2)
                        except AttributeError:
                            print "circles return empty!"
                            continue

    cv2.imshow("Output", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
