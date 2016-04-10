__author__ = 'Charlie'
import cv2
import os, sys, inspect

cmd_subfolder = os.path.realpath(
    os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..", "..", "Image_Lib")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import image_utils as utils

camera = cv2.VideoCapture(0)

count = 1

while True:
    grabbed, frame = camera.read()
    if not grabbed:
        raise EnvironmentError("Camera read failed!")

    cv2.imshow("Output", utils.image_resize(frame, height=600))
    key = cv2.waitKey() & 0xFF

    if key == ord('s'):
        cv2.imwrite("images/camera_calibration/chessboard%02d.jpg"%count, frame)
        print "Saved %02d" %count
        count +=1

    elif key==ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

