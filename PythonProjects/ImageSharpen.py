__author__ = 'Charlie'
import numpy as np
import cv2
import sys, inspect, os
import argparse

cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0],"..","Image_Lib")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import image_utils as utils

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to image file")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Input", utils.image_resize(image))
# cv2.imshow("Output", utils.image_resize(utils.sharpenImage(image)))
cv2.imwrite("image.jpg", utils.sharpenImage(image))
cv2.waitKey()
cv2.destroyAllWindows()