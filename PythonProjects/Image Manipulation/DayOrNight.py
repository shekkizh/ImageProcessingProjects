__author__ = 'Charlie'
import numpy as np
import cv2
import argparse
import sys, inspect,os

cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..", "Image_Lib")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import image_utils as utils

ap = argparse.ArgumentParser("Simple intensity measure to detect Day or Night picture")
ap.add_argument("-i", "--image", required = True, help = "Path to image file")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
(h,s,v) = cv2.split(imageHSV)
brightPixelCount = np.sum(v > 128)
pixelCount = image.shape[0]*image.shape[1]

if(brightPixelCount > pixelCount/2):
    print "DAY"
else:
    print "NIGHT"

cv2.imshow("Image", utils.image_resize(image, height = 600))
cv2.waitKey()
cv2.destroyAllWindows()

