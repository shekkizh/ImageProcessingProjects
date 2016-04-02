__author__ = 'Charlie'
import os, sys, inspect
import cv2
import numpy as np
import argparse

cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0],"..","Image_Lib")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import image_utils as utils

ap = argparse.ArgumentParser(description="Visualize image channels separately")
ap.add_argument("-i", "--image", required = True, help = "Path to image file")
ap.add_argument("-m", "--mode", required = False, help = "Enter image channels mode. Default = BGR")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
if(args["mode"] == None):
    (ch1, ch2, ch3) = cv2.split(image)
    print("BGR Color space")

elif(args["mode"].upper() == "HSV"):
    imageCvt = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    (ch1, ch2, ch3) = cv2.split(imageCvt)
    print("HSV Color space")

elif(args["mode"].upper() == "LAB"):
    imageCvt = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    (ch1, ch2, ch3) = cv2.split(imageCvt)
    print("LAB space")

cv2.imshow("Channel 1", utils.image_resize(ch1,height = 500))
cv2.imshow("Channel 2", utils.image_resize(ch2,height = 500))
cv2.imshow("Channel 3", utils.image_resize(ch3,height = 500))
cv2.waitKey(0)
cv2.destroyAllWindows()
