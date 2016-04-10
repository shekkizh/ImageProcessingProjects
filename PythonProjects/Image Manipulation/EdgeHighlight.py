__author__ = 'Charlie'
import cv2
import numpy as np
import sys, inspect, os
import argparse

cmd_subfolder = os.path.abspath(os.path.realpath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0],"..", "..","Image_Lib")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import image_utils as utils

ap = argparse.ArgumentParser("Highlight edges in images")
ap.add_argument("-i", "--image", required = True, help = "Path to image file")
ap.add_argument("-c", "--color", required = False, help = "Highlight color - Black (Default)/ White")
args = vars(ap.parse_args())

if not args.get("color", False):
    color = (0,0,0)
elif args["color"].lower() == "white":
    color = (255,255,255)

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5,5), 0)
edged = cv2.Canny(gray, 50, 175)

(_, cnts, _ ) = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# cnts = sorted(cnts, key = cv2.contourArea, reverse=True)
cv2.drawContours(image,cnts, -1, color, 1)
cv2.imshow("Output", utils.image_resize(image, height = 600))
cv2.waitKey()
cv2.destroyAllWindows()
cv2.imwrite("results.jpg", image)