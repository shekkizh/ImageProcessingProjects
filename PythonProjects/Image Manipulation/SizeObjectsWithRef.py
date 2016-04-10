__author__ = 'Charlie'
#This idea is based on PyImageSearch blog on Measuring size of objects in Image

import cv2
import os,sys, inspect
import numpy as np
import argparse
from scipy.spatial import distance

cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0],"..","..","Image_Lib")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import image_utils as utils

ap = argparse.ArgumentParser("Def size of object based on reference. Both objects are to be in image and reference is to be"
                             "the left part of the image.")
ap.add_argument("-i", "--image", required=True, help = "Path to image file")
ap.add_argument("-w", "--width", type=float, required=True,
                help = "Size of reference object (inches). Prefereably an object which is symmetrical else chooses longer side")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred_image = cv2.GaussianBlur(image_gray, (5,5), 0)

edged_image = cv2.Canny(blurred_image, 50, 100)
#Perform closing operation
edged_image = cv2.dilate(edged_image, None, iterations = 1)
edged_image = cv2.erode(edged_image, None, iterations=1)

_,cnts,_ = cv2.findContours(edged_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnts, bounding_boxes = utils.sort_contours(cnts)

pixelsPerInch = None
orig = image.copy()

for c in cnts:
    if cv2.contourArea(c) <150:
        continue
    box = cv2.boxPoints(cv2.minAreaRect(c))
    ordered_box = utils.order_points(np.array(box, dtype=np.int))

    # cv2.drawContours(orig, [ordered_box.astype(np.int)], -1, (255,0,0), 2)

    (tl, tr, br, bl) = ordered_box
    (topX, topY) = utils.get_midpoint(tl, tr)
    (botX, botY) = utils.get_midpoint(bl, br)

    (leftX, leftY) = utils.get_midpoint(tl, bl)
    (rightX, rightY) = utils.get_midpoint(tr, br)

    width_in_pixels = distance.euclidean((leftX,leftY), (rightX,rightY))
    height_in_pixels = distance.euclidean((topX, topY), (botX, botY))

    if pixelsPerInch is None:
        pixelsPerInch = max(width_in_pixels, height_in_pixels)/args["width"]

    width = width_in_pixels/pixelsPerInch
    height = height_in_pixels/pixelsPerInch

    cv2.putText(orig, "{:.2f}in".format(width),(int(topX - 10), int(topY - 10)), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0, 0,255), 2)
    cv2.putText(orig, "{:.2f}in".format(height),(int(rightX - 10), int(rightY)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 0,255), 2)

cv2.imshow("Output", orig)
cv2.waitKey()
cv2.destroyAllWindows()
cv2.imwrite("results.jpg", orig)