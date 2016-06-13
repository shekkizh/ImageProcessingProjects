import numpy as np
import cv2
import argparse
import os, sys, inspect

cmd_subfolder = os.path.realpath(
    os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..", "..", "Image_Lib")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import image_utils as utils

ap = argparse.ArgumentParser("Finds contour of foot")
ap.add_argument("-i", "--image", required=True, help="Path to image file")
ap.add_argument("-bg", "--background", required=True, help="Path to background file")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
background = cv2.imread(args["background"])
bg_lab = cv2.cvtColor(background, cv2.COLOR_BGR2LAB)

# fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
# fgbg.apply(background)
# output = fgbg.apply(image)


image_gray = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
background_gray = (cv2.cvtColor(background, cv2.COLOR_BGR2GRAY))
output = image_gray - background_gray
output[output < 128] = 0
output[output>=128] = 255
thresh, output = cv2.threshold(output, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

image_processed = np.ma.array(image_lab.copy(), mask = [output,output,output])
# print image_processed.shape
mean = np.mean(np.mean(image_processed, axis =0), axis=0)
print mean
std = np.std(np.std(image_processed, axis = 0), axis=0)
print std
output = cv2.inRange(image_lab, mean - 2*std, mean + 2*std)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
skinMask = cv2.erode(output, kernel)
skinMask = cv2.dilate(output, kernel)

skinMask = cv2.GaussianBlur(output, (3, 3), 0)
# skinMask=output
output = cv2.bitwise_and(image, image, mask=skinMask)
cv2.imshow("input", image)
cv2.imshow("Mask", skinMask)
cv2.imshow("output", output)
cv2.waitKey()
cv2.destroyAllWindows()
cv2.imwrite("skin_segment.png", output)