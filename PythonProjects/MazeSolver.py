__author__ = 'Charlie'
#Idea from http://opencv-code.com

import os, sys, inspect
import cv2
import numpy as np
import argparse

cmd_subfolder = os.path.realpath(
    os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..", "Image_Lib")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import image_utils as utils

ap = argparse.ArgumentParser(description="Solve orthogonal mazes")
ap.add_argument("-i", "--image", required = True, help = "Path to image file")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cnts = utils.get_contours(gray_image, 200)
# cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

thresholded_image = utils.adaptive_threshold(gray_image, cv2.THRESH_BINARY_INV)
cv2.imshow("Output", utils.image_resize(thresholded_image, height=600))
cv2.waitKey()
_, cnts, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if len(cnts) != 2:
    print len(cnts)
    raise ValueError("Unable to solve maze - Failed at Contour finding!")

solution_image = np.zeros(gray_image.shape, dtype=np.uint8)
cv2.drawContours(solution_image, cnts, 0, (255,255,255),cv2.FILLED)

cv2.imshow("Output", utils.image_resize(solution_image, height=600))
cv2.waitKey()

kernel = np.ones((15, 15),  dtype=np.uint8)
solution_image = cv2.dilate(solution_image, kernel)
eroded_image = cv2.erode(solution_image, kernel)
solution_image = cv2.absdiff(solution_image, eroded_image)

cv2.imshow("Output", utils.image_resize(solution_image, height=600))
cv2.waitKey()

b,g,r = cv2.split(image)
b &= ~solution_image
g |= solution_image
r &= ~solution_image

solution_image = cv2.merge([b,g,r]).astype(np.uint8)
cv2.imshow("Output", utils.image_resize(solution_image, height=600))
cv2.waitKey()
cv2.destroyAllWindows()
cv2.imwrite("result.jpg", solution_image)