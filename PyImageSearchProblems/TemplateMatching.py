__author__ = 'Charlie'
import os, sys, inspect
import cv2
import numpy as np
import argparse

# Info:
# cmd_folder = os.path.dirname(os.path.abspath(__file__))
# __file__ fails if script is called in different ways on Windows
# __file__ fails if someone does os.chdir() before
# sys.argv[0] also fails because it doesn't not always contains the path
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0],"..","Image_Lib")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import image_utils as utils


ap = argparse.ArgumentParser()
ap.add_argument("--image", "-i", required = True, help = "Path to input image")
ap.add_argument("--template", "-t", required = True, help = "Path to template image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
template = cv2.imread(args["template"])
templateShape = template.shape[:2]

result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF)
(_,_, minLoc, maxLoc) = cv2.minMaxLoc(result)

# the puzzle image
topLeft = maxLoc
botRight = (topLeft[0] + templateShape[1], topLeft[1] + templateShape[0])
roi = image[topLeft[1]:botRight[1], topLeft[0]:botRight[0]]
# construct a darkened transparent 'layer' to darken everything
# in the puzzle except for waldo
mask = np.zeros(image.shape, dtype = "uint8")
image = cv2.addWeighted(image, 0.25, mask, 0.75, 0)

image[topLeft[1]:botRight[1], topLeft[0]:botRight[0]] = roi
# display the images
cv2.imshow("Image", utils.image_resize(image, height = 650))
cv2.imshow("Template", template)
cv2.waitKey(0)