__author__ = 'Charlie'
import cv2
import argparse
import numpy as np
import os, inspect, sys

cmd_subFolder = os.path.realpath(
    os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..", "Image_lib")))
if cmd_subFolder not in sys.path:
    sys.path.insert(0, cmd_subFolder)

import image_utils as utils

def DitherImage(image):
    multiplier = np.array([[0, 0, 3 / 16], [5 / 16, 7 / 16, 1 / 16]], dtype=np.float)
    height, width = image.shape

    imageDithered = np.copy(image).astype(np.float)
    for row in range(height):
        for col in range(width):
            val = image[row][col]
            if (val > 192):
                imageDithered[row][col] = 192
            elif val > 128:
                imageDithered[row][col] = 128
            elif val > 64:
                imageDithered[row][col] = 64
            else:
                imageDithered[row][col] = 0

            error = val - imageDithered[row][col]

            if (row + 2 < height) and (col - 1 > 0) and (col + 2 < width):
                imageDithered[row:row + 2, col - 1:col + 2] += np.multiply(multiplier, error)

    return imageDithered


ap = argparse.ArgumentParser("Dither Halftoning - Floyd Steinberg. Truncated to 4 levels")
ap.add_argument("-i", "--image", required=True, help="Path to image file")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
# imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

imageDithered = cv2.merge([DitherImage(x) for x in cv2.split(image)])
cv2.imshow("Grayed Image", utils.image_resize(image, height = 600))
cv2.imshow("Dithered Image", utils.image_resize(imageDithered.astype(np.uint8), height=600))
cv2.waitKey()
cv2.destroyAllWindows()
