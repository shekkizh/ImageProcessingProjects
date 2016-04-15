__author__ = 'Charlie'
import numpy as np
import cv2
import argparse
import os, sys, inspect

cmd_subfolder = os.path.realpath(
    os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..","..", "Image_Lib")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import image_utils as utils
import image_transform as transform

ap = argparse.ArgumentParser(description="(Personal peeve) Unwarp a presentation image click at a meeting")
ap.add_argument("-i", "--image", required=True, help="Path to image file")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cnts = utils.get_contours(gray)
cnt = max(cnts, key=cv2.contourArea)

hull = cv2.convexHull(cnt)
# cv2.drawContours(image,[hull],0, (255,0,0),2)
# cv2.imshow("Outline", image)
# cv2.waitKey()


rect = cv2.minAreaRect(hull)
box = np.int0(cv2.boxPoints(rect))

# cv2.drawContours(image,[box],0, (0,0,255),2)
# cv2.imshow("Outline", image)
# cv2.waitKey()

approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
print (len(approx))
if len(approx) == 4:
    box = approx.reshape(4,2)
    print "Poly approximation"

output = transform.four_point_transform(image, box)
cv2.imshow("Output", output)
cv2.waitKey()
cv2.destroyAllWindows()
cv2.imwrite("unwarped_presentation.jpg", output)
