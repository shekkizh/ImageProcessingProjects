__author__ = 'Charlie'
import cv2
import numpy
import argparse, os , sys,inspect

cmd_folder = os.path.abspath(os.path.realpath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0],"..","Image_Lib")))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

import image_utils as utils

ap = argparse.ArgumentParser("Feature Detection in Images")
ap.add_argument("-i","--image", required = True, help = "Path to image file")
ap.add_argument("-f","--feature", help = "Feature Detection type. DEFAULT = Harris Corner")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

if not args.get("feature", False) or args["feature"]== "HarrisCorner":
    result = cv2.cornerHarris(gray, 2, 3, 0.04)
    # Harris Corner results are of 32C1 type. The value is the R value obtained from det(M) - k(trace(M))^2
    cv2.dilate(result, None) #Dilation done to smooth out result clustered in one place
    image[result>0.01*result.max()] = [0,255,0]

elif args["feature"] == "ShiTomasiCorner":
    #Similar to Harris Corner except the value is considered as lambda greater than a threshold
    corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
    for i in corners:
        x,y = i.ravel()
        cv2.circle(image,(x,y),2, [0,255,0], -1) # -1 thickness would fill the circle - any negative value for that case
elif args["feature"] == "SIFT":
    siftClass = cv2.xfeatures2d.SIFT_create()
    kps = siftClass.detect(gray, None)
    image = cv2.drawKeypoints(gray, kps,None)

cv2.imshow("Output", utils.image_resize(image, height = 500))
cv2.waitKey()
cv2.destroyAllWindows()