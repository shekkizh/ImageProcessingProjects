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
ap.add_argument("-s", "--source", required = True, help = "Path to source image")
ap.add_argument("-t", "--target", required = True, help = "Path to target image")

args = vars(ap.parse_args())
source = cv2.cvtColor(cv2.imread(args["source"]), cv2.COLOR_BGR2HSV).astype("float32")
target = cv2.cvtColor(cv2.imread(args["target"]), cv2.COLOR_BGR2HSV).astype("float32")

(h,s,v) = cv2.split(source)
(srcLMean, srcLStd) = (h.mean(), h.std())
# (srcVMean, srcVStd) = (v.mean(), v.std())

(h, s, v) = cv2.split(target)
(tarLMean, tarLStd) = (h.mean(), h.std())
# (tarVMean, tarVStd) = (v.mean(), v.std())

#subtract mean value of image
h -= tarLMean
# v -= tarVMean

#scale std deviation based on source image
h *= (tarLStd/srcLStd)
# v *= (tarVStd/srcVStd)

#add source image mean to target
h += srcLMean
# v += srcVMean

# clip the pixel intensities to [0, 255] if they fall outside
# this range
h = np.clip(h, 0, 360)
h *= 0.5

# v += 128
# v= np.clip(v, 0, 255)

# merge the channels together and convert back to the RGB color
# space, being sure to utilize the 8-bit unsigned integer data
# type
transfer = cv2.merge([h, s, v])
transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_HSV2BGR)

cv2.imshow("Color Transform", utils.image_resize(transfer, height = 500))
cv2.waitKey()
cv2.destroyAllWindows()
cv2.imwrite("result.jpg", transfer)
