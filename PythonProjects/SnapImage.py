__author__ = 'Charlie'
#An attempt at image security where screenshots will not be able to capture imagepyth
import cv2
import numpy as np
import argparse, sys, inspect, os
import time

cmd_subfolder = os.path.realpath(
    os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..", "Image_Lib")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import image_utils as utils

ap = argparse.ArgumentParser("Snap Images - snapchat images only better")
ap.add_argument("-i", "--image", required=True, help="Path to image file")
args = vars(ap.parse_args())

image = utils.image_resize(cv2.imread(args["image"]), height=500)
reset = np.zeros(image.shape)
output = image.copy()
width = image.shape[1]
height = image.shape[0]

mask = {0: (slice(0, height / 2), slice(0, width / 2)), 1: (slice(height / 2, height), slice(width / 2, width)), \
        2: (slice(height / 2, height), slice(0, width / 2)), 3: (slice(0, height / 2), slice(width / 2, width))}
image1 = image.copy()
image1[mask[0]] = cv2.GaussianBlur(image1[mask[0]], (5, 5), 0)
image2 = image.copy()
image2[mask[1]] = cv2.GaussianBlur(image2[mask[1]], (5, 5), 0)
image3 = image.copy()
image3[mask[2]] = cv2.GaussianBlur(image3[mask[2]], (5, 5), 0)
image4 = image.copy()
image4[mask[3]] = cv2.GaussianBlur(image4[mask[3]], (5, 5), 0)

imageDictionary = {0: image1, 1: image2, 2: image3, 3: image4}

count = 0
n = 0
start = time.time()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mov', fourcc, 200.0, (width, height))

while True:
    output = imageDictionary[count]
    count += 1
    n += 1
    if count == 4:
        count = 0
        # cv2.putText(output, "{:.2f}".format(n/(end - start)), (5,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0))

    # out.write(output)
    cv2.imshow("Output", output)
    # output = image
    if cv2.waitKey(1) & 0xff == ord("q"):
        break
end = time.time()
print "{:.2f}".format(n / (end - start))
cv2.destroyAllWindows()
out.release()
