__author__ = 'Charlie'
import numpy as np
import cv2
import argparse
import os, sys, inspect


cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0],"..","Image_Lib")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import image_utils as utils

def HalfToneImage(imageGray):
    height, width = imageGray.shape
    imageHalfTone = np.zeros((2*height, 2*width)).astype(np.uint8)

    dict = {0:[[0,0],[0,0]],
            51:[[255,0],[0,0]],
            102:[[0,255],[255,0]],
            153:[[255,255],[255,0]],
            204:[[255,255],[255,255]]}

    for row in range(height):
        for col in range(width):
            val = imageGray[row][col]
            if(val > 204):
                imageHalfTone[row*2:row*2+2, col*2:col*2+2] = dict[204]
            elif(val >153):
                imageHalfTone[row*2:row*2+2, col*2:col*2+2] = dict[153]
            elif(val > 102):
                imageHalfTone[row*2:row*2+2, col*2:col*2+2] = dict[102]
            elif(val > 51):
                imageHalfTone[row*2:row*2+2, col*2:col*2+2] = dict[51]
            else:
                imageHalfTone[row*2:row*2+2, col*2:col*2+2] = dict[0]

    return imageHalfTone

ap = argparse.ArgumentParser("Classical Half Toning [2x2 Mask]")
ap.add_argument('-i', '--image', required = True, help = 'Path to image file')
args = vars(ap.parse_args())

image = utils.image_resize(cv2.imread(args["image"]), height = 300)
imageHalfTone = cv2.merge([HalfToneImage(x) for x in cv2.split(image)])
cv2.imshow("HalfTone Image",imageHalfTone.astype(np.uint8))
cv2.waitKey()
cv2.destroyAllWindows()