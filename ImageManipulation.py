__author__ = 'Charlie'
# works with raw images only
import numpy as np, cv2 as cv
from matplotlib import pyplot as plt


def imageRead(filename):
    readFile = open(filename, 'rb')
    img_str = readFile.read()
    readFile.close()
    img = np.array(bytearray(img_str), np.uint8).reshape((300, 400, 3))  # shape height, width, 3
    cv.cvtColor(img, cv.COLOR_BGR2RGB, img)
    return img


# def imageWrite(filename, image):
#     writeFile = open(filename, 'wb')


def showImage(title, image):
    cv.namedWindow(title, cv.WINDOW_AUTOSIZE)
    cv.imshow(title, image)
    cv.waitKey(0)


# works with gray images
def convertToGray(image):
    return (0.21 * image[:, :, 0] + 0.72 * image[:, :, 1] + 0.07 * image[:, :, 2]).astype(np.uint8)


def fullRangeLinearScaling(image):
    max_value = np.max(image)
    return ((255 / max_value) * image).astype(np.uint8)


inputImage = imageRead('desk.raw')
showImage('Input', inputImage)

grayImage = convertToGray(inputImage)

# showImage('Full Range Scaling', fullRangeLinearScaling(grayImage))
showImage('Homogenous scaling')