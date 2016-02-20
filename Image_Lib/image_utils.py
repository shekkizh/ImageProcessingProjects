__author__ = 'Charlie'
import cv2
import numpy as np

def image_resize(image, width = -1, height = -1):
    shape = image.shape
    if(width == -1):
        if(height == -1):
            return image
        else:
            return cv2.resize(image, (int(height * shape[1]/shape[0]), height))
    elif (height == -1):
        return cv2.resize(image, (width, int(width * shape[0]/shape[1])))

# Image has to eb gray scaled and gaussian blurred before calling this function
def autoCanny(image, sigma = 0.33):
    val = np.median(image)
    lower = max(0,(1- sigma)*val)
    upper = min(255, (1+sigma)*val)
    return cv2.Canny(image, lower, upper)

def adaptive_threshold(image):
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
