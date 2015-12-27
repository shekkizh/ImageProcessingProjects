__author__ = 'Charlie'
import cv2

def image_resize(image, width = -1, height = -1):
    shape = image.shape
    if(width == -1):
        if(height == -1):
            return image
        else:
            return cv2.resize(image, (int(height * shape[1]/shape[0]), height))
    elif (height == -1):
        return cv2.resize(image, (width, int(width * shape[0]/shape[1])))


def adaptive_threshold(image):
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
