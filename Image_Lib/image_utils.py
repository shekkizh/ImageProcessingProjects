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

def sharpenImage(image):
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float)
    return cv2.filter2D(image, -1, kernel)

def order_points(pts):
	# The first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")

	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect

def image_rotate_by_90_clockwise(image):
    rows, cols, depth = image.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
    return cv2.warpAffine(image, M,(rows,cols))

def image_rotate_by_90_anticlockwise(image):
    rows, cols, depth = image.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-90,1)
    return cv2.warpAffine(image, M,(rows,cols))

def sort_contours(cnts, method = "left_to_right"):
    axis = 0
    reverse = False
    if method=="right_to_left" or method=="bottom_to_top":
        reverse = True
    if method=="top_to_bottom" or method=="bottom_to_top":
        axis = 1

    bounding_boxes = [cv2.boundingRect(c) for c in cnts]
    cnts, bounding_boxes = zip(* sorted(zip(cnts, bounding_boxes), key=lambda b:b[1][axis], reverse=reverse))
    return (cnts, bounding_boxes)

def get_midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0])*0.5, (ptA[1] + ptB[1])*0.5)