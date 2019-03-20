__author__ = 'Charlie'
import cv2
import numpy as np


def image_resize(image, width=-1, height=-1):
    shape = image.shape
    if width == -1:
        if height == -1:
            return image
        else:
            return cv2.resize(image, (int(height * shape[1] / shape[0]), height))
    elif height == -1:
        return cv2.resize(image, (width, int(width * shape[0] / shape[1])))
    else:
        cv2.resize(image, (width, height))


# Image has to eb gray scaled and gaussian blurred before calling this function
def auto_canny(image, sigma=0.33):
    val = np.median(image)
    lower = max(0, (1 - sigma) * val)
    upper = min(255, (1 + sigma) * val)
    return cv2.Canny(image, lower, upper)


def adaptive_threshold(image, option=cv2.THRESH_BINARY):
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, option, 11, 2)


def sharpen_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float)
    return cv2.filter2D(image, -1, kernel)


def order_points(pts):
    # The first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def image_rotate_by_90_clockwise(image):
    rows, cols, depth = image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
    return cv2.warpAffine(image, M, (rows, cols))


def image_rotate_by_90_anticlockwise(image):
    rows, cols, depth = image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -90, 1)
    return cv2.warpAffine(image, M, (rows, cols))


def get_contours(gray_image, param=None):
    '''
    :param gray_image: input gray level image
    :param param: higher threshold for canny edge detector, lower threshold is twice smaller. Default Auto Canny with sigma=0.33
    :return:contours
    '''

    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    if not param:
        edged = auto_canny(gray_image)
        # t, edged = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        edged = cv2.Canny(blurred, param / 2, param)
    _, cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cnts


def sort_contours(cnts, method="left_to_right"):
    axis = 0
    reverse = False
    if method == "right_to_left" or method == "bottom_to_top":
        reverse = True
    if method == "top_to_bottom" or method == "bottom_to_top":
        axis = 1

    bounding_boxes = [cv2.boundingRect(c) for c in cnts]
    cnts, bounding_boxes = zip(*sorted(zip(cnts, bounding_boxes), key=lambda b: b[1][axis], reverse=reverse))
    return cnts, bounding_boxes


def get_midpoint(ptA, ptB):
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5


def detect_face(face_cascade, image):
    gray_image = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)
    if len(faces) > 0:
        return max(faces, key=lambda item: item[2] * item[3])

    return None


def add_text(image, text):
    cv2.putText(image, text, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
