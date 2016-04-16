__author__ = 'Charlie'
import numpy as np
import cv2
import sys, inspect, os

cmd_subfolder = os.path.realpath(
    os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..", "..", "Image_Lib")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import image_utils as utils

EYE_ROI_WIDTH = 50

showImage = True


def find_eye_center(image):
    """
    Find center of eye using Fabian's algorithm
    :param image: Gray scale image of eye
    :return: row, col identified as center
    """
    # print image.shape
    global showImage

    scaled_image = utils.image_resize(image.copy(), width=EYE_ROI_WIDTH)
    gradient_energy_x = cv2.Sobel(scaled_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_energy_y = cv2.Sobel(scaled_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = (gradient_energy_x ** 2 + gradient_energy_y ** 2) ** 0.5
    threshold = np.mean(gradient_magnitude) + np.std(gradient_magnitude) * 3
    gradient_energy_x /= gradient_magnitude
    gradient_energy_y /= gradient_magnitude
    mask = gradient_magnitude < threshold
    gradient_energy_x[mask] = 0
    gradient_energy_y[mask] = 0
    scaled_image = cv2.GaussianBlur(scaled_image, (5, 5), 0, 0)

    inverted_image = 255 * np.ones_like(scaled_image) - scaled_image
    if showImage:
        # cv2.HoughCircles(scaled_image, cv2.HOUGH_GRADIENT, 2, 12.0)
        cv2.imshow("EyeDebug", inverted_image)
        if (cv2.waitKey() & 0xFF) == ord('s'):
            showImage = False
            cv2.destroyWindow('EyeDebug')

    indices = np.indices(inverted_image.shape).astype(np.float32)
    indices += 1e-8
    output_sum = np.zeros_like(inverted_image).astype(np.float32)
    for row in range(output_sum.shape[0]):
        for col in range(output_sum.shape[1]):
            val1 = (indices[0] - row) * gradient_energy_y
            val2 = (indices[1] - col) * gradient_energy_x
            val = (val1 + val2)
            output_sum += inverted_image * (val - val.mean()) / val.std()
            # compute_location_weight(row, col, inverted_image, gradient_energy_x, gradient_energy_y)

    index = np.unravel_index(np.argmax(output_sum), output_sum.shape)
    rescaled_index = (
        index[0] * image.shape[0] / scaled_image.shape[0], index[1] * image.shape[1] / scaled_image.shape[1])
    return rescaled_index


def compute_location_weight(c_row, c_col, weight, grad_x, grad_y):
    output = np.zeros_like(weight)
    for row in range(weight.shape[0]):
        for col in range(weight.shape[1]):
            if row == c_row and col == c_col:
                continue
            displacement_x = c_col - col
            displacement_y = c_row - row
            disp_magnitude = (displacement_x ** 2 + displacement_y ** 2) ** 0.5
            displacement_x /= disp_magnitude
            displacement_y /= disp_magnitude
            dot_product = grad_x[row, col] * displacement_x + grad_y[row, col] * displacement_y
            dot_product = max(0.0, dot_product)
            output[row, col] = weight[row, col] * dot_product * dot_product

    return output
