__author__ = 'Charlie'
import cv2
import os, sys, inspect
import numpy as np
from glob import glob

cmd_subfolder = os.path.realpath(
    os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..", "..", "Image_Lib")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import image_utils as utils

IMG_MASK = "images/camera_calibration/chessboard*.jpg"

images = glob(IMG_MASK)

termination = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

obj_P = np.zeros((9 * 7, 3), np.float32)
obj_P[:, :2] = np.mgrid[0:7, 0:9].reshape(-1, 2)

objpoints = []
imgpoints = []

h,w = 0,0
for f in images:
    print "Processing - %s" % f
    image = cv2.imread(f, 0)  # read as grayscale
    if image is None:
        print "Read failed! - skipping"
        continue

    h,w = image.shape[:2]
    ret ,corners = cv2.findChessboardCorners(image, (7,9))
    if ret:
        objpoints.append(obj_P)
        cv2.cornerSubPix(image, corners, (5,5), (-1,-1), termination)
        imgpoints.append(corners.reshape(-1,2))

        # cv2.drawChessboardCorners(image, (7,9), corners, ret)
        # cv2.imshow("Output", image)
        # cv2.waitKey(500)
    else:
        print "Could not find chessboard corners in %s" % f

cv2.destroyWindow("Output")
rms, camera_matrix, distortion_coeff, rotation_vec, translation_vec = cv2.calibrateCamera(objpoints,
                                                                                          imgpoints,
                                                                                          (w,h), None, None)

print "RMS: ", rms
print camera_matrix
print distortion_coeff

image_to_undistort = cv2.imread("images/camera_calibration/chessboard05.jpg", 0)

# Method 1 as suggested in OpenCV docs - produces a wierd radial distorted output
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coeff, (w,h), 0)
undistorted_image = cv2.undistort(image_to_undistort, camera_matrix, distortion_coeff, None, new_camera_matrix)

# x,y,w,h = roi
# undistorted_image = undistorted_image[y:y+h, x:x+w]

# #Method 2 - Also fails :(
# mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, distortion_coeff,
#                                          None, camera_matrix, (w,h), 5)
# undistorted_image = cv2.remap(image_to_undistort, mapx, mapy, cv2.INTER_LINEAR)


cv2.imshow('Distorted Image', image_to_undistort)
cv2.imshow("Undistorted Image", undistorted_image)
cv2.waitKey()
cv2.destroyAllWindows()