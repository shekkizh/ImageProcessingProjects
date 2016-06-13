import numpy as np
import cv2
import argparse
import os, sys, inspect

cmd_subfolder = os.path.realpath(
    os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..", "..", "Image_Lib")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import image_utils as utils


def filter_contours(contours, min_area=100, max_area=300, angle_thresh=15.0):
    filtered = []
    for cnt in contours:
        if len(cnt) < 5:
            continue
        # rect = cv2.minAreaRect(cnt)
        (x, y), (major, minor), angle = cv2.fitEllipse(cnt)
        area = cv2.contourArea(cnt)
        # cv2.ellipse(image, ((x,y), (major,minor), angle), (0,255,0), 2)

        if abs(angle - 90) < angle_thresh:
            c = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, False), False)
            filtered.append(c)
    return filtered


def show_image(image1, image2, name="Images"):
    cv2.imshow(name, np.hstack([image1, image2]))
    cv2.waitKey()
    cv2.destroyWindow(name)


def morph_close(input, kernel):
    morph = cv2.morphologyEx(input, cv2.MORPH_CLOSE, kernel)

    morph = cv2.GaussianBlur(morph, (3, 3), 0)
    show_image(input, morph, "Morphology")
    return morph.astype(np.uint8)


ap = argparse.ArgumentParser("Finds contour of foot")
ap.add_argument("-i", "--image", required=True, help="Path to image file")
ap.add_argument("-bg", "--background", required=True, help="Path to background file")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
background = cv2.imread(args["background"])

equlize = cv2.equalizeHist
image_gray = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
background_gray = (cv2.cvtColor(background, cv2.COLOR_BGR2GRAY))

img_cnts = utils.get_contours(image_gray, param = 75)
bg_cnts = utils.get_contours(background_gray, param=75)
image_edges = cv2.drawContours(np.zeros_like(image_gray), img_cnts, -1, 255, 2)
bg_edges = cv2.drawContours(np.zeros_like(background_gray), bg_cnts, -1, 255, 2)
show_image(image_edges, bg_edges, "Edges")

output_diff = image_edges - bg_edges  # cv2.absdiff(image_edges, bg_edges)
output_diff[output_diff < 128] = 0
output_diff[output_diff > 128] = 255
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
morphed = morph_close(output_diff, kernel1)

kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3))
morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, kernel2)
morphed = cv2.erode(morphed, kernel1)
(_, cnts, _) = cv2.findContours(morphed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
cnts = filter_contours(cnts)
output = np.zeros_like(output_diff)
cv2.drawContours(output, cnts, -1, 255, 3)
output = cv2.erode(output, kernel1)
show_image(morphed, output, "Final contour detection")

# edges = utils.auto_canny(morphed)

image[output > 0] = [255, 0, 0]

# cv2.drawContours(image, cnts, -1, (255, 0, 0), 1)
cv2.imshow("Output", image)
cv2.waitKey()
cv2.destroyAllWindows()
filename = "foot_contour_" + args['image'].split("/")[-1]
cv2.imwrite(filename, image)