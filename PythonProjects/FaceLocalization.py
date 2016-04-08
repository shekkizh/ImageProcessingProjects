__author__ = 'Charlie'
#Using Bayes inference to infer location of head with respect to camera after calibration


import numpy as np
import cv2
import sys, inspect, os
import argparse
import collections

cmd_subfolder = os.path.realpath(
    os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..", "Image_Lib")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import image_utils as utils

z_axis_length = 11 #Has to be odd
loc_probability = [1.0/z_axis_length for i in range(z_axis_length)] # uniform prior
pHit = 0.8
pMiss = 0.2

def closest_location(rect):
    closest_index = None
    min_value = float('Inf')
    for k, v in calibration_rects.items():
        value = abs(rect[2]*rect[3] - v[0]*v[1])
        if  value < min_value:
            closest_index = k
            min_value = value

    return closest_index

def sense(p, rect):
    '''
    :param p: prior
    :param rect: sensed measurement of rectangle
    :return: posterior
    '''
    index = closest_location(rect)
    if index is not None:
        p = [pHit*p[i] if i==index else pMiss*p[i] for i in range(z_axis_length)]
        # print p, sum(p)
        p = [p[i]/sum(p) for i in range(z_axis_length)]

    return p

camera = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('Image_Lib/Face_Data/haarcascade_frontalface_default.xml')
calibrate = True

calibration_rects = {}

while True:
    face_box = None
    grabbed, frame = camera.read()
    frame = utils.image_resize(frame , height = 600)
    face_box = utils.detect_face(face_cascade, frame)

    if face_box is None:
        continue

    cv2.rectangle(frame, (face_box[0], face_box[1]), (face_box[0] + face_box[2], face_box[1] + face_box[3]),
              (255, 0, 0), 2)

    if calibrate:
        utils.add_text(frame, "Press: W - closest,S - farthest,C - neutral, Q - Done")
        no_points_either_side = z_axis_length/2
        cv2.imshow("Calibrating...", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('w'):
            calibration_rects[0] = (face_box[2], face_box[3])
            print calibration_rects[0]

        elif key == ord('c'):
            calibration_rects[no_points_either_side] = (face_box[2], face_box[3])
            print calibration_rects[no_points_either_side]

        elif key == ord('s'):
            calibration_rects[z_axis_length - 1] = (face_box[2], face_box[3])
            print calibration_rects[z_axis_length - 1]

        elif key == ord('q'):
            if len(calibration_rects.keys()) == 3:
                print "Calibrating .... "
                calibrate = False
                front_diff = map(lambda a,b: abs(a-b)/no_points_either_side, calibration_rects[0], calibration_rects[no_points_either_side])
                back_diff = map(lambda a,b: abs(a-b)/no_points_either_side, calibration_rects[z_axis_length - 1], calibration_rects[no_points_either_side])

                for i in range(1,no_points_either_side):
                    calibration_rects[no_points_either_side - i] = tuple(sum(x) for x in zip(calibration_rects[no_points_either_side], tuple([i*val for val in front_diff])))
                    calibration_rects[no_points_either_side + 1 + i ] = tuple(sum(x) for x in zip(calibration_rects[no_points_either_side], tuple([i*val for val in back_diff])))

                print calibration_rects
                cv2.destroyWindow("Calibrating...")
            # print len(calibration_rects.keys())
                # print calibration_rects

    else:
        loc_probability = sense(loc_probability, face_box)
        # print loc_probability
        location = loc_probability.index(max(loc_probability))
        # print loc_probability
        utils.add_text(frame, ("Location %d, prob %g" % (location, loc_probability[location])))
        cv2.imshow("Output Estimation", frame)
        if cv2.waitKey() & 0xFF == ord('q'):
            break

camera.release()
cv2.destroyAllWindows()


