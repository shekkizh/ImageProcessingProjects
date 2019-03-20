__author__ = 'Charlie'
# Borrowed some code structure from opencv samples

import numpy as np
import cv2
import sys, inspect, os
import argparse
import collections

cmd_subfolder = os.path.realpath(
    os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..","..", "Image_Lib")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import image_utils as utils

ap = argparse.ArgumentParser("Lucas Kanade flow tracking")
ap.add_argument("-v", "--video", required=False, help="Path to video file. Default - Camera")
ap.add_argument("-n", "--max_corners", required=False, type=int, help="Max no. of corners to track. Default 100")
args = vars(ap.parse_args())

if not args.get("max_corners", None):
    max_corners = 100
else:
    max_corners = args["max_corners"]

if not args.get("video", None):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args["video"])

feature_params = dict(maxCorners=max_corners, qualityLevel=0.3, minDistance=15, blockSize=7)

termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=termination)

track_length = 15
track_interval = 0
tracks = []
prev_gray_frame = None

print ("Reading camera...")
while True:
    grabbed, frame = camera.read()
    if not grabbed:
        raise EnvironmentError("Camera read failed!")
    frame = utils.image_resize(frame, height=600)
    output = frame.copy()
    curr_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if len(tracks) > 0:
        p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1,1,2)
        p1, state, err = cv2.calcOpticalFlowPyrLK(prev_gray_frame, curr_gray_frame, p0, None, **lk_params)
        p0r, state, err = cv2.calcOpticalFlowPyrLK(curr_gray_frame, prev_gray_frame, p1, None, **lk_params)
        d = abs(p0 -p0r).reshape(-1,2).max(-1)
        good = d < 1
        new_tracks = []

        for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1,2), good):
            if not good_flag:
                continue
            tr.append((x,y))
            if len(tr) > track_length:
                del tr[0]

            new_tracks.append(tr)
        tracks = new_tracks
        # print color
        cv2.polylines(output, [np.int32(tr) for tr in tracks], False, [200, 200, 200], 2)
        utils.add_text(output, ("tracking %d" % len(tracks)))
        track_interval += 1

    if track_interval % 100 == 0:
        mask = 255 * np.ones(curr_gray_frame.shape)
        for x,y in [np.int32(tr[-1]) for tr in tracks]:
            cv2.circle(mask, (x,y), 5, 0, -1)
        p = cv2.goodFeaturesToTrack(prev_gray_frame, mask=None, **feature_params)
        tracks = []
        if p is not None:
            for x,y in np.float32(p).reshape(-1,2):
                tracks.append([(x,y)])


    prev_gray_frame = curr_gray_frame
    cv2.imshow("Output", output)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        tracks = []
        track_length = 0
        print "Resetting tracks"

cv2.destroyAllWindows()
camera.release()

