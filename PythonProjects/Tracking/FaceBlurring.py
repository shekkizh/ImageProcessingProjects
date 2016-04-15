__author__ = 'Charlie'
import numpy as np
import cv2
import sys, inspect, os
import argparse

cmd_subfolder = os.path.realpath(
    os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..","..", "Image_Lib")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import image_utils as utils

ap = argparse.ArgumentParser("Track and blur faces in video input")
ap.add_argument("-v", "--video", help="Path to video file. Defaults to webcam video")

args = vars(ap.parse_args())

if not args.get("video", False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args["video"])

face_cascade = cv2.CascadeClassifier('Image_Lib/Face_Data/haarcascade_frontalface_default.xml')

while (True):
    grabbed, frame = camera.read()
    if not grabbed:
        print "Camera read failed!"
        break

    frame = utils.image_resize(frame, height=600)
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # black_frame = np.zeros(frame.shape)
    faces = face_cascade.detectMultiScale(gray_image, 1.2, 2)

    if(len(faces) > 0):
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0),2)
            roi = frame[y:y+h, x:x+w,:]
            # noise = (np.random.randn(roi.shape[0], roi.shape[1], roi.shape[2])).reshape(roi.shape)
            frame[y:y+h, x:x+w,:] = cv2.GaussianBlur(roi, (25,25), 100) #roi + roi*noise
            cv2.imshow("Output", frame)
    # else:
    #     cv2.imshow("Output", black_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()



def camshift_track(prev, box, termination):
    hsv = cv2.cvtColor(prev,cv2.COLOR_BGR2HSV)
    x,y,w,h = box
    roi = prev[y:y+h, x:x+w]
    hist = cv2.calcHist([roi], [0], None, [16], [0, 180])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    backProj = cv2.calcBackProject([hsv], [0], hist, [0, 180], 1)
    (r, box) = cv2.CamShift(backProj, tuple(box), termination)
    return box

def camshift_face_track():
    face_cascade = cv2.CascadeClassifier('Image_Lib/Face_Data/haarcascade_frontalface_default.xml')
    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    ALPHA = 0.5

    camera = cv2.VideoCapture(0)
    face_box = None

    #wait till first face box is available
    print "Waiting to get first face frame..."
    while face_box is None:
        grabbed, frame = camera.read()
        if not grabbed:
            raise EnvironmentError("Camera read failed!")
        image_prev = cv2.pyrDown(frame)
        face_box = utils.detect_face(face_cascade, image_prev)

    print "Face found!"
    prev_frames = image_prev.astype(np.float32)
    while (True):
        _, frame = camera.read()
        image_curr = cv2.pyrDown(frame)
        cv2.accumulateWeighted(image_curr, prev_frames, ALPHA)
        image_curr = cv2.convertScaleAbs(prev_frames)
        if face_box is not None:
            face_box = camshift_track(image_curr, face_box, termination)
            cv2.rectangle(image_curr, (face_box[0], face_box[1]), (face_box[0]+face_box[2], face_box[1] + face_box[3]),
                          (255, 0,0), 2)
            # cv2.rectangle(image_curr, (box[0], box[1]), (box[0]+box[2], box[1] + box[3]),
            #               (0, 0,255), 2)

        else:
            face_box = utils.detect_face(face_cascade, image_curr)

        cv2.imshow("Output", image_curr)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

        elif key & 0xFF == ord('r'):
            print "Reseting face detection!"
            face_box = None
