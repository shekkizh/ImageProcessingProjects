__author__ = 'Charlie'
# Implementaion uses code from opencv samples

import numpy as np
import cv2
import argparse, os, sys, inspect

PLY_HEADER = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''
FILENAME = "results/stereo.ply"


def write_ply():
    with open(FILENAME, 'w') as f:
        f.write(PLY_HEADER % dict(vert_num=len(accumulated_verts)))
        np.savetxt(f, accumulated_verts, '%f %f %f %d %d %d')


def append_ply_array(verts, colors):
    global accumulated_verts
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts_new = np.hstack([verts, colors])
    if accumulated_verts is not None:
        accumulated_verts = np.vstack([accumulated_verts, verts_new])
    else:
        accumulated_verts = verts_new




def stereo_match(imgL, imgR):
    # disparity range is tuned for 'aloe' image pair
    window_size = 5
    min_disp = 16
    num_disp = 112 - min_disp
    stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                   numDisparities=num_disp,
                                   blockSize=16,
                                   P1=8 * 3 * window_size ** 2,
                                   P2=32 * 3 * window_size ** 2,
                                   disp12MaxDiff=1,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=32
                                   )

    # print('computing disparity...')
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    # print('generating 3d point cloud...',)
    h, w = imgL.shape[:2]
    f = 0.8 * w  # guess for focal length
    Q = np.float32([[1, 0, 0, -0.5 * w],
                    [0, -1, 0, 0.5 * h],  # turn points 180 deg around x-axis,
                    [0, 0, 0, -f],  # so that y-axis looks up
                    [0, 0, 1, 0]])
    points = cv2.reprojectImageTo3D(disp, Q)
    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]
    append_ply_array(out_points, out_colors)

    disparity_scaled = (disp - min_disp) / num_disp
    disparity_scaled += abs(np.amin(disparity_scaled))
    disparity_scaled /= np.amax(disparity_scaled)
    disparity_scaled[disparity_scaled < 0] = 0
    return np.array(255 * disparity_scaled, np.uint8)


if __name__ == '__main__':
    # ap = argparse.ArgumentParser("Stereo matching in video!")
    # ap.add_argument("-l", "--left", required=True, help="Path to left image")
    # ap.add_argument("-r", "--right", required=True, help="Path to right image")
    # args = vars(ap.parse_args())

    # imgL = cv2.pyrDown(cv2.imread(args["left"]))  # downscale images for faster processing
    # imgR = cv2.pyrDown(cv2.imread(args["right"]))
    # stereo_match(imgL, imgR)
    camera = cv2.VideoCapture(0)
    accumulated_verts = None

    grabbed, frame = camera.read()
    if not grabbed:
        raise EnvironmentError("Camera read failed!")
    imgRef = cv2.pyrDown(frame)
    disparity = np.zeros((imgRef.shape[0], imgRef.shape[1]), dtype=np.uint8)
    while (True):
        _, frame = camera.read()
        img_curr = cv2.pyrDown(frame)
        disparity |= stereo_match(imgRef, img_curr)  # cv2.bitwise_and(disparity, stereo_match(imgRef, imgR))
        cv2.imshow('disparity', disparity)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            write_ply()
            break
        elif key & 0xFF == ord('r'):
            print "Resetting"
            imgRef = img_curr
            disparity *= 0
            accumulated_verts = None

    cv2.destroyAllWindows()
    camera.release()
