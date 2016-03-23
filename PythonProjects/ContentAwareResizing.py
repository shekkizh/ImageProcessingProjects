__author__ = 'Charlie'

import numpy as np
import cv2
import argparse
import os,sys,inspect

cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0],"..","Image_Lib")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import image_utils as utils

LARGE_VAL = float('Inf')

ap = argparse.ArgumentParser("Content Aware Image resizing - only width reduction DO NOT CHANGE AXIS")
ap.add_argument("-i", "--image", required=True, help="Path to image file to resize")
ap.add_argument("-a", "--axis", required=False, help="Axis by which to reduce size\
                                                     0 = height, 1= width. Default 1")
ap.add_argument("-p", "--percent", required=False, help="Percentage by which to reduce width of image\
                                                        (value: 0 - 100). Default 10")
args = vars(ap.parse_args())

if not args.get("percent", False):
    width_reduction = 0.1
else:
    width_reduction = float(args["percent"])/100

if not args.get("axis", False):
    rotate = False
else:
    rotate = not int(args["axis"])

image =cv2.imread(args["image"])
print "Image shape",
print image.shape

if rotate:
    print "rotating image"
    image = utils.image_rotate_by_90_clockwise(image)

# gradient_energy = cv2.Laplacian(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY), cv2.CV_64F)
gradient_energy_x = cv2.convertScaleAbs(cv2.Sobel((cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)), cv2.CV_64F, 1, 0, ksize=3))
gradient_energy_y = cv2.convertScaleAbs(cv2.Sobel((cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)), cv2.CV_64F, 0, 1, ksize=3))
gradient_energy = cv2.addWeighted(gradient_energy_x, 0.5, gradient_energy_y, 0.5, 0)
image_shape = image.shape

seam_energy_map = LARGE_VAL * np.ones((image_shape[0], image_shape[1]+2)) #Padding image on all sides

seam_energy_map[:,1:image_shape[1]+1] = gradient_energy

for iii in xrange(1, image_shape[0]):
    for jjj in xrange(1,image_shape[1]+2):
        seam_energy_map[iii][jjj] +=  min(seam_energy_map[iii-1,jjj-1:jjj+1])

number_of_cols_to_remove = int(width_reduction * image_shape[1])

single_column_bool = np.ones((image_shape[0],1), dtype=np.bool)
image_seam_reduced = image

for cols_removed in xrange(0, number_of_cols_to_remove):
    new_shape = (image_shape[0], image_shape[1]- cols_removed, image_shape[2])
    seam_bool_map = np.ones(new_shape, dtype=np.bool)

    min_energy_index = np.argmin(seam_energy_map[new_shape[0]-1,:])
    # count = 0
    try:
        for iii in xrange(new_shape[0]-1, 0, -1):
            seam_bool_map[iii, min_energy_index - 1,:] = False
            # count += 1
            # seam_premap[iii, min_energy_index] = LARGE_VAL
            min_energy_index += -1 + np.argmin(seam_energy_map[iii-1, min_energy_index - 1 : min_energy_index + 1])
        seam_bool_map[0, min_energy_index -1, :] = False
        # count+=1

    except ValueError:
        print iii, min_energy_index, seam_energy_map[iii, min_energy_index]

    # print count
    reduced_shape = (new_shape[0],new_shape[1]-1,new_shape[2])
    image_seam_reduced = np.reshape(image_seam_reduced[seam_bool_map], reduced_shape)

    seam_energy_bool_map = np.concatenate((single_column_bool, seam_bool_map[:,:,1], single_column_bool), axis=1)
    seam_energy_map = np.reshape(seam_energy_map[seam_energy_bool_map],(reduced_shape[0], reduced_shape[1]+2))

if rotate:
    image = utils.image_rotate_by_90_anticlockwise(image)
    image_seam_reduced = utils.image_rotate_by_90_anticlockwise(image_seam_reduced)

print "Seam reduced size",
print image_seam_reduced.shape

cv2.imshow("Input", image)
cv2.imshow("Output", image_seam_reduced)
cv2.waitKey()
cv2.destroyAllWindows()