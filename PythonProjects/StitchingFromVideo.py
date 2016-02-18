# Unfinished code - erroneous creates no stitch
__author__ = 'Charlie'
import argparse
import numpy as np
import cv2
import os,sys,inspect

cmd_subFolder = os.path.abspath(os.path.realpath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..","Image_Lib")))
if cmd_subFolder not in sys.path:
    sys.path.insert(0, cmd_subFolder)

import image_utils as utils

class SurfStitcher:
    def __init__(self, image, ratio = 0.75, reprojThresh = 4.0):
        self.leftImage = image
        self.ratio = ratio
        self.reprojThresh = reprojThresh

        self.surfFeature = cv2.xfeatures2d.SURF_create(500, extended = False, upright = False)
        #HessianThreshold = 500
        #No orientation calculation
        #64 dimension feature vector

        self.matcher = cv2.DescriptorMatcher_create('BruteForce')

        self.leftKps, self.leftDescriptor = self.detectAndDescribe(image)



    def detectAndDescribe(self, image):
        kps, des = self.surfFeature.detectAndCompute(image, None)
        kps = np.float32([kp.pt for kp in kps])
        return kps, des

    def stitch(self, image):
        print "stitch called"
        # cv2.imshow("StitchImage", utils.image_resize(image, height = 400))
        # cv2.waitKey()
        # cv2.destroyWindow("StitchImage")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rightKps, rightDescriptor = self.detectAndDescribe(gray)
        H = self.getHomography(rightKps, rightDescriptor)
        if H is None:
            return None
        leftImageShape = self.leftImage.shape
        result = cv2.warpPerspective(image, H, (leftImageShape[1] + image.shape[1], image.shape[0]))
        result[0:leftImageShape[0], 0:leftImageShape[1]] = self.leftImage

    #     Update leftImage stats
        print("Stitch done!")
        self.leftImage = result
        self.leftKps = rightKps
        self.leftDescriptor = rightDescriptor
        # cv2.imshow("StitchImage", utils.image_resize(result, height = 400))
        # cv2.waitKey()
        # cv2.destroyWindow("StitchImage")
        return

    def getHomography(self, rightKps, rightDescriptor):
        rawMatches = self.matcher.knnMatch(self.leftDescriptor, rightDescriptor, 2)
        matches = []

        for m in rawMatches:
            if(len(m)==2 and m[0].distance < m[1].distance*self.ratio):
                matches.append((m[0].trainIdx, m[0].queryIdx))

        if(len(matches) >=4):
            # print(matches)
            ptsB = np.float32([self.leftKps[i] for (_, i) in matches])
            ptsA = np.float32([rightKps[i] for (i, _) in matches])

            # ptsB = H*ptsA
            H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, self.reprojThresh)
            return H

        return None

    def saveImage(self):
        cv2.imshow("Image", utils.image_resize(self.leftImage, width = 900))
        cv2.waitKey()
        # cv2.imwrite("stitchedImage.jpg", self.leftImage)


def main():
    ap = argparse.ArgumentParser("Stitching a panorama from video input")
    ap.add_argument("-v", "--video", required = False, help = "Path to Video file (Defaults to camera input)")

    args = vars(ap.parse_args())
    if not args.get("video", False):
        camera = cv2.VideoCapture(0)
    else:
        camera = cv2.VideoCapture(args["video"])
    for i in range(10):
        grabbed, frame = camera.read()
    if grabbed:
        imageStitcher = SurfStitcher(frame)
        # showImage(frame)
        # cv2.imwrite("Image1.jpg", frame)

    for i in range(4):
        grabbed, frame = camera.read()
        if not grabbed:
            break

        imageStitcher.stitch(frame)
        # showImage(frame)
        # cv2.imwrite("Image2.jpg", frame)
        cv2.imshow("Video", utils.image_resize(frame, height = 500))

        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break
            # imageStitcher.stitch(frame)

    imageStitcher.saveImage()
    cv2.destroyAllWindows()
    camera.release()

def showImage(image):
    cv2.imshow("Image", utils.image_resize(image, height = 400))
    cv2.waitKey()
    cv2.destroyWindow("Image")
if __name__ == "__main__":
    main()

