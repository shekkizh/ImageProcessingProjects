__author__ = 'Charlie'
import cv2
import numpy as np
import imutils
import argparse


class ImageStitcher:
    def stitch(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False):
        # unpack the images, then detect keypoints and extract
        # local invariant descriptors from them
        (imageB, imageA) = images
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        # match features between the two images
        M = self.matchKeypoints(kpsA, kpsB,
                                featuresA, featuresB, ratio, reprojThresh)

        # if the match is None, then there aren't enough matched
        # keypoints to create a panorama
        if M is None:
            return None
        # apply a perspective warp to stitch the images
        # together
        (matches, H, status) = M
        result = cv2.warpPerspective(imageA, H,
                                     (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        # check to see if the keypoint matches should be visualized
        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,
                                   status)

            # return a tuple of the stitched image and the
            # visualization
            return (result, vis)

        # return the stitched image
        return result

    def detectAndDescribe(self, image):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        descriptor = cv2.xfeatures2d.SIFT_create()
        (kps, features) = descriptor.detectAndCompute(image, None)

        # convert the keypoints from KeyPoint objects to NumPy
        # arrays
        kps = np.float32([kp.pt for kp in kps])  # return a tuple of keypoints and features
        return kps, features


    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
                       ratio, reprojThresh):
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

                # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                             reprojThresh)

            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, H, status)

        # otherwise, no homograpy could be computed
        return None


    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

            # return the visualization
        return vis

class SurfStitcher:
    def __init__(self, image, ratio = 0.75, reprojThresh = 4.0):
        self.leftImage = image
        self.ratio = ratio
        self.reprojThresh = reprojThresh

        self.surfFeature = cv2.xfeatures2d.SURF_create(500, extended = False, upright = True)
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
        # cv2.imshow("Image", utils.image_resize(self.leftImage, width = 900))
        # cv2.waitKey()
        cv2.imwrite("stitchedImage.jpg", self.leftImage)



def main():
    ap = argparse.ArgumentParser("Image Stitching")
    ap.add_argument("--first", required=True, help="Path to first image to be stiched")
    ap.add_argument("--second", required=True, help="Path to second image to be stitched")
    args = vars(ap.parse_args())

    imageA = cv2.imread(args["first"])
    imageB = cv2.imread(args["second"])
    imageA = imutils.resize(imageA, width=400)
    imageB = imutils.resize(imageB, width=400)

    # stitch the images together to create a panorama
    # stitcher = ImageStitcher()
    # (result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)
    stitcher = SurfStitcher(imageA)
    stitcher.stitch(imageB)
    # stitcher.saveImage()

    # show the images
    # cv2.imshow("Image A", imageA)
    # cv2.imshow("Image B", imageB)
    # cv2.imshow("Keypoint Matches", vis)
    cv2.imshow("Result", stitcher.leftImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
