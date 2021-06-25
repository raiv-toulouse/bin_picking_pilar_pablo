# import libraries
import cv2 #OpenCV
import imutils #functions to make basic image processing functions such as translation, rotation, resizing, skeletonization, displaying Matplotlib images, sorting contours, detecting edges, and much more easier with OpenCV
import skimage
import random
import os
import numpy as np
from matplotlib import pyplot as plt
from imutils import contours
from skimage.feature import canny
from scipy import ndimage as ndi
from skimage import img_as_float, img_as_ubyte
from skimage import exposure
from skimage import feature

class EdgeDetector():
    def __init__(self):
        super(EdgeDetector, self).__init__()

    # HARRIS CORNER DETECTION
    def harris_corner_detection(image, display):
        '''
        OpenCV has the function cv2.cornerHarris() for this purpose. Its arguments are :

        img - Input image, it should be grayscale and float32 type.
        blockSize - It is the size of neighbourhood considered for corner detection
        ksize - Aperture parameter of Sobel derivative used.
        k - Harris detector free parameter in the equation.
        '''
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)

        # result is dilated for marking the corners, not important
        dst = cv2.dilate(dst, None)

        # Threshold for an optimal value, it may vary depending on the image.
        ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
        dst = np.uint8(dst)

        # find centroids
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

        # define the criteria to stop and refine the corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

        if display:
            # Now draw them
            res = np.hstack((centroids, corners))
            res = np.int0(res)
            image[res[:, 1], res[:, 0]] = [0, 0, 255]
            image[res[:, 3], res[:, 2]] = [0, 255, 0]
            # show the image
            cv2.imshow("dst", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # SHI-TOMASI CORNER DETECTOR
    def shi_tomasi_corner_detection(image, n_corners=20, quality_level=0.01, euclidean_distance=10):
        '''
        OpenCV has a function, cv2.goodFeaturesToTrack().
        It finds N strongest corners in the image by Shi-Tomasi method (or Harris Corner Detection, if you specify it).
        As usual, image should be a grayscale image.
        Then you specify number of corners you want to find.
        Then you specify the quality level, which is a value between 0-1,
        which denotes the minimum quality of corner below which everyone is rejected.
        Then we provide the minimum euclidean distance between corners detected.
        '''
        img = cv2.imread('one_block.jpg', 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, n_corners, quality_level, euclidean_distance)
        corners = np.int0(corners)
        for i in corners:
            x, y = i.ravel()
            cv2.circle(img, (x, y), 10, 255, -1)
        plt.imshow(img), plt.show()



