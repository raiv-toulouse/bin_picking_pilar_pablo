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

class ImagePreprocessing():

    # TODO: ELECCION DE PARAMETROS - PIPELINE (CONSTRUCTOR)
    # defines the network
    def __init__(self):
        super(ImagePreprocessing, self).__init__()
        self.display = True


    # BLURRING
    # type_blurring = ['averaging','gaussian','median','bilateral']
    def blurring(image, type_blurring, display):

        if type_blurring == 'averaging':
            blurred = cv2.blur(image, (5, 5))
        elif type_blurring == 'gaussian':
            blurred = cv2.GaussianBlur(image, (5, 5), 0)
        elif type_blurring == 'median':
            blurred = cv2.medianBlur(image, 5)
        elif type_blurring == 'bilateral':
            blurred = cv2.bilateralFilter(image, 9, 75, 75)
        else:
            blurred = image
        if display:
            plt.imshow(blurred, 'gray')
            plt.title("Blurred Image")
            plt.xticks([]), plt.yticks([])
            plt.show()
        return blurred

    # BRIGHTNESS AND CONTRAST
    def brightness(image, alpha, beta):
        '''
        The α parameter will modify how the levels spread.
        If α<1, the color levels will be compressed and the result will be an image with less contrast.
        Increasing (/ decreasing) the β value will add (/ subtract) a constant value to every pixel.
        Pixel values outside of the [0 ; 255] range will be saturated
        '''
        # new_image = np.zeros(image.shape, image.dtype)
        # alpha = 1.0 # Simple contrast control
        # beta = 0    # Simple brightness control
        # Initialize values
        # Do the operation new_image(i,j) = alpha*image(i,j) + beta
        new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        cv2.imshow('Original Image', image)
        cv2.imshow('New Image', new_image)
        # Wait until user press some key
        cv2.waitKey()
        cv2.destroyAllWindows()
        return new_image

    # CLOSING
    def closing(image):
        closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        return closing

    # DILATION
    def dilation(image, display):
        kernel = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(image, kernel, iterations=1)
        if display:
            # show the image
            cv2.imshow("dilation", dilation)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return dilation

    # EQUALIZATION - ADAPTIVE EQUALIZATION
    def adaptive_histogram_equalization(image, display):
        '''
         Image is divided into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV).
         Then each of these blocks are histogram equalized as usual.
         So in a small area, histogram would confine to a small region (unless there is noise).
         If noise is there, it will be amplified. To avoid this, contrast limiting is applied.
         If any histogram bin is above the specified contrast limit (by default 40 in OpenCV),
        '''
        cl_ahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        dts = cl_ahe.apply(image)
        cv2.imshow('Source image', image)
        cv2.imshow('Equalized Image', dst)
        cv2.waitKey()
        cv2.destroyAllWindows()

        if display:
            # Display image Histogram
            # plt.hist(dst.ravel(), bins=256, histtype='step', color='black');
            f = plt.figure(figsize=(20, 10))
            f.add_subplot(2, 2, 1)
            plt.hist(dst.ravel(), bins=256, histtype='step', color='black')
            plt.title("Equalization")
            f.add_subplot(2, 2, 2)
            plt.hist(image.ravel(), bins=256, histtype='step', color='black')
            plt.title("Original")
            # plt.xticks([]),plt.yticks([])
            plt.show()
        return dts

    # EQUALIZATION - HISTOGRAM EQUALIZATION
    def histogram_equalization(image, display):

        dst = cv2.equalizeHist(image)
        cv2.imshow('Source image', image)
        cv2.imshow('Equalized Image', dst)
        cv2.waitKey()
        cv2.destroyAllWindows()

        if display:
            # Display image Histogram
            # plt.hist(dst.ravel(), bins=256, histtype='step', color='black');
            f = plt.figure(figsize=(20, 10))
            f.add_subplot(2, 2, 1)
            plt.hist(dst.ravel(), bins=256, histtype='step', color='black')
            plt.title("Equalization")
            f.add_subplot(2, 2, 2)
            plt.hist(image.ravel(), bins=256, histtype='step', color='black')
            plt.title("Original")
            # plt.xticks([]),plt.yticks([])
            plt.show()

        return dst
    #EROSION
    def erosion(image, display):
        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(image, kernel, iterations=1)

        if display:
            # show the image
            cv2.imshow("erosion", erosion)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return erosion

    # OPENING
    def opening(image):
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        return opening

    # RESIZING
    def resize(image, heigth, display):
        # we need to keep the image aspect ratio
        r = heigth / image.shape[1]
        dim = (heigth, int(image.shape[0] * r))

        # perform the actual resizing of the image and show it
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        # if we want to the show the result of the image
        if display:
            cv2.imshow("resized", resized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return resized

    # TRANSFORMING
    def convert_to_gray(image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return gray

    # Convert to RGB image
    def convert_to_RGB(image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return rgb_image


    # THRESHOLD
    # type_threshold = ['binary','binary_inv','trunc','tozero','tozero_inv']
    def threshold(image, type_threshold, display):
        if type_threshold == 'binary':
            thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

        elif type_threshold == 'binary_inv':
            thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

        elif type_threshold == 'trunc':
            thresh = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)

        elif type_threshold == 'tozero':
            thresh = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)

        elif type_threshold == 'tozero_inv':
            thresh = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

        if display:
            plt.imshow(thresh, 'gray')
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
            plt.show()

        return thresh

    # type_adaptive_threshold = ['adaptive_mean', 'adaptive_gaussian']
    def adaptive_threshold(image, type_adaptive_threshold, display):

        if type_adaptive_threshold == 'adaptative_mean':
            adaptive_thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        elif type_adaptive_threshold == 'adaptative_gaussian':
            adaptive_thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        if display:
            plt.imshow(adaptive_thresh, 'gray')
            plt.title("Adaptive Threshold")
            plt.xticks([]), plt.yticks([])
            plt.show()

        return adaptive_thresh

    # TODO: PREPARAR UN METODO DE DATA AUGMENTATION

    # ROTATE
    def rotate(self, image, angle=90, scale=1.0):
        '''
        Rotate the image
        :param image: image to be processed
        :param angle: Rotation angle in degrees. Positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the top-left corner).
        :param scale: Isotropic scale factor.
        '''
        w = image.shape[1]
        h = image.shape[0]
        # rotate matrix
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        # rotate
        image = cv2.warpAffine(image, M, (w, h))
        return image



