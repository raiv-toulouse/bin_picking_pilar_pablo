# Python code for Blob Detection
import os

import imutils
import numpy as np
import cv2
from math import ceil
import matplotlib.pyplot as plt
from .camera_calibration.PerspectiveCalibration import PerspectiveCalibration


class BlobDetector:
    def __init__(self, x_length, y_length, columns=3, rows=3, draw=False):
        self.draw = draw
        # Setup camera with camera calibration
        self.pc = PerspectiveCalibration(draw=self.draw)
        self.pc.setup_camera()
        # Get image
        current_path = os.path.dirname(os.path.realpath(__file__))
        self.image_path = os.path.join(current_path, 'blob_images/img1610361325.5.png')
        self.image = cv2.imread(self.image_path)
        # self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.quadrants, self.quadrants_center = self.get_quadrants(x_length, y_length, columns, rows)

    def get_quadrants(self, x_length, y_length, columns, rows):
        # Conversion from centimeters to logical coordinates length (0.054 is the length of square side)
        x_length = x_length / 0.054
        y_length = y_length / 0.054

        # Size of each quadrant in coordinates
        quadrant_width = y_length / columns
        quadrant_height = x_length / rows

        n_quadrants = columns * rows  # number of quadrants

        # Creation of all the points that forms each quadrant
        points = [
            [x*quadrant_height-x_length/2,
             y*quadrant_width-y_length/2,
             0]
            for x in range(rows + 1) for y in range(columns + 1)]

        # Conversion of these points from logical coordinates to image pixels
        point_pixels = []
        for point in points:
            point_pixels.append(self.pc.from_3d_to_2d(self.image, point)[0][0])

        # Creation of the quadrants, each quadrant is formed by 4 points
        quadrants = []
        quadrants_center = []
        for i in range(n_quadrants):
            row = int(ceil((i + 1) / (columns)) - 1)

            # Creation of the quadrants
            quadrants.append([
                point_pixels[i + row],
                point_pixels[i + row + 1],
                point_pixels[i + row + columns + 2],
                point_pixels[i + row + columns + 1]
            ])

            # Center of the quadrants
            quadrants_center.append(np.subtract(
                points[i + row],
                np.subtract(points[i + row], points[i + row + columns + 2]) / 2
            ))

        return quadrants, quadrants_center

    def check_pixel_quadrant(self, cx, cy):
        def limit(a1, a2, a, b1, b2):
            return (((b2 - b1) / (a2 - a1)) * (a - a1)) + b1

        for idx, quadrant in enumerate(self.quadrants):
            check_1 = cy >= limit(quadrant[0][0], quadrant[1][0], cx, quadrant[0][1], quadrant[1][1])
            check_2 = cx >= limit(quadrant[0][1], quadrant[3][1], cy, quadrant[0][0], quadrant[3][0])
            check_3 = cx < limit(quadrant[1][1], quadrant[2][1], cy, quadrant[1][0], quadrant[2][0])
            check_4 = cy < limit(quadrant[3][0], quadrant[2][0], cx, quadrant[3][1], quadrant[2][1])
            if check_1 and check_2 and check_3 and check_4:
                return idx
        return None

    @staticmethod
    def _find_contours(thresh):
        # Find contours in the binary image
        contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        return contours

    def find_optimal_quadrant(self, image, draw=False):
        image = np.asarray(image)
        thresh = self._image_preprocessing(image)
        contours = self._find_contours(thresh)
        # initialize quadrant count
        index_quadrant = [0] * len(self.quadrants)
        for c in contours:
            # calculate moments for each contour
            M = cv2.moments(c)
            if M["m00"] != 0.0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(image, (cX, cY), 5, (255, 255, 255), -1)
                cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
                index = self.check_pixel_quadrant(cX, cY)
                if index is not None:
                    index_quadrant[index] += 1
            else:
                cX = 0
                cY = 0

        # display the image
        if draw or self.draw:
            # cv2.imshow("Image", image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.imshow(image)
            plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
            plt.show()
        return index_quadrant.index(max(index_quadrant))

    @staticmethod
    def _image_preprocessing(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Convert to gray-scale
        # Convert the grayscale image to binary image
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        return thresh


if __name__ == '__main__':
    current_path = os.path.dirname(os.path.realpath(__file__))
    image_path = os.path.join(current_path, 'blob_images/img1610977646.87.png')
    image = cv2.imread(image_path)
    blob_detector = BlobDetector(x_length=0.175, y_length=0.225, columns=4, rows=4)
    optimal_quadrant = blob_detector.find_optimal_quadrant(image)
    optimal_point = blob_detector.quadrants_center[optimal_quadrant]
    blob_detector.pc.from_3d_to_2d(image, optimal_point, draw=True)
