"""
This class defines a RL environment for a pick and place task with a UR3 robot.
This environment is defined by its center (both cartesian and angular coordinates), the total length of its x and y axis
and other parameters
"""

import random
from BlobDetector.BlobDetector import BlobDetector
from ai_manager.ImageController import ImageController
from math import floor

class Env1:

    X_LENGTH = 0.24  # Total length of the x axis environment in meters
    Y_LENGTH = 0.167  # Total length of the y axis environment in meters

    CAMERA_SECURITY_MARGIN = 0.03  # As the camera is really close to the gripping point, it needs  a security marging
    X_LIMIT = X_LENGTH - CAMERA_SECURITY_MARGIN  # Robot boundaries of movement in axis X
    Y_LIMIT = Y_LENGTH - CAMERA_SECURITY_MARGIN  # Robot boundaries of movement in axis Y

    FALL_SECURITY_MARGIN = 0.08  # As the camera is really close to the gripping point, it needs  a security marging
    X_LIMIT_FALL = X_LENGTH - FALL_SECURITY_MARGIN  # Robot boundaries of movement in axis X
    Y_LIMIT_FALL = Y_LENGTH - FALL_SECURITY_MARGIN  # Robot boundaries of movement in axis Y

    CARTESIAN_CENTER = [-0.32688, 0.09797, 0.3]  # Cartesian center of the RL environment
    ANGULAR_CENTER = [2.5307274, -1.5184364, 1.553343, -1.6755161,
                      -1.553343, -0.6250663916217249]  # Angular center of the RL environment
    PLACE_CARTESIAN_CENTER = [-0.32485, -0.08828, CARTESIAN_CENTER[2]]  # Cartesian center of the place box
    ANGULAR_PICTURE_PLACE = [1.615200161933899, -1.235102955495016, 0.739865779876709, -1.2438910643206995,
                             -1.5095704237567347, -0.06187755266298467]

    PICK_DISTANCE = 0.01  # Distance to the object when the robot is performing the pick and place action
    ACTION_DISTANCE = 0.02  # Distance to the object when the robot is performing the pick and place action

    ENV_BOUNDS_TOLERANCE = 0

class Env2:

    X_LENGTH = 0.24  # Total length of the x axis environment in meters
    Y_LENGTH = 0.165  # Total length of the y axis environment in meters

    CAMERA_SECURITY_MARGIN = 0.03  # As the camera is really close to the gripping point, it needs  a security marging
    X_LIMIT = X_LENGTH - CAMERA_SECURITY_MARGIN  # Robot boundaries of movement in axis X
    Y_LIMIT = Y_LENGTH - CAMERA_SECURITY_MARGIN  # Robot boundaries of movement in axis Y

    FALL_SECURITY_MARGIN = 0.08  # As the camera is really close to the gripping point, it needs  a security marging
    X_LIMIT_FALL = X_LENGTH - FALL_SECURITY_MARGIN  # Robot boundaries of movement in axis X
    Y_LIMIT_FALL = Y_LENGTH - FALL_SECURITY_MARGIN  # Robot boundaries of movement in axis Y

    CARTESIAN_CENTER = [-0.32485, -0.08828, 0.3]  # Cartesian center of the RL environment
    ANGULAR_CENTER = [3.0194196, -1.4660766, 1.48353, -1.6231562,
                      -1.5707963, -0.3250663916217249]  # Angular center of the RL environment
    PLACE_CARTESIAN_CENTER = [-0.32688, 0.09797, CARTESIAN_CENTER[2]]  # Cartesian center of the place box
    ANGULAR_PICTURE_PLACE = [1.615200161933899, -1.235102955495016, 0.739865779876709, -1.2438910643206995,
                             -1.5095704237567347, -0.06187755266298467]

    PICK_DISTANCE = 0.01  # Distance to the object when the robot is performing the pick and place action
    ACTION_DISTANCE = 0.02  # Distance to the object when the robot is performing the pick and place action

    ENV_BOUNDS_TOLERANCE = 0

class Environment:

    def __init__(self, cst):
        self.cst = cst

    def generate_random_state(self, image=None, strategy='ncc'):
        """
        Calculates random coordinates inside the Relative Environment defined.
        To help the robot empty the box, the generated coordinates won't be in the center of the box, because this is
        the most reachable place of the box.

        :param strategy: strategy used to calculate random_state coordinates
        :return:
        """
        def generate_random_coordinates():
            coordinate_x = random.uniform((-self.cst.X_LIMIT + self.cst.ENV_BOUNDS_TOLERANCE) / 2,
                                          (self.cst.X_LIMIT - self.cst.ENV_BOUNDS_TOLERANCE) / 2)
            coordinate_y = random.uniform((-self.cst.Y_LIMIT + self.cst.ENV_BOUNDS_TOLERANCE) / 2,
                                          (self.cst.Y_LIMIT - self.cst.ENV_BOUNDS_TOLERANCE) / 2)
            return coordinate_x, coordinate_y

        # Random coordinates avoiding the ones in the center, which have a bigger probability of being reached by the
        # robot.
        if strategy == 'ncc' or strategy == 'non_centered_coordinates':
            coordinates_in_center = True
            while coordinates_in_center:
                coordinate_x, coordinate_y = generate_random_coordinates()
                if abs(coordinate_x) > (self.cst.X_LIMIT / 4) or abs(coordinate_y) > (self.cst.Y_LIMIT / 4):
                    coordinates_in_center = False
        elif strategy == 'optimal' and image is not None:  # Before going to a random state, we check that there are pieces in this place
            blob_detector = BlobDetector(x_length=self.cst.X_LENGTH, y_length=self.cst.Y_LENGTH, columns=4, rows=4)
            optimal_quadrant = blob_detector.find_optimal_quadrant(image)
            optimal_point = blob_detector.quadrants_center[optimal_quadrant]

            coordinate_x = optimal_point[0] * 0.056
            coordinate_y = optimal_point[1] * 0.056
        else: # Totally random coordinates
            coordinate_x, coordinate_y = generate_random_coordinates()

        return [coordinate_x, coordinate_y]

    def generate_random_state_fall(self, image=None, strategy='ncc'):
        """
        Calculates random coordinates inside the Relative Environment defined.
        To help the robot empty the box, the generated coordinates won't be in the center of the box, because this is
        the most reachable place of the box.

        :param strategy: strategy used to calculate random_state coordinates
        :return:
        """

        def generate_random_coordinates_fall():
            coordinate_x_fall = random.uniform((-self.cst.X_LIMIT_FALL + self.cst.ENV_BOUNDS_TOLERANCE) / 2,
                                          (self.cst.X_LIMIT_FALL -self.cst.ENV_BOUNDS_TOLERANCE) / 2)
            coordinate_y_fall = random.uniform((-self.cst.Y_LIMIT_FALL + self.cst.ENV_BOUNDS_TOLERANCE) / 2,
                                          (self.cst.Y_LIMIT_FALL - self.cst.ENV_BOUNDS_TOLERANCE) / 2)
            return coordinate_x_fall, coordinate_y_fall

        # Random coordinates avoiding the ones in the center, which have a bigger probability of being reached by the
        # robot.
        if strategy == 'ncc' or strategy == 'non_centered_coordinates':
            coordinates_in_center = True
            while coordinates_in_center:
                coordinate_x_fall, coordinate_y_fall = generate_random_coordinates_fall()
                if abs(coordinate_x_fall) > (self.cst.X_LIMIT_FALL / 4) or abs(coordinate_y_fall) > (self.cst.Y_LIMIT_FALL / 4):
                    coordinates_in_center = False
        elif strategy == 'optimal' and image is not None:  # Before going to a random state, we check that there are pieces in this place
            blob_detector = BlobDetector(x_length=self.cst.X_LENGTH, y_length=self.cst.Y_LENGTH, columns=4,
                                         rows=4)
            optimal_quadrant = blob_detector.find_optimal_quadrant(image)
            optimal_point = blob_detector.quadrants_center[optimal_quadrant]

            coordinate_x_fall = optimal_point[0] * 0.056
            coordinate_y_fall = optimal_point[1] * 0.056
        else:  # Totally random coordinates
            coordinate_x_fall, coordinate_y_fall = generate_random_coordinates_fall()

        return [coordinate_x_fall, coordinate_y_fall]


    def get_relative_corner(self, corner):
        """
        Function used to calculate the coordinates of the environment corners relative to the CARTESIAN_CENTER.

        :param corner: it indicates the corner that we want to get the coordinates. It' s composed by two letters
        that indicate the cardinality. For example: ne indicates North-East corner
        :return coordinate_x, coordinate_y:
        """
        if corner == 'sw' or corner == 'ws':
            return -self.cst.X_LIMIT / 2, self.cst.Y_LIMIT / 2
        if corner == 'nw' or corner == 'wn':
            return self.cst.X_LIMIT / 2, self.cst.Y_LIMIT / 2
        if corner == 'ne' or corner == 'en':
            return self.cst.X_LIMIT / 2, -self.cst.Y_LIMIT / 2
        if corner == 'se' or corner == 'es':
            return -self.cst.X_LIMIT / 2, -self.cst.Y_LIMIT / 2


    def is_terminal_state(self, coordinates, object_gripped):
        """
        Function used to determine if the current state of the robot is terminal or not
        :return: bool
        """
        def get_limits(length): return length / 2 - self.cst.ENV_BOUNDS_TOLERANCE  # functon to calculate the box boundaries
        x_limit_reached = abs(coordinates[0]) > get_limits(self.cst.X_LIMIT)  # x boundary reached
        y_limit_reached = abs(coordinates[1]) > get_limits(self.cst.Y_LIMIT)  # y boundary reached
        return x_limit_reached or y_limit_reached or object_gripped # If one or both or the boundaries are reached --> terminal state
