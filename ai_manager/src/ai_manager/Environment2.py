"""
This class defines a RL environment for a pick and place task with a UR3 robot.
This environment is defined by its center (both cartesian and angular coordinates), the total length of its x and y axis
and other parameters
"""

import random
from BlobDetector.BlobDetector import BlobDetector
from ai_manager.ImageController import ImageController
from math import floor


class Environment2:
    X_LENGTH = 0.34  # Total length of the x axis environment in meters
    Y_LENGTH = 0.24  # Total length of the y axis environment in meters

    CAMERA_SECURITY_MARGIN = 0.03  # As the camera is really close to the gripping point, it needs  a security marging
    X_LIMIT = X_LENGTH - CAMERA_SECURITY_MARGIN  # Robot boundaries of movement in axis X
    Y_LIMIT = Y_LENGTH - CAMERA_SECURITY_MARGIN  # Robot boundaries of movement in axis Y

    CARTESIAN_CENTER = [0.03562, 0.28812, 0.34]  # Cartesian center of the RL environment
    ANGULAR_CENTER = [1.2217305, -1.6580628, 1.6231562, -1.5184364,
                      -1.5009832, 0.1396263]  # Angular center of the RL environment
    PLACE_CARTESIAN_CENTER = [0.03562, -0.28812, CARTESIAN_CENTER[2]]  # Cartesian center of the place box
    ANGULAR_PICTURE_PLACE = [1.615200161933899, -1.235102955495016, 0.739865779876709, -1.2438910643206995, -1.5095704237567347, -0.06187755266298467]

    PICK_DISTANCE = 0.01  # Distance to the object when the robot is performing the pick and place action
    ACTION_DISTANCE = 0.02  # Distance to the object when the robot is performing the pick and place action

    ENV_BOUNDS_TOLERANCE = 0

    @staticmethod
    def generate_random_state(image=None, strategy='ncc'):
        """
        Calculates random coordinates inside the Relative Environment defined.
        To help the robot empty the box, the generated coordinates won't be in the center of the box, because this is
        the most reachable place of the box.

        :param strategy: strategy used to calculate random_state coordinates
        :return:
        """
        def generate_random_coordinates():
            coordinate_x = random.uniform((-Environment.X_LIMIT + Environment.ENV_BOUNDS_TOLERANCE) / 2,
                                          (Environment.X_LIMIT - Environment.ENV_BOUNDS_TOLERANCE) / 2)
            coordinate_y = random.uniform((-Environment.Y_LIMIT + Environment.ENV_BOUNDS_TOLERANCE) / 2,
                                          (Environment.Y_LIMIT - Environment.ENV_BOUNDS_TOLERANCE) / 2)
            return coordinate_x, coordinate_y

        # Random coordinates avoiding the ones in the center, which have a bigger probability of being reached by the
        # robot.
        if strategy == 'ncc' or strategy == 'non_centered_coordinates':
            coordinates_in_center = True
            while coordinates_in_center:
                coordinate_x, coordinate_y = generate_random_coordinates()
                if abs(coordinate_x) > (Environment2.X_LIMIT / 4) or abs(coordinate_y) > (Environment2.Y_LIMIT / 4):
                    coordinates_in_center = False
        elif strategy == 'optimal' and image is not None:  # Before going to a random state, we check that there are pieces in this place
            blob_detector = BlobDetector(x_length=Environment2.X_LENGTH, y_length=Environment2.Y_LENGTH, columns=4, rows=4)
            optimal_quadrant = blob_detector.find_optimal_quadrant(image)
            optimal_point = blob_detector.quadrants_center[optimal_quadrant]

            coordinate_x = optimal_point[0] * 0.056
            coordinate_y = optimal_point[1] * 0.056
        else: # Totally random coordinates
            coordinate_x, coordinate_y = generate_random_coordinates()

        return [coordinate_x, coordinate_y]

    @staticmethod
    def get_relative_corner(corner):
        """
        Function used to calculate the coordinates of the environment corners relative to the CARTESIAN_CENTER.

        :param corner: it indicates the corner that we want to get the coordinates. It' s composed by two letters
        that indicate the cardinality. For example: ne indicates North-East corner
        :return coordinate_x, coordinate_y:
        """
        if corner == 'sw' or corner == 'ws':
            return -Environment2.X_LIMIT / 2, Environment2.Y_LIMIT / 2
        if corner == 'nw' or corner == 'wn':
            return Environment2.X_LIMIT / 2, Environment2.Y_LIMIT / 2
        if corner == 'ne' or corner == 'en':
            return Environment2.X_LIMIT / 2, -Environment2.Y_LIMIT / 2
        if corner == 'se' or corner == 'es':
            return -Environment2.X_LIMIT / 2, -Environment2.Y_LIMIT / 2

    @staticmethod
    def is_terminal_state(coordinates, object_gripped):
        """
        Function used to determine if the current state of the robot is terminal or not
        :return: bool
        """
        def get_limits(length): return length / 2 - Environment2.ENV_BOUNDS_TOLERANCE  # functon to calculate the box boundaries
        x_limit_reached = abs(coordinates[0]) > get_limits(Environment2.X_LIMIT)  # x boundary reached
        y_limit_reached = abs(coordinates[1]) > get_limits(Environment2.Y_LIMIT)  # y boundary reached
        return x_limit_reached or y_limit_reached or object_gripped # If one or both or the boundaries are reached --> terminal state
