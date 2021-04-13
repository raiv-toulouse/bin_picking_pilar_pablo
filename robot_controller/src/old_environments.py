# Environment 1

X_LENGTH = 0.24  # Total length of the x axis environment in meters
Y_LENGTH = 0.34  # Total length of the y axis environment in meters

CAMERA_SECURITY_MARGIN = 0.03  # As the camera is really close to the gripping point, it needs  a security marging
X_LIMIT = X_LENGTH - CAMERA_SECURITY_MARGIN  # Robot boundaries of movement in axis X
Y_LIMIT = Y_LENGTH - CAMERA_SECURITY_MARGIN  # Robot boundaries of movement in axis Y

FALL_SECURITY_MARGIN = 0.08  # As the camera is really close to the gripping point, it needs  a security marging
X_LIMIT_FALL = X_LENGTH - FALL_SECURITY_MARGIN  # Robot boundaries of movement in axis X
Y_LIMIT_FALL = Y_LENGTH - FALL_SECURITY_MARGIN  # Robot boundaries of movement in axis Y

CARTESIAN_CENTER = [-0.32775, -0.00901, 0.34]  # Cartesian center of the RL environment
ANGULAR_CENTER = [2.7776150703430176, -1.5684941450702112, 1.299912452697754, -1.3755658308612269,
                  -1.5422008673297327, -0.3250663916217249]  # Angular center of the RL environment
PLACE_CARTESIAN_CENTER = [0, 0.25, CARTESIAN_CENTER[2]]  # Cartesian center of the place box
ANGULAR_PICTURE_PLACE = [1.615200161933899, -1.235102955495016, 0.739865779876709, -1.2438910643206995,
                         -1.5095704237567347, -0.06187755266298467]

PICK_DISTANCE = 0.01  # Distance to the object when the robot is performing the pick and place action
ACTION_DISTANCE = 0.02  # Distance to the object when the robot is performing the pick and place action

ENV_BOUNDS_TOLERANCE = 0

# Environment 2

X_LENGTH = 0.34  # Total length of the x axis environment in meters
Y_LENGTH = 0.24  # Total length of the y axis environment in meters

CAMERA_SECURITY_MARGIN = 0.03  # As the camera is really close to the gripping point, it needs  a security marging
X_LIMIT = X_LENGTH - CAMERA_SECURITY_MARGIN  # Robot boundaries of movement in axis X
Y_LIMIT = Y_LENGTH - CAMERA_SECURITY_MARGIN  # Robot boundaries of movement in axis Y

FALL_SECURITY_MARGIN = 0.08  # As the camera is really close to the gripping point, it needs  a security marging
X_LIMIT_FALL = X_LENGTH - FALL_SECURITY_MARGIN  # Robot boundaries of movement in axis X
Y_LIMIT_FALL = Y_LENGTH - FALL_SECURITY_MARGIN  # Robot boundaries of movement in axis Y

CARTESIAN_CENTER = [0.02866, 0.26812, 0.34]  # Cartesian center of the RL environment
ANGULAR_CENTER = [0.97738438, -1.7104227, 1.2740904, -1.1519173,
                  -1.4835299, -0.6]  # Angular center of the RL environment
PLACE_CARTESIAN_CENTER = [-0.02866, -0.26812, CARTESIAN_CENTER[2]]  # Cartesian center of the place box
ANGULAR_PICTURE_PLACE = [1.615200161933899, -1.235102955495016, 0.739865779876709, -1.2438910643206995,
                         -1.5095704237567347, -0.06187755266298467]

PICK_DISTANCE = 0.01  # Distance to the object when the robot is performing the pick and place action
ACTION_DISTANCE = 0.02  # Distance to the object when the robot is performing the pick and place action

ENV_BOUNDS_TOLERANCE = 0