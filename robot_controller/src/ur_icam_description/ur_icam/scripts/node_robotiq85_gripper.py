#!/usr/bin/python
#
# Send a value to change the opening of the Robotiq gripper using an action
#

import argparse
import rospy
from ur_icam_description.gripper import Robotiq85Gripper

if __name__ == '__main__':
    try:
        # Get the angle from the command line
        parser = argparse.ArgumentParser()
        parser.add_argument("--value", type=float, default="0.2",
                            help="Value betwewen 0.0 (open) and 0.8 (closed)")
        args = parser.parse_args()
        gripper_value = args.value
        # Start the ROS node
        rospy.init_node('gripper_command')
        myGripper = Robotiq85Gripper()
        # Set the value to the gripper
        myGripper.close(gripper_value)
    except rospy.ROSInterruptException:
        print ("Program interrupted before completion")
