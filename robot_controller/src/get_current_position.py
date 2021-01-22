#!/usr/bin/env python
# coding: utf-8

import copy
from math import pi
import rospy
from geometry_msgs.msg import Pose

from ur_icam_description.robotUR import RobotUR

if __name__ == '__main__':
    myRobot = RobotUR()
    rospy.init_node('robotUR')

    print("============ Printing robot current pose")
    print(myRobot.get_current_pose())
    print("============ Printing robot state")
    print(myRobot.robot.get_current_state())