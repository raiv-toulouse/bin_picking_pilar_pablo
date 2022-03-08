#!/usr/bin/env python
from math import pi

import rospy
from ur_icam_description.robotUR import RobotUR

if __name__ == '__main__':
    try:
        # Start the ROS node
        rospy.init_node('node_test_collision_moveit')
        myRobot = RobotUR()
        initialJointsState = [0, -pi / 2, pi / 2, -pi / 2, -pi / 2, 0]
        myRobot.go_to_joint_state(initialJointsState)
        # On ajoute les obstacles
        myRobot.add_obstacle_box('left_cube', size=(1, 1, 1), position=(0.75, 0.5, 0))
        myRobot.add_obstacle_box('table', size=(2, 2, 0.03), position=(0, 0, 0))
        myRobot.add_obstacle_box('wall', size=(0.5, 0.5, 2), position=(0.6, -0.4, 0))

        # myRobot.add_obstacle_table('table_cafe', size=(0.3, 0.01, 0.4), position=(0.6, 0, 0.4))
    except rospy.ROSInterruptException:
        print ("Program interrupted before completion")
