#!/usr/bin/env python
# coding: utf-8

"""
Used to get picking images from random position. The images go to 'success' or 'fail'  folders, depending if the gripper succeed or not

- We need to connect the camera and the nodes
roslaunch ur_icam_description webcam.launch

- We need to establish a connection to the robot with the following comand:
roslaunch ur_robot_driver ur3_bringup.launch robot_ip:=10.31.56.102 kinematics_config:=${HOME}/Calibration/ur3_calibration.yaml

- Then, we ned to activate moovit server:
roslaunch ur3_moveit_config ur3_moveit_planning_execution.launch

- Activate the talker
rosrun ai_manager main_controller.py

- Activate the node
rosrun robot_controller random_picks_for_training.py

"""

import rospy
from robot import Robot


if __name__ == '__main__':
    rospy.init_node('random_picks_for_training')
    robot = Robot()
    robot.go_to_initial_pose()
    ind_image = 0
    while True:
        robot.take_random_state()
        object_gripped = robot.take_pick()
        rospy.loginfo("Image #{}".format(ind_image))
        ind_image += 1
