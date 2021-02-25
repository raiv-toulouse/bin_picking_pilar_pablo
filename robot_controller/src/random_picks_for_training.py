#!/usr/bin/env python
# coding: utf-8

"""
Used to get picking images from random position. The images go to 'success' or 'fail'  folders, depending if the gripper succeed or not

- We need to establish a connection to the robot with the following comand:
roslaunch ur_robot_driver ur3_bringup.launch robot_ip:=10.31.56.102 kinematics_config:=${HOME}/Calibration/ur3_calibration.yaml

- Then, we ned to activate moveit server:
roslaunch ur3_moveit_config ur3_moveit_planning_execution.launch

- Information from Arduino
rosrun rosserial_arduino serial_node.py _port:=/dev/ttyACM0

- We need to connect the to cameras
roslaunch usb_cam usb_2_cameras.launch

- Activate the node
rosrun robot_controller random_picks_for_training.py

"""

import rospy
from robot import Robot


if __name__ == '__main__':
    from ai_manager.src.ai_manager.ImageController import ImageController
    rospy.init_node('random_picks_for_training')
    image_controller = ImageController(path='/home/student1/ros_pictures', image_topic='/usb_cam/image_raw')
    robot = Robot()
    robot.go_to_initial_pose()
    ind_image = 0
    while True:
        robot.take_random_state()
        img, width, height = image_controller.get_image()
        object_gripped = robot.take_pick(no_rotation=True)
        image_controller.record_image(img, object_gripped)
        rospy.loginfo("Image #{}, object gripped: {}".format(ind_image,object_gripped==True))
        ind_image += 1
