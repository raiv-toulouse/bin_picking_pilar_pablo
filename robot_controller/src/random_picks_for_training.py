#!/usr/bin/env python
# coding: utf-8

"""
Used to get picking images from random position. The images go to 'success' or 'fail'  folders, depending if the gripper succeed or not

- We need to establish a connection to the robot with the following comand:
roslaunch ur_robot_driver ur3_bringup.launch robot_ip:=10.31.56.102 kinematics_config:=${HOME}/Calibration/ur3_calibration.yaml

- Then, we need to activate moveit server:
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

from ai_manager.Environment import Environment
from ai_manager.Environment import Env1
from ai_manager.Environment import Env2

if __name__ == '__main__':
    number_box = int(input("entrez le num de la boite: "))


    change_box = False
    limit_piece = int(input("entrez le nombre de pièce: "))

    compt_object = 0
    from ai_manager.ImageController import ImageController

    rospy.init_node('random_picks_for_training')
    image_controller = ImageController(path='/home/student1/ros_pictures', image_topic='/usb_cam/image_raw')
    robot = Robot(Env1)

    robot.relative_move(0, 0, 0.2)
    if number_box == 1:
        robot.go_to_initial_pose()
    elif number_box == 2:
        robot.change_environment(Env2)
        robot.go_to_initial_pose()

    ind_image = 0

    while True:
        print("1")

        if number_box == 1:
            if change_box == True:
                robot.change_environment(Env1)
                robot.go_to_initial_pose()
            # robot.take_random_state()
            img, width, height = image_controller.get_image()
            object_gripped = robot.take_pick(no_rotation=True)
            if object_gripped == True:
                compt_object += 1
                robot.relative_move(0, 0, 0.08)
                robot.relative_move(0, 0.2, 0)
                # robot2.go_to_initial_pose()
                robot.change_environment(Env2)
                robot.take_random_state_fall()
                robot.send_gripper_message(False)  # We turn off the gripper

                if compt_object < limit_piece:
                    robot.change_environment(Env1)
                    robot.relative_move(0, -0.2, 0)
                    robot.go_to_initial_pose()


        elif number_box == 2:
            if change_box == True:
                robot.change_environment(Env2)
                robot.go_to_initial_pose()
            robot.take_random_state()
            img, width, height = image_controller.get_image()

            object_gripped = robot.take_pick(no_rotation=True)
            if object_gripped == True:
                compt_object += 1
                robot.relative_move(0, 0, 0.08)
                robot.relative_move(0, -0.2, 0)
                # robot.go_to_initial_pose()
                robot.change_environment(Env1)
                robot.take_random_state_fall()
                robot.send_gripper_message(False)  # We turn off the gripper

                if compt_object < limit_piece:

                    robot.relative_move(0, 0.2, 0)
                    robot.change_environment(Env2)
                    robot.go_to_initial_pose()
        print("2")

        if compt_object == limit_piece:
            change_box = True
        else:
            change_box = False

        if change_box == True:

            if number_box == 1:

                number_box = 2
                compt_object = 0
            else:

                number_box = 1
                compt_object = 0
        print("3")
        print("actuellement box ", number_box, " objets attrapés: ", compt_object)

        image_controller.record_image(img, object_gripped)
        rospy.loginfo("Image #{}, object gripped: {}".format(ind_image, object_gripped == True))
        ind_image += 1
