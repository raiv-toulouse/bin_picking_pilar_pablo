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
from datetime import datetime
import time
from robot import Robot
import random
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
import cv2
import os
from BlobDetector.camera_calibration.PerspectiveCalibration import PerspectiveCalibration
from ur_icam_description.robotUR import RobotUR
from ai_manager.ImageController import ImageController
from ai_manager.Environment import Environment
from ai_manager.Environment import Env_cam_bas

if __name__ == '__main__':

    # création d'un objet de la classe PrepectiveCalibration (dans Blobdetector)
    dPoint = PerspectiveCalibration()
    dPoint.setup_camera()

    # création d'un objet de la classe RobotUR (dans ur_icam)
    myRobot = RobotUR()

    # création d'un objet de la classe Robot (dans robot_controller)
    robot = Robot(Env_cam_bas)

    # initialisation du noeud robotUr
    rospy.init_node('robotUR')

    # initialisation des index pour les images success et fail
    ind_success = 0
    ind_fail = 0

    # capture de la camera
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 960)
    nom_fenetre = "webcam"

    # on remonte de 10cm au cas ou le robot serait trop bas
    robot.relative_move(0, 0, 0.1)

    # création d'une position initiale
    pose_goal = Pose()
    pose_goal.position.x = -28.195 / 100
    pose_goal.position.y = 19.085 / 100
    pose_goal.position.z = 0.3
    pose_goal.orientation.x = -0.4952562586434166
    pose_goal.orientation.y = 0.49864161678730506
    pose_goal.orientation.z = 0.5082803126324129
    pose_goal.orientation.w = 0.497723718615624

    # mouvement vers la position initiale
    myRobot.go_to_pose_goal(pose_goal)

    while True:

        # on prend une photo
        success, img = cap.read()

        # création d'un point de coordonnées pixel random
        x = random.randrange(300, 676)
        y = random.randrange(213, 460)

        # calcul du centre de l'image crop
        pixel_random = [x+112, y+112]

        # taille du crop
        h = 224
        w = 224

        # réalisation du crop
        crop = img[y:y + h, x:x + w]

        # transposition des coordonnées pixel du point en coordonnées réelles dans le repère du robot
        xyz = dPoint.from_2d_to_3d(pixel_random)

        # calcul des coordonnées cibles (en m)
        goal_x = -xyz[0][0] / 100
        goal_y = -xyz[1][0] / 100

        # calcul du déplacement à effectuer pour passer du point courant au point cible
        move_x = goal_x - robot.robot.get_current_pose().pose.position.x
        move_y = goal_y - robot.robot.get_current_pose().pose.position.y

        # mouvement vers le point cible
        robot.relative_move(move_x, move_y, 0)

        # Lancement de l'action de prise
        object_gripped = robot.take_pick(no_rotation=True)

        # si un objet est attrapé
        if object_gripped:

            # on incrémente l'index success
            ind_success = ind_success + 1

            # on éteind la pompe
            robot.send_gripper_message(False)

            # on enregistre la photo dans le dossier success
            cv2.imwrite(os.path.join('/home/student1/ros_pictures/Camera_haute/success', 'success'+str(ind_success) + str(datetime.now()) + '.jpg'), crop)

        # sinon
        else:
            # on incrémente l'index fail
            ind_fail = ind_fail + 1

            # on éteind la pompe
            robot.send_gripper_message(False)

            # on enregistre la photo dans le dossier fail
            cv2.imwrite(os.path.join('/home/student1/ros_pictures/Camera_haute/fail', 'fail' + str(ind_fail) + str(datetime.now())+ '.jpg'), crop)

        print("target reached")

        # coordonées de la position de décalage (pour que le robot de soit pas sur la prochaine photo)
        init_x = -28.195 / 100
        init_y = 19.085 / 100

        # calcul du déplacement à effectuer pour passer du point courant au point de décalage
        move_init_x = init_x - robot.robot.get_current_pose().pose.position.x
        move_init_y = init_y - robot.robot.get_current_pose().pose.position.y

        # mouvement vers le point de décalage
        robot.relative_move(move_init_x, move_init_y, 0)
