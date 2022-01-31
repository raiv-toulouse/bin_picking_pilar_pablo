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

- launch program:
python random_picks_birdview.py (in robot_controller/src)

"""
from PIL import Image as PILImage
from sensor_msgs.msg import Image
import rospy
from datetime import datetime
import time
import argparse
import imutils
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
from updateFrame import webcamImageGetter



if __name__ == '__main__':

    # création d'un objet de la classe ImageController relatif à la prise de photo (dans ai_manager)
    image_controller = ImageController(image_topic='/usb_cam/image_raw')

    # création d'un objet de la classe PrepectiveCalibration (dans Blobdetector)
    dPoint = PerspectiveCalibration()
    dPoint.setup_camera()

    # création d'un objet de la classe RobotUR (dans ur_icam)
    myRobot = RobotUR()

    # création d'un objet de la classe Robot (dans robot_controller)
    robot = Robot(Env_cam_bas)

    # initialisation du noeud robotUr
    rospy.init_node('robotUR')

    # on remonte de 10cm au cas ou le robot serait trop bas
    robot.relative_move(0, 0, 0.1)

    # création d'une position initiale
    init_x = -28.195 / 100
    init_y = 19.085 / 100
    init_z = 0.25

    # calcul du déplacement à effectuer pour passer du point courant à la position initiale
    move_init_x = init_x - robot.robot.get_current_pose().pose.position.x
    move_init_y = init_y - robot.robot.get_current_pose().pose.position.y
    move_init_z = init_z - robot.robot.get_current_pose().pose.position.z

    # mouvement vers la position initiale
    robot.relative_move(move_init_x, move_init_y, move_init_z)

    pose_goal = Pose()
    pose_goal.position.x = robot.robot.get_current_pose().pose.position.x
    pose_goal.position.y = robot.robot.get_current_pose().pose.position.y
    pose_goal.position.z = 0.25
    pose_goal.orientation.x = 0
    pose_goal.orientation.y = 1
    pose_goal.orientation.z = 0
    pose_goal.orientation.w = 0

    PointList = []

    myRobot.go_to_pose_goal(pose_goal)
    for i in range(279-112, 879):
        for j in range(259-112, 638):
            PointList.append([i, j])

    PointList = PointList[::1000]


    # boucle du programme de prise d'image
    while True:

        # on prend la photo
        img, width, height = image_controller.get_image()

        # préparation de la variable de sauvegarde (nom du fichier, dossier de sauvegarde...)
        image_path = '{}/img{}.png'.format(  # Saving image
            "{}".format('/home/student1/catkin_ws_noetic/src/bin_picking/ai_manager/src/ImageProcessing/updating_image'),  # Path
            "update")  # FIFO queue

        # sauvegarde de la photo
        img.save(image_path)

        # chemin d'accès à la photo prise
        path = r'/home/student1/catkin_ws_noetic/src/bin_picking/ai_manager/src/ImageProcessing/updating_image/imgupdate.png'

        # chargement de la photo avec OpenCV
        frame = cv2.imread(path)

        for i in PointList:
            center_coordinates = (i[0]+112, i[1]+112)
            radius = 3
            color = (0, 0, 255)
            thickness = -1
            frame = cv2.circle(frame, center_coordinates, radius, color, thickness)

        # taille du crop
        h = 224
        w = 224



        # création d'un point de coordonnées pixel random
        a = random.randrange(0,len(PointList))

        x = PointList[a][0]
        y = PointList[a][1]


        # calcul du centre de l'image crop
        pixel_random = [x + 112, y + 112]
        frame = cv2.circle(frame, (x+112,y+112), radius, (0,255,0), thickness)

        # réalisation du crop
        crop = frame[y:y + h, x:x + w]




        cv2.imshow("crop", frame)
        cv2.waitKey(1000)

        # transposition des coordonnées pixel du point en coordonnées réelles dans le repère du robot
        xyz = dPoint.from_2d_to_3d(pixel_random)
        xyz[-1] = -20
        print('Les coordonnées en pixels X et Y sont: ', x, ';', y, '\n')
        print('Les coordonnées réelles sont:', xyz)



        # calcul des coordonnées cibles (en m)
        goal_x = -(xyz[0][0] / 100 ) #- 1.7/ 100)
        goal_y = -(xyz[1][0] / 100 ) #- 0.4 / 100

        # calcul du déplacement à effectuer pour passer du point courant au point cible

        move_x = goal_x - robot.robot.get_current_pose().pose.position.x
        move_y = goal_y - robot.robot.get_current_pose().pose.position.y


        print('STARTING RELATIVE MOVE')
        # mouvement vers le point cible
        robot.relative_move(move_x, move_y, 0)
        print('RELATIVE MOVE FINISHED')
        # Lancement de l'action de prise
        object_gripped = robot.take_pick(no_rotation=True)

        # Si un objet est attrapé
        if object_gripped == True:

            # création d'un point de lâcher aléatoire
            # release_goal_x = -random.randrange(40,45)/100
            # release_goal_y = -random.randrange(-9, 9)/100
            release_goal_x = -0.4
            release_goal_y = -0.2

            # calcul du déplacement à effectuer pour passer du point courant au point de lâcher
            move_release_x = release_goal_x - robot.robot.get_current_pose().pose.position.x
            move_release_y = release_goal_y - robot.robot.get_current_pose().pose.position.y

            # mouvement vers le point de lâcher
            robot.relative_move(move_release_x, move_release_y, 0)

            # on éteind la pompe
            robot.send_gripper_message(False)

            # on sauvegarde l'image crop dans le dossier success
            cv2.imwrite(os.path.join('/home/student1/ros_pictures/images_random_picks/success',
                                     'success'  + str(datetime.now()) + '.jpg'), crop)

        else:

            # on éteind la pompe
            robot.send_gripper_message(False)

            # on sauvegarde l'image crop dans le dossier fail
            cv2.imwrite(os.path.join('/home/student1/ros_pictures/images_random_picks/fail',
                                     'fail'  + str(datetime.now()) + '.jpg'), crop)


        print("target reached")
        print('Length of PointList :' , len(PointList))

        del PointList[a]

        # calcul du déplacement à effectuer pour passer du point courant au point de décalage
        move_init_x = init_x - robot.robot.get_current_pose().pose.position.x
        move_init_y = init_y - robot.robot.get_current_pose().pose.position.y

        # mouvement vers le point de décalage
        robot.relative_move(move_init_x, move_init_y, 0)
        myRobot.go_to_pose_goal(pose_goal)

        cv2.destroyAllWindows()






