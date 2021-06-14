import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from torchvision.transforms.functional import crop
from torchvision import transforms
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
import os
import time
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import queue
from threading import Thread
#########################################################################"
import cv2


from BlobDetector.camera_calibration.PerspectiveCalibration import PerspectiveCalibration
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import moveit_commander
import rospy
import rospkg
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from moveit_commander.conversions import pose_to_list
from robot2 import Robot

from ImageProcessing.srv import GetBestPrediction
from ur_icam_description.robotUR import RobotUR
from std_msgs.msg import Bool, Int32MultiArray


# nom service: get_best_prediction

rospy.init_node("move_robot_to_predict")

dPoint = PerspectiveCalibration()
dPoint.setup_camera()
myRobot = RobotUR()

robot2 = Robot(Env_cam_bas)
print("1")

def move_robot_to_take_pic():
    # création d'une position initiale
    # coordonées de la position de décalage (pour que le robot de soit pas sur la prochaine photo)
    init_x = -30.312 / 100
    init_y = 27.68 / 100
    init_z = 0.3

    # calcul du déplacement à effectuer pour passer du point courant au point de décalage
    move_init_x = init_x - robot2.robot.get_current_pose().pose.position.x
    move_init_y = init_y - robot2.robot.get_current_pose().pose.position.y
    move_init_z = init_z - robot2.robot.get_current_pose().pose.position.z

    # mouvement vers le point de décalage
    robot2.relative_move(move_init_x, move_init_y, move_init_z)

    pose_init = Pose()
    pose_init.position.x = robot2.robot.get_current_pose().pose.position.x
    pose_init.position.y = robot2.robot.get_current_pose().pose.position.y
    pose_init.position.z = 0.3
    pose_init.orientation.x = -0.4952562586434166
    pose_init.orientation.y = 0.49864161678730506
    pose_init.orientation.z = 0.5082803126324129
    pose_init.orientation.w = 0.497723718615624
    myRobot.go_to_pose_goal(pose_init)
#
def move_robot_client():

    rospy.wait_for_service('get_best_prediction')
    try:
        service_add = rospy.ServiceProxy('get_best_prediction', GetBestPrediction)
        resp = service_add()
        coord = [resp.x, resp.y]
        return coord
    except rospy.ServiceExeption as e:
        print("Service call failes: %s"%e)

if __name__=="__main__":

    while True:

        coord_pixel = move_robot_client()

        xyz = dPoint.from_2d_to_3d(coord_pixel)

        goal_x = -xyz[0][0] / 100 + 0.7 / 100
        goal_y = -xyz[1][0] / 100 + 2.2 / 100

        # calcul du déplacement à effectuer pour passer du point courant au point cible
        move_x = goal_x - robot2.robot.get_current_pose().pose.position.x
        move_y = goal_y - robot2.robot.get_current_pose().pose.position.y

        # mouvement vers le point cible
        robot2.relative_move(move_x, move_y, 0)

        object_gripped = robot2.take_pick(no_rotation=True)

        move_robot_to_take_pic()

        robot2.send_gripper_message(False)





#
# Validate_movement_pub = rospy.Publisher("validate_movement", Bool, queue_size=10)
#
# # Validate_movement_pub.publish(False)
#
# while True:
#
#     coord_pixel_msg = rospy.wait_for_message('pixel_coordinates', Int32MultiArray).data
#
#     print(coord_pixel_msg)
#
#     xyz = dPoint.from_2d_to_3d(coord_pixel_msg)
#
#     goal_x = -xyz[0][0] / 100 + 0.7 / 100
#     goal_y = -xyz[1][0] / 100 + 2.2 / 100
#
#     # calcul du déplacement à effectuer pour passer du point courant au point cible
#     move_x = goal_x - robot2.robot.get_current_pose().pose.position.x
#     move_y = goal_y - robot2.robot.get_current_pose().pose.position.y
#
#     # mouvement vers le point cible
#     robot2.relative_move(move_x, move_y, 0)
#
#     object_gripped = robot2.take_pick(no_rotation=True)
#     move_robot_to_take_pic()
#     robot2.send_gripper_message(False)
#
#     Validate_movement_pub.publish(True)
#
#
#
#
# print(msg)
