from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import torch
import sys
from PyQt5 import uic
from ImageProcessing.ImageModel import ImageModel
from torchvision.transforms.functional import crop
from torchvision import transforms
import os
import time
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import queue
import datetime
from BlobDetector.camera_calibration.PerspectiveCalibration import PerspectiveCalibration
#########################################################################"
import cv2
import threading
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from BlobDetector.camera_calibration.PerspectiveCalibration import PerspectiveCalibration
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import moveit_commander
import rospy
import rospkg
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from moveit_commander.conversions import pose_to_list
from robot2 import Robot
from ai_manager.Environment import Environment
from ai_manager.Environment import Env_cam_bas
from ai_manager.ImageController import ImageController
import random
from ur_icam_description.robotUR import RobotUR
WIDTH = HEIGHT = 56
dPoint = PerspectiveCalibration()
dPoint.setup_camera()
robot2 = Robot(Env_cam_bas)
rospy.init_node('explore2')
myRobot = RobotUR()
image_controller = ImageController(image_topic='/usb_cam2/image_raw')
class Canvas(QWidget):

    def __init__(self,parent):
        super().__init__(parent)
        self.parent = parent
        self.pressed = self.moving = False
        self.previous_image = None
        self.all_preds = None

    def set_image(self,filename):
        self.image = QImage(filename)
        self.setMinimumSize(self.image.width(), self.image.height())
        self.previous_image = None
        self.all_preds = None
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            if self.previous_image:  # A new click, we restore the previous image without the rectangle
                self.image = self.previous_image
                self.setMinimumSize(self.image.width(), self.image.height())
            self.pressed = True
            self.center = event.pos()
            self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:

            pos = event.pos()
            print(pos)

            # enregistrement du crop au click

            # # on prend la photo
            # img, width, height = image_controller.get_image()
            #
            # # préparation de la variable de sauvegarde (nom du fichier, dossier de sauvegarde...)
            # image_path = '{}/img{}.png'.format(  # Saving image
            #     "{}/success".format(
            #         '/home/student1/catkin_ws_noetic/src/bin_picking/ai_manager/src/ImageProcessing/image_camHaute/Update_images'),
            #     # Path
            #     "update")  # FIFO queue
            #
            # # sauvegarde de la photo
            # img.save(image_path)
            #
            # # chemin d'accès à la photo prise
            # path = r'/home/student1/catkin_ws_noetic/src/bin_picking/ai_manager/src/ImageProcessing/image_camHaute/Update_images/success/imgupdate.png'
            #
            # # chargement de la photo avec OpenCV
            # frame = cv2.imread(path)
            # center = [pos.x() + 112, pos.y() + 112]
            # h = 224
            # w = 224
            # y = pos.y() - 112
            # x = pos.x() - 112
            # crop = frame[y:y + h, x:x + w]
            #
            # cv2.imshow("crop", crop)
            # cv2.waitKey(1000)

            image_coord = [pos.x(), pos.y()]
            xyz = dPoint.from_2d_to_3d(image_coord)
            print(xyz)  # ok
            goal_x = -xyz[0][0] / 100
            goal_y = -xyz[1][0] / 100

            # calcul du déplacement à effectuer pour passer du point courant au point cible
            move_x = goal_x - robot2.robot.get_current_pose().pose.position.x
            move_y = goal_y - robot2.robot.get_current_pose().pose.position.y

            # mouvement vers le point cible
            robot2.relative_move(move_x, move_y, 0)

            # Lancement de l'action de prise
            object_gripped = robot2.take_pick(no_rotation=True)

            self.parent.move_robot_to_take_pic()
            robot2.send_gripper_message(False)

            self.update()
            self.parent._change_image()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.RightButton:
            self.moving = True
            self.center = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self.previous_image = self.image.copy()
            qp = QPainter(self.image)
            self._draw_rectangle(qp)
            self.pressed = self.moving = False
            self.update()

    def paintEvent(self, event):
        qp = QPainter(self)
        rect = event.rect()
        qp.drawImage(rect, self.image, rect)
        if self.moving or self.pressed:
            self._draw_rectangle(qp)
        if self.all_preds:
            self._draw_pred(qp)


    def _draw_rectangle(self, qp):
        if self.parent.inference_model:  # A model exists, we can do inference
            x = self.center.x() - WIDTH / 2
            y = self.center.y()-HEIGHT/2
            qp.setRenderHint(QPainter.Antialiasing)
            qp.setPen(QPen(Qt.blue, 5))
            qp.drawPoint(self.center)
            pred = self.parent.predict(self.center.x(), self.center.y())  # calculate the prediction wih CNN
            prob, cl = self._compute_prob_and_class(pred)
            fail_or_success = 'Fail' if cl.item()==0 else 'Success'
            text = f'{fail_or_success} : {prob:.1f}%'
            qp.setPen(Qt.black)
            qp.setFont(QFont('Decorative', 14))
            qp.drawText(x, y, text)
            if cl.item() == 1:  # Success
                qp.setPen(QPen(Qt.green, 1, Qt.DashLine))
            else:  # Fail
                qp.setPen(QPen(Qt.red, 1, Qt.DashLine))
            qp.drawRect(x, y, WIDTH, HEIGHT)

    def _draw_pred(self, qp):
        threshold = self.parent.sb_threshold.value()
        for (x, y , pred) in self.all_preds:
            tensor_prob,tensor_cl = torch.max(pred, 1)
            if tensor_cl.item()==0 or (tensor_cl.item()==1 and tensor_prob.item()*100 < threshold):  # Fail
                qp.setPen(QPen(Qt.red, 5))
            else:
                qp.setPen(QPen(Qt.green, 5))
            qp.drawPoint(x, y)
            qp.setPen(Qt.black)
            qp.setFont(QFont('Decorative', 8))
            prob, cl = self._compute_prob_and_class(pred)
            text = f'{prob:.1f}%'
            qp.drawText(x, y, text)

    def _compute_prob_and_class(self, pred):
        prob, cl = torch.max(pred, 1)
        if cl.item() == 0:  # Fail
            prob = 100 * (1 - prob.item())
        else:  # Success
            prob = 100 * prob.item()
        return prob, cl




