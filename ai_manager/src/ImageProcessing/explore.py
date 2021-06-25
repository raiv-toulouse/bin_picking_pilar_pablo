import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from ImageProcessing.ImageModel import ImageModel
from torchvision.transforms.functional import crop
from torchvision import transforms
from PIL import Image as image
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
import threading

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
from ur_icam_description.robotUR import RobotUR
from std_msgs.msg import Bool, Int32MultiArray


# global variables
#image_path = './Image_point/2021-05-07-143556.jpg'
#image_coordinates = []

# Création d'un objet de la classe PerspectiveCalibration

dPoint = PerspectiveCalibration()
dPoint.setup_camera()

robot2 = Robot(Env_cam_bas)

Pub3 = rospy.Publisher("pixel_coordinates", Int32MultiArray, queue_size=10)

rospy.init_node('explore2')
rate = rospy.Rate(0.5)

myRobot = RobotUR()

matplotlib.use('Qt5Agg')

WIDTH = HEIGHT = 56 # Size of cropped image

image_controller = ImageController(image_topic='/usb_cam2/image_raw')

mtx = np.load('/home/student1/catkin_ws_noetic/src/bin_picking/ai_manager/src/BlobDetector/camera_calibration/Camera_data/' + 'newcam_mtx.npy')
dist = np.load('/home/student1/catkin_ws_noetic/src/bin_picking/ai_manager/src/BlobDetector/camera_calibration/Camera_data/' + 'dist.npy')
new_camera_mtx = np.load('/home/student1/catkin_ws_noetic/src/bin_picking/ai_manager/src/BlobDetector/camera_calibration/Camera_data/' + 'new_camera_mtx.npy')
def imshow(images, title=None, pil_image = False):
    """Imshow for Tensor."""
    if pil_image:
        inp = images
    else:
        img_grid = torchvision.utils.make_grid(images).cpu()
        inp = img_grid.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(2)

class MainWindow(QWidget):
    """
    Load an image and a CNN model from a CKPT file and display the prediction for some sub-images at some specific points
    """

    def __init__(self):
        super().__init__()
        uic.loadUi("explore_ihm.ui",self) #needs the canvas.py file in the current directory
        self.title = 'Camera'
        self.label = QLabel(self)
        lay = QVBoxLayout()
        lay.addWidget(self.label)
        self.setLayout(lay)

        self.btn_change_image.clicked.connect(self._move_robot)
        # self.btn_pick.clicked.connect(self.predict)
        self.btn_load_model.clicked.connect(self._load_model)
        self.btn_find_best.clicked.connect(self._find_best_solution)
        self.btn_map.clicked.connect(self._compute_map)
        self.btn_find_and_pick.clicked.connect(self._find_and_pick)
        self.sb_threshold.valueChanged.connect(self._change_threshold)
        self.image_model = ImageModel(model_name='resnet18')
        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: self._crop_xy(img)),
            transforms.Resize(size=256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.inference_model = None
        self._move_robot()
        self._load_model()



    # @pyqtSlot(QImage)
    # def setImage(self, image):
    #     self.label.setPixmap(QPixmap.fromImage(image))
    #     self.th.writer.write(image)

    def _load_model(self):
        """ Load a new model """
        fname = QFileDialog.getOpenFileName(self, 'Open model file', '.', "Model files (*.ckpt)", options=QFileDialog.DontUseNativeDialog)
        if fname[0]:
            #model_name = self.image_model._find_name_of_best_model()
            model_name = os.path.basename(fname[0])
            self.inference_model = self.image_model.load_model(model_name)  # Load the selected model
            self.lbl_model_name.setText(model_name)

    def _move_robot(self):

        # """ Load a new image """
        # fname = QFileDialog.getOpenFileName(self, 'Open image file', '.', "Image files (*.jpg *.gif *.png)",
        #                                     options=QFileDialog.DontUseNativeDialog)
        # if fname[0]:
        #     self.lbl_image_name.setText(os.path.basename(fname[0]))
        #     self._set_image(fname[0])

        self.move_robot_to_take_pic()

        img, width, height = image_controller.get_image()

        # préparation de la variable de sauvegarde (nom du fichier, dossier de sauvegarde...)
        image_path = '{}/img{}.png'.format("{}".format('./'),"update")  # FIFO queue
        img.save(image_path)
        path = r'/home/student1/catkin_ws_noetic/src/bin_picking/ai_manager/src/ImageProcessing/updating_image/imgupdate.png'

        # chargement de la photo avec OpenCV
        frame = cv2.imread(path)
        dst = cv2.undistort(frame, mtx, dist, None, new_camera_mtx)
        # sauvegarde de la photo
        cv2.imwrite(
            os.path.join('/home/student1/catkin_ws_noetic/src/bin_picking/ai_manager/src/ImageProcessing/updating_image',
                         "imgupdate.png"), dst)
        fname = "imgupdate.png"

        self._set_image(fname)

        msg = True

    def _change_image(self):
        # """ Load a new image """
        # fname = QFileDialog.getOpenFileName(self, 'Open image file', '.', "Image files (*.jpg *.gif *.png)",
        #                                     options=QFileDialog.DontUseNativeDialog)
        # if fname[0]:
        #     self.lbl_image_name.setText(os.path.basename(fname[0]))
        #     self._set_image(fname[0])

        img, width, height = image_controller.get_image()

        # préparation de la variable de sauvegarde (nom du fichier, dossier de sauvegarde...)
        image_path = '{}/updating_image/img{}.png'.format("{}".format('./'), "update")  # FIFO queue
        img.save(image_path)
        path = r'/home/student1/catkin_ws_noetic/src/bin_picking/ai_manager/src/ImageProcessing/updating_image/imgupdate.png'

        # chargement de la photo avec OpenCV
        frame = cv2.imread(path)
        # Method 1 to undistort the image
        dst = cv2.undistort(frame, mtx, dist, None, new_camera_mtx)
        # sauvegarde de la photo
        cv2.imwrite(os.path.join('/home/student1/catkin_ws_noetic/src/bin_picking/ai_manager/src/ImageProcessing/updating_image',
                                 "imgupdate.png"), dst)
        fname = "imgupdate.png"

        self._set_image(fname)




    def move_robot_to_take_pic(self):



        # création d'une position initiale
        # coordonées de la position de décalage (pour que le robot de soit pas sur la prochaine photo)
        self.init_x = -30.312 / 100
        self.init_y = 27.68 / 100
        self.init_z = 0.3

        # calcul du déplacement à effectuer pour passer du point courant au point de décalage
        self.move_init_x = self.init_x - robot2.robot.get_current_pose().pose.position.x
        self.move_init_y = self.init_y - robot2.robot.get_current_pose().pose.position.y
        self.move_init_z = self.init_z - robot2.robot.get_current_pose().pose.position.z

        # mouvement vers le point de décalage
        robot2.relative_move(self.move_init_x, self.move_init_y, self.move_init_z)

        self.pose_init = Pose()
        self.pose_init.position.x = robot2.robot.get_current_pose().pose.position.x
        self.pose_init.position.y = robot2.robot.get_current_pose().pose.position.y
        self.pose_init.position.z = 0.3
        self.pose_init.orientation.x = -0.4952562586434166
        self.pose_init.orientation.y = 0.49864161678730506
        self.pose_init.orientation.z = 0.5082803126324129
        self.pose_init.orientation.w = 0.497723718615624
        myRobot.go_to_pose_goal(self.pose_init)

    def _set_image(self, filename):
        self.canvas.set_image(filename)
        self.image = image.open(filename)


    def _crop_xy(self, image):
        """ Crop image at position (predict_center_x,predict_center_y) and with size (WIDTH,HEIGHT) """
        return crop(image, self.predict_center_y - HEIGHT/2, self.predict_center_x - WIDTH/2, HEIGHT, WIDTH)  # top, left, height, width

    def predict(self, x, y):
        """ Predict probability and class for a cropped image at (x,y) """
        self.predict_center_x = x
        self.predict_center_y = y
        img = self.transform(self.image)  # Get the cropped transformed image
        # imshow(img)
        img = img.unsqueeze(0)  # To have a 4-dim tensor ([nb_of_images, channels, w, h])
        features, preds = self.image_model.evaluate_image(img, self.inference_model, False)  # No processing
        #self.lbl_result.setText(str(torch.exp(preds)))
        return torch.exp(preds)

    def _change_threshold(self):
        ''' Redraw the predictions if the threshold has been changed '''
        self.canvas.repaint()

    def _find_best_solution(self):
        """ Compute the best prediction and ask the canvas to draw it """
        all_preds = self._compute_all_preds()  # [ [x, y, tensor([[prob_fail, proba_success]])], ...]
        all_preds.sort(key=lambda pred: pred[2][0][1].item(), reverse=True)
        self.canvas.all_preds = [all_preds[0]]
        self.canvas.repaint()
        return  self.canvas.all_preds

    def _compute_map(self):
        """ Compute a list of predictions and ask the canvas to draw them"""
        all_preds = self._compute_all_preds()
        self.canvas.all_preds = all_preds
        self.canvas.repaint()

    def _compute_all_preds(self):
        """ Compute a list of predictions like :
        [ [x, y, tensor([[prob_fail, proba_success]])], ...] with x,y the center of cropped image size (WIDTH,HEIGHT)
        """
        start = time.time()
        all_preds = []
        steps = int(self.edt_nb_pixels_per_step.text())
        (im_width, im_height) = self.image.size
        half_width = int(WIDTH/2)
        half_height = int(HEIGHT/2)
        count = 0
        start_width = int(0.3*im_width+half_width)
        end_width = int(im_width -(half_width+0.2*im_width))
        start_height = int(0.25*im_height+half_height)
        end_height = int(im_height - (half_height + 0.25*im_height))
        for x in range(start_width, end_width, steps):
            for y in range(start_height, end_height, steps):
                preds = self.predict(x,y)
                all_preds.append([x, y, preds])
                count += 1
        end = time.time()
        self.lbl_result_map.setText(f'{count} inferences in {end-start:.1f} s')
        return all_preds

    def _find_and_pick(self):
        """Compute a list of predictions like _compute_all_preds, sort them like _find_best_solution
        Transform pixel to real coordinates
        Command robot"""

        for i in range (1,11):
            list = []
            all_preds = self._compute_all_preds()

            all_preds.sort(key=lambda pred: pred[2][0][1].item(), reverse=True)
            print(len(all_preds))
            self.canvas.all_preds = [all_preds[0]]
            self.canvas.repaint()

            image_coord = [all_preds[0][0], all_preds[0][1]]
            print(image_coord)

            xyz = dPoint.from_2d_to_3d(image_coord)

            goal_x = -(xyz[0][0] / 100 - 0 / 100)
            goal_y = -(xyz[1][0] / 100 - 0 / 100)

            # calcul du déplacement à effectuer pour passer du point courant au point cible
            move_x = goal_x - robot2.robot.get_current_pose().pose.position.x
            move_y = goal_y - robot2.robot.get_current_pose().pose.position.y

            # mouvement vers le point cible
            robot2.relative_move(move_x, move_y, 0)

            object_gripped = robot2.take_pick(no_rotation=True)

            self.move_robot_to_take_pic()

            robot2.send_gripper_message(False)

            self._move_robot()



if __name__ == '__main__':


    app = QApplication(sys.argv)
    gui = MainWindow()

    gui.show()

    sys.exit(app.exec_())