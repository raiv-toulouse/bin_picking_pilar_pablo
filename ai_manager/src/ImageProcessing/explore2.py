import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from ImageProcessing.ImageModel import ImageModel
from torchvision.transforms.functional import crop
from torchvision import transforms
from PIL import Image
import os
import time
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


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
from ai_manager.Environment import Environment
from ai_manager.Environment import Env_cam_bas

from ur_icam_description.robotUR import RobotUR


# global variables
#image_path = './Image_point/2021-05-07-143556.jpg'
#image_coordinates = []

# Création d'un objet de la classe PerspectiveCalibration

dPoint = PerspectiveCalibration()
dPoint.setup_camera()

robot2 = Robot(Env_cam_bas)


matplotlib.use('Qt5Agg')

WIDTH = HEIGHT = 224  # Size of cropped image

# def imshow(images, title=None, pil_image = False):
#     """Imshow for Tensor."""
#     if pil_image:
#         inp = images
#     else:
#         img_grid = torchvision.utils.make_grid(images).cpu()
#         inp = img_grid.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     plt.imshow(inp)
#     if title is not None:
#         plt.title(title)
#     plt.pause(2)


class MainWindow(QWidget):
    """
    Load an image and a CNN model from a CKPT file and display the prediction for some sub-images at some specific points
    """

    def __init__(self):
        super().__init__()
        uic.loadUi("explore_ihm2.ui",self) #needs the canvas.py file in the current directory
        self.btn_change_image.clicked.connect(self._change_image)
        self.btn_pick.clicked.connect(self.predict)
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
        self._change_image()

    def _load_model(self):
        """ Load a new model """
        fname = QFileDialog.getOpenFileName(self, 'Open model file', '.', "Model files (*.ckpt)", options=QFileDialog.DontUseNativeDialog)
        if fname[0]:
            #model_name = self.image_model._find_name_of_best_model()
            model_name = os.path.basename(fname[0])
            self.inference_model = self.image_model.load_model(model_name)  # Load the selected model
            self.lbl_model_name.setText(model_name)

    def _change_image(self):
        """ Load a new image """
        fname = QFileDialog.getOpenFileName(self, 'Open image file', '.',"Image files (*.jpg *.gif *.png)", options=QFileDialog.DontUseNativeDialog)
        if fname[0]:
            self.lbl_image_name.setText(os.path.basename(fname[0]))
            self._set_image(fname[0])

    def _set_image(self, filename):
        self.canvas.set_image(filename)
        self.image = Image.open(filename)

    def _crop_xy(self, image):
        """ Crop image at position (predict_center_x,predict_center_y) and with size (WIDTH,HEIGHT) """
        return crop(image, self.predict_center_y - HEIGHT/2, self.predict_center_x - WIDTH/2, HEIGHT, WIDTH)  # top, left, height, width

    def predict(self, x, y):
        """ Predict probability and class for a cropped image at (x,y) """
        self.predict_center_x = x
        self.predict_center_y = y
        img = self.transform(self.image)  # Get the cropped transformed image
        #imshow(img)
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
        for x in range(half_width, im_width -half_width, steps):
            for y in range(half_height, im_height - half_height, steps):
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
        all_preds = self._compute_all_preds()  # [ [x, y, tensor([[prob_fail, proba_success]])], ...]
        all_preds.sort(key=lambda pred: pred[2][0][1].item(), reverse=True)
        self.canvas.all_preds = [all_preds[0]]
        self.canvas.repaint()
        #print("Best x", all_preds[0][0]) #ok
        #print("Best y", all_preds[0][1]) #ok
        image_coord=[all_preds[0][0],all_preds[0][1]]
        #print(image_coord) #ok

        xyz = dPoint.from_2d_to_3d(image_coord)
        print(xyz) #ok
        goal_x = -xyz[0][0] / 100
        goal_y = -xyz[1][0] / 100

        # calcul du déplacement à effectuer pour passer du point courant au point cible
        move_x = goal_x - robot2.robot.get_current_pose().pose.position.x
        move_y = goal_y - robot2.robot.get_current_pose().pose.position.y

        # mouvement vers le point cible
        robot2.relative_move(move_x, move_y, 0)




if __name__ == '__main__':

    app = QApplication(sys.argv)
    gui = MainWindow()
    gui.show()

    sys.exit(app.exec_())