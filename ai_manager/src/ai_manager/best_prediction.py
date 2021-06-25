#!/usr/bin/env python

import rospy
import random
import math
from ai_manager.srv import GetBestPrediction, GetBestPredictionResponse
from ai_manager.msg import Prediction
from ai_manager.msg import ListOfPredictions
from PIL import Image as PILImage
from sensor_msgs.msg import Image
import torch
from torchvision.transforms.functional import crop
from torchvision import transforms
from ImageProcessing.CNN import CNN
import cv2
import numpy as np

INVALIDATION_RADIUS = 150  # When a prediction is selected, we invalidate all the previous predictions in this radius
IMAGE_TOPIC = '/usb_cam2/image_raw'
CROP_WIDTH = CROP_HEIGHT = 56 # Size of cropped image
MODEL_NAME = '/home/student1/catkin_ws_noetic/src/bin_picking/ai_manager/src/ImageProcessing/model/resnet18/Cylindresx4000.ckpt'
# min and max HSV values for color thresholding for object recognition (max S and V = 255, max H = 180)

LOW_H = 0
LOW_S = 0
LOW_V = 40
HIGH_H = 180
HIGH_S = 18
HIGH_V = 230

# LOW_H = 0
# LOW_S = 0
# LOW_V = 0
# HIGH_H = 128
# HIGH_S = 128
# HIGH_V = 128



class NodeBestPrediction:
    """
    This node is both a service and a publisher.
    * Service best_prediction_service : use a GetBestPrediction message (no input message and a ListOfPredictions output message)
    When this service is called, return the current best prediction and invalidate all the predictions in its neighborhood.
    * Publisher : publish on the 'predictions' topic a ListOfPredictions message

    How to run?
    * roslaunch usb_cam usb_cam-test.launch   (to provide a /usb_cam/image_raw topic)
    * python visu_prediction.py   (to view the success/fail prediction points on the image)
    * python best_prediction.py   (to provide a /predictions and /new_image topics)
    * rosservice call /best_prediction_service  (to get the current best prediction. Il load a new image and invalidate the points in the picking zone)

    """
    def __init__(self):
        self.predictions = []
        msg_list_pred = ListOfPredictions()
        rospy.init_node('best_prediction')
        rospy.Service('best_prediction_service', GetBestPrediction, self._get_best_prediction)
        pub = rospy.Publisher('predictions', ListOfPredictions, queue_size=10)
        self.pub_image = rospy.Publisher('new_image', Image, queue_size=10)
        self._load_model()
        self.picking_point = None # No picking point yet
        self.image, self.width, self.height = self._get_image()  # Get the first image to process
        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: self._crop_xy(img)),
            transforms.Resize(size=256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        while not rospy.is_shutdown():
            x = random.randint(int(CROP_WIDTH / 2), self.width - int(CROP_WIDTH / 2)) # for a (x,y) centered crop to fit in the image
            y = random.randint(int(CROP_HEIGHT / 2), self.height - int(CROP_HEIGHT / 2))
            if self._ok_to_compute_proba(x, y):
                msg = Prediction()
                msg.x = x
                msg.y = y
                msg.proba = random.random() #self._predict(x, y)
                self.predictions.append(msg)
                msg_list_pred.predictions = self.predictions
                pub.publish(msg_list_pred)
            rospy.sleep(0.01)

    def _ok_to_compute_proba(self, x, y):
        """ Return True if this (x,y) point is a good candidate i.e. is on an object and not in the picking zone"""
        # Test if inside the picking zone
        if self.picking_point and self._distance(self.picking_point[0], self.picking_point[1] ,x ,y) < INVALIDATION_RADIUS:
            return False  # This point is inside the picking zone
        # Test if on an object
        if self.object_image[y, x] == 0:
            return False  # Not on an object (it is on a black pixel)
        # test if it's on the box zone
        if x < 295 or x > 1000 or y < 260 or y > 690:
            return False

        return True

    def _compute_object_image(self,pil_image):
        opencv_rgb_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        frame_HSV = cv2.cvtColor(opencv_rgb_image, cv2.COLOR_BGR2HSV)
        self.object_image = cv2.inRange(frame_HSV, (LOW_H, LOW_S, LOW_V), (HIGH_H, HIGH_S, HIGH_V))
        cv2.imwrite("test.png", self.object_image)

    def _load_model(self):
        """
        Load a pretrained 'resnet18' model from a CKPT filename, freezed for inference
        :return:
        """
        self.model = CNN(backbone='resnet18')
        self.inference_model = self.model.load_from_checkpoint(MODEL_NAME)   #  Load the selected model
        self.inference_model.freeze()

    def _predict(self, x, y):
        """ Predict probability and class for a cropped image at (x,y) """
        self.predict_center_x = x
        self.predict_center_y = y
        img = self.transform(self.image)  # Get the cropped transformed image
        img = img.unsqueeze(0)  # To have a 4-dim tensor ([nb_of_images, channels, w, h])
        features, preds = self._evaluate_image(img, self.inference_model)
        pred = torch.exp(preds)
        return pred[0][1].item()  # Return the success probability

    @torch.no_grad()
    def _evaluate_image(self, image, model):
        features, prediction = model(image)
        return features.detach().numpy(), prediction.detach()

    def _crop_xy(self, image):
        """ Crop image at position (predict_center_x,predict_center_y) and with size (WIDTH,HEIGHT) """
        return crop(image, self.predict_center_y - CROP_HEIGHT / 2, self.predict_center_x - CROP_WIDTH / 2, CROP_HEIGHT, CROP_WIDTH)  # top, left, height, width

    def _get_image(self):
        """
        Recover an image from the IMAGE_TOPIC topic
        :return: an RGB image
        """
        msg_image = rospy.wait_for_message(IMAGE_TOPIC, Image)
        self.pub_image.publish(msg_image)
        pil_image = self._to_pil(msg_image)
        self._compute_object_image(pil_image)
        return pil_image, msg_image.width, msg_image.height

    def _to_pil(self, msg, display=False):
        size = (msg.width, msg.height)  # Image size
        img = PILImage.frombytes('RGB', size, msg.data)  # sensor_msg to Image
        return img

    def _get_best_prediction(self, req):
        """ best_prediction_service service callback which return a Prediction message (the best one)"""
        # Get a new image to process
        self.image, width, height = self._get_image()
        # Find best prediction
        best_pred = self.predictions[0]
        for pred in self.predictions:
            if pred.proba > best_pred.proba:
                best_pred = pred
        self._invalidate_neighborhood(best_pred.x, best_pred.y)
        self.picking_point = (best_pred.x, best_pred.y)
        return GetBestPredictionResponse(best_pred)

    def _invalidate_neighborhood(self, x, y):
        """
        Invalidate (remove) all the predictions in a circle of radius INVALIDATION_RADIUS centered in (x,y)
        :param x:
        :param y:
        :return:
        """
        self.predictions = [pred for pred in self.predictions if self._distance(pred.x, pred.y, x, y) > INVALIDATION_RADIUS]
        
    def _distance(self, p1_x, p1_y, p2_x, p2_y):
        """ Compute the distance between 2 points """
        dx = (p1_x - p2_x) * (p1_x - p2_x)
        dy = (p1_y - p2_y) * (p1_y - p2_y)
        return math.sqrt(dx + dy)

if __name__ == '__main__':
    try:
        n = NodeBestPrediction()
    except rospy.ROSInterruptException:
        pass