#!/usr/bin/env python
# coding: utf-8

import os
import time

import rospy
from PIL import Image as PILImage
from sensor_msgs.msg import Image

"""
This class is used to manage sensor_msgs Images.
"""

class ImageController:
    def __init__(self, path=os.path.dirname(os.path.realpath(__file__)), image_topic='/usb_cam/image_raw'):
        self.ind_saved_images = 0  # Index which will tell us the number of images that have been saved
        self.success_path = "{}/success".format(path)  # Path where the images are going to be saved
        self.fail_path = "{}/fail".format(path)  # Path where the images are going to be saved
        self.image_topic = image_topic

        # If it does not exist, we create the path folder in our workspace
        try:
            os.stat(self.success_path)
        except:
            os.mkdir(self.success_path)

        # If it does not exist, we create the path folder in our workspace
        try:
            os.stat(self.fail_path)
        except:
            os.mkdir(self.fail_path)

    def get_image(self):
        msg = rospy.wait_for_message(self.image_topic, Image)

        return self.to_pil(msg), msg.width, msg.height

    def record_image(self, img, success):
        path = self.success_path if success else self.fail_path  # The path were we want to save the image is

        image_path = '{}/img{}.png'.format(  # Saving image
            path,  # Path
            time.time())  # FIFO queue

        img.save(image_path)

        self.ind_saved_images += 1  # Index increment

    def to_pil(self, msg, display=False):
        size = (msg.width, msg.height)  # Image size
        img = PILImage.frombytes('RGB', size, msg.data)  # sensor_msg to Image
        return img

if __name__ == '__main__':
    rospy.init_node('image_recorder')  # ROS node initialization
    image_controller = ImageController(path='/home/student1/ros_pictures', image_topic='/usb_cam2/image_raw')
    img, width, height = image_controller.get_image()
    image_controller.record_image(img, True)