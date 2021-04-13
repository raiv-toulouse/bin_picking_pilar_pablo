#!/usr/bin/env python
# coding: utf-8

import rospy
from robot import Robot
from robot2 import Robot2
from ai_manager.Environment import Environment
from ai_manager.Environment2 import Environment2
from std_msgs.msg import Bool
import time
from ai_manager.ImageController import ImageController


def take_pic():
    img, width, height = image_controller.get_image()
    return img

def save_pic(img, led):
    image_controller.record_image(img, led)

if __name__ == '__main__':

    rospy.init_node('random_training_multiflash')
    image_controller = ImageController(path='/home/student1/ros_pictures', image_topic='/usb_cam/image_raw')
    img_list = []

    number_box = 1
    change_box = False
    compt_object = 0
    from ai_manager.ImageController import ImageController

    rospy.init_node('random_training_multiflash')
    image_controller = ImageController(path='/home/student1/ros_pictures', image_topic='/usb_cam/image_raw')
    robot = Robot()
    robot2 = Robot2()
    robot.go_to_initial_pose()
    ind_image = 0

    while True:

        for i in range(1, 5):
            i = str(i)  # convert to string
            pub = rospy.Publisher('led' + i + '_on_off', Bool, queue_size=50)
            time.sleep(0.6)

            # turn on the led
            led_on_off = True
            pub.publish(led_on_off)

            print("allumé")
            time.sleep(0.1)

            # take picture
            img = take_pic()
            img_list.append(img)

            # turn off the led
            led_on_off = False
            pub.publish(led_on_off)

            time.sleep(0.1)

            # save the 4 pictures
        for j in img_list:
            save_pic(j, True)

        break
'''
        if number_box == 1:
            if change_box == True:
                robot.go_to_initial_pose()
            robot.take_random_state()
            
            object_gripped = robot.take_pick(no_rotation=True)
            if object_gripped == False:
                compt_object += 1


        elif number_box == 2:
            if change_box == True:
                robot2.go_to_initial_pose()
            robot2.take_random_state()
            img, width, height = image_controller.get_image()

            object_gripped = robot2.take_pick(no_rotation=True)
            if object_gripped == False:
                compt_object += 1
        print("2")

        if compt_object == 1:
            change_box = True
        else:
            change_box = False

        if change_box == True:

            if number_box == 1:
                robot.relative_move(0,0,0.1)
                number_box = 2
                compt_object = 0
            else:
                robot2.relative_move(0,0,0.1)
                number_box = 1
                compt_object = 0
        print("3")
        print("actuellement box ", number_box, " objets attrapés: ", compt_object)

        image_controller.record_image(img, object_gripped)
        rospy.loginfo("Image #{}, object gripped: {}".format(ind_image, object_gripped == True))
        ind_image += 1  '''
