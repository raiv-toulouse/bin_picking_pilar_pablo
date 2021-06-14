#!/usr/bin/env python


import rospy
import random

if __name__ == '__main__':
    try:
        rospy.init_node('compute_prediction')
        rospy.set_param('list_of_predictions', [])
        while not rospy.is_shutdown():
            liste = rospy.get_param("/list_of_predictions")
            proba = random.random()
            x = random.randint(0,640)
            y = random.randint(0,480)
            prediction = [proba, x, y]
            liste.append(prediction)
            rospy.set_param('list_of_predictions', liste)
            rospy.sleep(0.2)
    except rospy.ROSInterruptException:
        pass