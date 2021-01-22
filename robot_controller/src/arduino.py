#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import Float32
from std_msgs.msg import Bool

from ur_icam_description.robotUR import RobotUR

import random

# Global variable to calibrate distance between the robot and the table
PICK_MOVEMENT_DISTANCE = 0.24

# Global variable for myRobot
MY_ROBOT = RobotUR()


def talker():
    distance_pub = rospy.Publisher('/distance', Float32)
    gripper_pub = rospy.Publisher('/object_gripped', Bool)
    rospy.init_node('arduino', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        distance = MY_ROBOT.get_current_pose().pose.position.z - PICK_MOVEMENT_DISTANCE
        rospy.loginfo("Measure distance: {}".format(distance))
        distance_pub.publish(distance)


        # object_gripped = random.random() > 0.4
        # rospy.loginfo("Object_gripped: {}".format(object_gripped))
        # gripper_pub.publish(object_gripped)
        rate.sleep()


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass