#!/usr/bin/env python
# coding: utf-8

# Test de commande du robot avec des coordonées entrées au clavier

from tf.transformations import euler_from_quaternion, quaternion_from_euler
import moveit_commander
import rospy
import rospkg
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from moveit_commander.conversions import pose_to_list

from ur_icam_description.robotUR import RobotUR

if __name__ == '__main__':
    myRobot = RobotUR()
    rospy.init_node('robotUR')
    x = float(input("Entrez x en cm: "))
    y = float(input("Entrez y en cm: "))
    z = float(input("Entrez z en cm: "))

    if 10 < abs(x) < 40 and abs(y) < 30 and 7 < z < 60:

        pose_goal = Pose()
        pose_goal.position.x = -x / 100
        pose_goal.position.y = -y / 100
        pose_goal.position.z = z / 100
        pose_goal.orientation.x = 0.6679889495197082
        pose_goal.orientation.y = 0.20682604239647592
        pose_goal.orientation.z = -0.6823111630646372
        pose_goal.orientation.w = 0.21322576829161657

        myRobot.go_to_pose_goal(pose_goal)
        print("target reached")

    else:
        print("out of bonds")
