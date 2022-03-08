#!/usr/bin/env python
# coding: utf-8
import geometry_msgs.msg
from math import pi
from ur_icam_description.robotUR import RobotUR
import rospy
#
# Permet de positionner le robot à un endroit précis à l'aide de coord cartésiennes
#
if __name__ == '__main__':
      robot = RobotUR()
      rospy.init_node('robotUR')
      objectifAtteint = robot.go_to_joint_state([0, -pi / 2, pi/2, -pi / 2, -pi/2, 0])
      # On teste le positionnement par rapprot à des coordonnées cartésiennes
      pose_goal = geometry_msgs.msg.Pose()
      pose_goal.orientation.x = 0.0
      pose_goal.orientation.y = 0.707
      pose_goal.orientation.z = 0.0
      pose_goal.orientation.w = 0.707
      pose_goal.position.x = 0.5
      pose_goal.position.y = 0.1
      pose_goal.position.z = 0.0
      robot.go_to_pose_goal(pose_goal)
