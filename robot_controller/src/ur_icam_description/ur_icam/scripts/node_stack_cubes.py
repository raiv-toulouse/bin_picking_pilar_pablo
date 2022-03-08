#!/usr/bin/python
# coding: utf-8
#
# Empilement des 3 cubes
#

from math import pi
import rospy
from ur_icam_description.gripper import Robotiq85Gripper
from ur_icam_description.robotUR import RobotUR
import control_msgs.msg
import moveit_msgs.msg
from moveit_msgs.msg import MotionPlanRequest
from control_msgs.msg import GripperCommandResult

def pick_and_place(x,y,z,h,initialJointsState):
    '''
    Attrape un cube et le place sur la pile au dessus du cube du milieu (se trouvant en [x,0,z])
    :param x: abscisse du cube
    :param y: ordonnée du cube
    :param z: élévation du cube (les coord sont par rapport au repère lié à la base du robot)
    :param h: hauteur du cube
    :param initialJointsState: position initiale de repos du robot
    :return:
    '''
    pose_goal = myRobot.get_current_pose().pose  # On récupère ses coord articulaires et cartésiennes
    pose_goal.position.x = x
    pose_goal.position.y = y
    # Move the robot just over the left cube
    pose_goal.position.z = z + 0.2  # 20 cm au dessus de la prise
    myRobot.go_to_pose_goal(pose_goal)
    # Go down to grasp position
    pose_goal.position.z = z
    myRobot.go_to_pose_goal(pose_goal)
    # Close the gripper
    print myGripper
    myGripper.close(close_pos=0.4)
    rospy.sleep(1)  # Pour laisser le temps au contact de se faire
    # Go up
    pose_goal.position.z = z + 0.2  # 20 cm au dessus de la prise
    myRobot.go_to_pose_goal(pose_goal)
    print control_msgs.msg.GripperCommandResult()
    moveit_msgs.msg.MotionPlanRequest().max_velocity_scaling_factor=0.1
    print moveit_msgs.msg.MotionPlanRequest().max_velocity_scaling_factor
    # Move over the middle cube
    pose_goal.position.y = 0
    myRobot.go_to_pose_goal(pose_goal)
    # Move just above the middle cube(s)
    pose_goal.position.z = z + h
    myRobot.go_to_pose_goal(pose_goal)
    # Open the gripper
    myGripper.open()
    rospy.sleep(1)  # Pour laisser le temps au contact de se défaire
    # Go up to initial pose
    myRobot.go_to_joint_state(initialJointsState)

if __name__ == '__main__':
    try:
        # Start the ROS node
        rospy.init_node('stack_cubes')
        myGripper = Robotiq85Gripper()
        myRobot = RobotUR()
        myRobot.velocity_factor(0.1)
        myRobot.acceleration_factor(0.1)
        initialJointsState = [0, -pi / 2, pi / 2, -pi / 2, -pi / 2, 0]
        myRobot.go_to_joint_state(initialJointsState)
        myRobot.velocity_factor(0.5)
        myRobot.acceleration_factor(0.5)
        pick_and_place(0.6, -0.2, -0.1, 0.05, initialJointsState)  # Left cube
        myRobot.velocity_factor(1)
        myRobot.acceleration_factor(1)
        pick_and_place(0.6, 0.2, -0.1, 0.05 + 0.05, initialJointsState)  # Right cube, h is 0.05 more
    except rospy.ROSInterruptException:
        print ("Program interrupted before completion")