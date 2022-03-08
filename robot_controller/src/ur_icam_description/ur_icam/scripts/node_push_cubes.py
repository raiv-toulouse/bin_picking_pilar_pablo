#!/usr/bin/python
# coding: utf-8
#
# Pousser des cubes à différentes vitesses
#
import copy
from math import pi
import rospy
from ur_icam_description.gripper import Robotiq85Gripper

def push(x,y,z,initialJointsState):
    '''
    Pousser les trois cubes un par un avec des vitesses et accelerations différentes
    :param x: abscisse du cube
    :param y: ordonnée du cube
    :param z: élévation du cube (les coord sont par rapport au repère lié à la base du robot)
    :param h: hauteur du cube
    :param initialJointsState: position initiale de repos du robot
    :return:
    '''
    pose_goal = myRobot.get_current_pose().pose  # On récupère ses coord articulaires et cartésiennes
    # Close the gripper
    #myGripper.close(close_pos=0.8)

    pose_goal.position.y = y
    pose_goal.position.z = z
    # Move the robot just over the left cube
    pose_goal.position.x = x + 0.05 # 15 cm avant l'objet
    myRobot.go_to_pose_goal(pose_goal)

    waypoints = []
    wpose = myRobot.get_current_pose().pose

    # Close the gripper
    #myGripper.close(close_pos=0.8)

    wpose.position.x = x - 0.1
    waypoints.append(copy.deepcopy(wpose))

    myRobot.exec_cartesian_path(waypoints)

    # Go up to initial pose
    myRobot.go_to_joint_state(initialJointsState)


if __name__ == '__main__':
    try:
        # Start the ROS node
        rospy.init_node('push_cubes')
        #myGripper = Robotiq85Gripper()
        myRobot = RobotUR()
        initialJointsState = [pi, -pi / 2, pi / 2, -pi / 2, -pi / 2, 0]
        myRobot.velocity_factor(0.2)
        myRobot.acceleration_factor(0.2)
        myRobot.go_to_joint_state(initialJointsState)
        push(-0.4, -0.2, 0.22, initialJointsState)  # Left cube
        myRobot.velocity_factor(0.5)
        myRobot.acceleration_factor(0.5)
        push(-0.4, 0, 0.22, initialJointsState)  # Middle cube
        myRobot.velocity_factor(1)
        myRobot.acceleration_factor(1)
        push(-0.4, 0.2, 0.22, initialJointsState)  # Right cube, h is 0.05 more




        print "Fin"



        # # myRobot.velocity_factor(0.2)
        # # myRobot.acceleration_factor(0.2)
        #
        # # myRobot.velocity_factor(0.5)
        # # myRobot.acceleration_factor(0.5)
        #
        # # myRobot.velocity_factor(1)
        # # myRobot.acceleration_factor(1)
        #




    except rospy.ROSInterruptException:
        print ("Program interrupted before completion")