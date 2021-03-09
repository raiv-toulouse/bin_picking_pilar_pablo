#!/usr/bin/env python
# coding: utf-8

"""
- We need to connect the camera and the nodes
roslaunch ur_icam_description webcam.launch

- We need to establish a connection to the robot with the following comand:
roslaunch ur_robot_driver ur3_bringup.launch robot_ip:=10.31.56.102 kinematics_config:=${HOME}/Calibration/ur3_calibration.yaml

- Then, we ned to activate moovit server:
roslaunch ur3_moveit_config ur3_moveit_planning_execution.launch

- Activate the talker
rosrun ai_manager main_controller.py

- Activate the node
rosrun robot_controller arduino.py

- Finally, we can run the program
rosrun robot_controller main_controller.py

"""

import rospy

from ai_manager.srv import GetActions
from robot import Robot


def get_action(robot, object_gripped):
    relative_coordinates = robot.calculate_current_coordinates()
    rospy.wait_for_service('get_actions')
    try:
        get_actions = rospy.ServiceProxy('get_actions', GetActions)
        print(get_actions(relative_coordinates[0], relative_coordinates[1], object_gripped).action)
        return get_actions(relative_coordinates[0], relative_coordinates[1], object_gripped).action
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)


# This function defines the movements that robot should make depending on the action listened
def take_action(action, robot):
    rospy.loginfo("Action received: {}".format(action))
    object_gripped = False
    stop = False
    if action == 'north':
        robot.take_north()
    elif action == 'south':
        robot.take_south()
    elif action == 'east':
        robot.take_east()
    elif action == 'west':
        robot.take_west()
    elif action == 'pick':
        object_gripped = robot.take_pick()
    elif action == 'random_state':
        robot.take_random_state()
    elif action == 'place':
        robot.take_place()
    elif action == 'initial':
        robot.go_to_initial_pose()
    elif action == 'end':
        stop = True
    else:
        rospy.loginfo("The action {} is unknowned".format(action))
    return object_gripped, stop


if __name__ == '__main__':


    rospy.init_node('robotUR')

    robot = Robot()
    i = int(input("nombre de tentatives : "))
    for x in range(0, i-1):
        print(x+1)
        robot.take_random_state()
        object_gripped = robot.take_pick()

        robot.take_random_state()
    # Test of positioning with angular coordinates
    robot.go_to_initial_pose()
    print('1')
   # robot.take_place()
    print('2')

    stop = False
    while not stop:

        action = input("Which action to perform (north, south, east, west, pick, random_state, end, place, initial)? ")
        object_gripped, stop = take_action(action, robot)
