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
rosrun ai_manager main.py

- Activate the node
rosrun robot_controller arduino.py

- Finally, we can run the program
rosrun robot_controller main.py

"""

import rospy

from ai_manager.srv import GetActions
from Robot import Robot


def get_action(robot, object_gripped):
    relative_coordinates = robot.calculate_current_coordinates()
    rospy.wait_for_service('get_actions')
    try:
        get_actions = rospy.ServiceProxy('get_actions', GetActions)
        return get_actions(relative_coordinates[0], relative_coordinates[1], object_gripped).action
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)


# This function defines the movements that robot should make depending on the action listened
def take_action(action, robot):
    rospy.loginfo("Action received: {}".format(action))
    object_gripped = False
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
    return object_gripped


if __name__ == '__main__':

    rospy.init_node('robotUR')

    robot = Robot()

    # Test of positioning with angular coordinates
    robot.go_to_initial_pose()
    robot.take_place()

    # Let's put the robot in a random position to start, creation of new state
    object_gripped = take_action('random_state', robot)

    while True:
        action = get_action(robot, object_gripped)
        object_gripped = take_action(action, robot)