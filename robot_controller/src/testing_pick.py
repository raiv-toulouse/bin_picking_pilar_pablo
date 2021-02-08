#!/usr/bin/env python2
from os import access

import rospy
import time
import copy
from std_msgs.msg import Bool
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
import sys

import datetime

from robot import Robot
from ai_manager.Environment import Environment


def change_plan_speed(plan, new_speed):
    """
    Function used for changing Robot velocity of a cartesian path once the movement have been planned.
    :param plan: RobotTrajectory object. For example, the one calculated by compute_cartesian_path() MoveGroup function.
    :param new_speed: speed factor of the robot, been 1 the original speed and 0 the minimum.
    :return: RobotTrajectory object (new plan).
    """
    new_plan = plan
    n_joints = len(plan.joint_trajectory.joint_names)
    n_points = len(plan.joint_trajectory.points)

    points = []
    for i in range(n_points):
        plan.joint_trajectory.points[i].time_from_start = plan.joint_trajectory.points[i].time_from_start / new_speed
        velocities = []
        accelerations = []
        positions = []
        for j in range(n_joints):
            velocities.append(plan.joint_trajectory.points[i].velocities[j] * new_speed)
            accelerations.append(plan.joint_trajectory.points[i].accelerations[j] * new_speed)
            positions.append(plan.joint_trajectory.points[i].positions[j])

        point = plan.joint_trajectory.points[i]
        point.velocities = velocities
        point.accelerations = accelerations
        point.positions = positions

        points.append(point)

    new_plan.joint_trajectory.points = points

    return new_plan


def back_to_original_pose(robot):
    """
    Function used to go back to the original height once a vertical movement has been performed.
    :param robot: robot_controller.robot.py object
    :return:
    """
    distance = Environment.CARTESIAN_CENTER[2] - robot.robot.get_current_pose().pose.position.z
    robot.relative_move(0, 0, distance)


def down_movement(robot, movement_speed):
    """
    This function performs the down movement of the pick action.

    It creates an asynchronous move group trajectory planning. This way the function is able to receive distance
    messages while the robot is moving and stop it once the robot is in contact with an object.

    Finally, when there is any problems with the communications the movement is stopped and
    communication_problem boolean flag is set to True. It is considered that there is a problem with
    communications when the robot is not receiving any distance messages during 200 milli-seconds (timeout=0.2)

    :param robot: robot_controller.robot.py object
    :return: communication_problem flag
    """

    distance_ok = rospy.wait_for_message('distance', Bool).data  # We retrieve sensor distance
    communication_problem = False

    if not distance_ok:  # If the robot is already in contact with an object, no movement is performed
        waypoints = []
        wpose = robot.robot.get_current_pose().pose
        wpose.position.z -= (wpose.position.z - 0.26)  # Third move sideways (z)
        waypoints.append(copy.deepcopy(wpose))

        (plan, fraction) = robot.robot.move_group.compute_cartesian_path(
            waypoints,  # waypoints to follow
            0.01,  # eef_step
            0.0)  # jump_threshold

        plan = change_plan_speed(plan, movement_speed)
        robot.robot.move_group.execute(plan, wait=False)

        while not distance_ok:
            try:
                distance_ok = rospy.wait_for_message('distance', Bool, 0.2).data  # We retrieve sensor distance
            except:
                print("Unexpected error:", sys.exc_info()[0])
                communication_problem = True
                rospy.loginfo("Error in communications, trying again")
                break

        # Both stop and 10 mm up movement to stop the robot
        robot.robot.move_group.stop()
        robot.relative_move(0, 0, 0.001)

    return communication_problem


if __name__ == '__main__':
    rospy.init_node('robotUR')

    robot = Robot()
    robot.go_to_initial_pose()
    while True:
        while True:
            communication_problem = down_movement(robot, 0.5)
            if communication_problem:
                rospy.loginfo("Problem in communications")
            else:
                break

        back_to_original_pose(robot)