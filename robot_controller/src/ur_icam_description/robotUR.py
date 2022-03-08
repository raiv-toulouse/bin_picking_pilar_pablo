#!/usr/bin/env python
# coding: utf-8

import time
import copy
import sys
import socket
from math import pi
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import moveit_commander
import rospy
import rospkg
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from moveit_commander.conversions import pose_to_list
from std_srvs.srv import Trigger, TriggerRequest

# Config
tcp_robot_ip = "10.31.56.102"
tcp_port = 30002

# Use this class to drive a Universal Robot with ROS
# First, run the communication between the robot and ROS :
# roslaunch ur_robot_driver ur3_bringup.launch robot_ip:=10.31.56.102 kinematics_config:=${HOME}/Calibration/ur3_calibration.yaml
# Then, MoveIt! must be launched : roslaunch ur3_moveit_config ur3_moveit_planning_execution.launch
# Finally, run this node : rosrun ur_icam_description robotUR.py

class RobotUR(object):
    def __init__(self):
        super(RobotUR, self).__init__()
        # First initialize `moveit_commander`_ and a `rospy`_ node:
        moveit_commander.roscpp_initialize(sys.argv)
        # Instantiate a `RobotCommander`_ object. Provides information such as the robot's
        # kinematic model and the robot's current joint states
        self.robot = moveit_commander.RobotCommander()
        # Instantiate a `MoveGroupCommander`_ object.  This object is an interface
        # to a planning group (group of joints).  The group is the primary
        # arm joints in the UR robot, so we set the group's name to "manipulator".
        # This interface can be used to plan and execute motions:
        group_name = "manipulator"
        self.move_group = moveit_commander.MoveGroupCommander(group_name)
        ## Instantiate a `PlanningSceneInterface`_ object.  This provides a remote interface
        ## for getting, setting, and updating the robot's internal understanding of the
        ## surrounding world:
        self.scene = moveit_commander.PlanningSceneInterface()

    def get_current_pose(self):
        """

        @return:
        """
        return self.move_group.get_current_pose()

    def get_current_joint(self):
        """

        @return:
        """
        return self.move_group.get_current_joint_values()

    def disconnect(self):
        """

        @return:
        """
        moveit_commander.roscpp_shutdown()

    def setPose(self, x, y, z, phi, theta, psi):
        """

        @param x:
        @param y:
        @param z:
        @param phi:
        @param theta:
        @param psi:
        @return:
        """
        orient = Quaternion(quaternion_from_euler(phi, theta, psi))
        pose = Pose(Point(x, y, z), orient)
        self.move_group.set_pose_target(pose)
        self.move_group.go(True)
        self.move_group.stop()

    def go_to_joint_state(self, joints_goal):
        """
        go to a position specified by angular joint coordinates
        @param joints_goal:
        @return:
        """
        # The go command can be called with joint values, poses, or without any
        # parameters if you have already set the pose or joint target for the group
        self.move_group.go(joints_goal, wait=True)
        # Calling ``stop()`` ensures that there is no residual movement
        self.move_group.stop()
        # On teste si l'on a atteind l'objectif
        current_joints = self.move_group.get_current_joint_values()
        return self.all_close(joints_goal, current_joints, 0.01)

    def go_to_pose_goal(self, pose_goal):
        """
        Planning to a cartesian goal
        @param pose_goal:
        @return:
        """
        # We can plan a motion for this group to a desired pose for the end-effector:
        self.move_group.set_pose_target(pose_goal)
        # Now, we call the planner to compute the plan and execute it.
        plan = self.move_group.go(wait=True)
        # Calling `stop()` ensures that there is no residual movement
        self.move_group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        self.move_group.clear_pose_targets()
        current_pose = self.move_group.get_current_pose().pose
        return self.all_close(pose_goal, current_pose, 0.01)

    def exec_cartesian_path(self, waypoints):
        """
        Execute a cartesian path throw a list of waypoints
        @param waypoints:
        @return:
        """
        # We want the Cartesian path to be interpolated at a resolution of 1 cm
        # which is why we will specify 0.01 as the eef_step in Cartesian
        # translation.  We will disable the jump threshold by setting it to 0.0,
        # ignoring the check for infeasible jumps in joint space, which is sufficient
        # for this tutorial.
        (plan, fraction) = self.move_group.compute_cartesian_path(
            waypoints,  # waypoints to follow
            0.01,  # eef_step
            0.0)  # jump_threshold
        self.move_group.execute(plan, wait=True)
        self.move_group.stop()

    def add_obstacle_box(self,name,size,position):
        """

        @param name:
        @param size:
        @param position:
        @return:
        """
        self.scene.remove_world_object(name)
        pose = PoseStamped()
        pose.header.frame_id = "world"
        pose.pose.position.x = position[0]
        pose.pose.position.y = position[1]
        pose.pose.position.z = position[2]
        self.scene.add_box(name, pose, size=size)
        return self.wait_for_state_update(name, box_is_known=True)

    def add_obstacle_table(self,name,size,position):
        """

        @param name:
        @param size:
        @param position:
        @return:
        """
        # Create table obstacle
        self.scene.remove_world_object(name)
        pose = PoseStamped()
        pose.header.frame_id = "world"
        pose.pose.position.x = position[0]
        pose.pose.position.y = position[1]
        pose.pose.position.z = position[2]
        r = rospkg.RosPack()
        path = r.get_path('ur_icam_description')
        self.scene.add_mesh(name, pose, path+'/models/cafe_table/meshes/cafe_table.dae')
        return self.wait_for_state_update(name, box_is_known=True)

    def acceleration_factor(self, scaling_value):
        """

        @param scaling_value:
        @return:
        """
        return self.move_group.set_max_acceleration_scaling_factor(scaling_value)

    def velocity_factor(self, scaling_value):
        """

        @param scaling_value:
        @return:
        """
        return self.move_group.set_max_velocity_scaling_factor(scaling_value)

    def all_close(self, goal, actual, tolerance):
        """
        Convenience method for testing if a list of values are within a tolerance of their counterparts in another list
        @param: goal       A list of floats, a Pose or a PoseStamped
        @param: actual     A list of floats, a Pose or a PoseStamped
        @param: tolerance  A float
        @returns: bool
        """
        try:
            all_equal = True
            if type(goal) is list:
                for index in range(len(goal)):
                    if abs(actual[index] - goal[index]) > tolerance:
                        return False
            elif type(goal) is PoseStamped:
                return self.all_close(goal.pose, actual.pose, tolerance)
            elif type(goal) is Pose:
                return self.all_close(pose_to_list(goal), pose_to_list(actual), tolerance)
            return True
        except TypeError:
            rospy.logerr("Incompatible types between goal and actual in 'RobotUR.allClose'")

    def wait_for_state_update(self, box_name, box_is_known=False, box_is_attached=False, timeout=4):
        """

        @param box_name:
        @param box_is_known:
        @param box_is_attached:
        @param timeout:
        @return:
        """
        ## Ensuring Collision Updates Are Received
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        ## If the Python node dies before publishing a collision object update message, the message
        ## could get lost and the box will not appear. To ensure that the updates are
        ## made, we wait until we see the changes reflected in the
        ## ``get_attached_objects()`` and ``get_known_object_names()`` lists.
        ## For the purpose of this tutorial, we call this function after adding,
        ## removing, attaching or detaching an object in the planning scene. We then wait
        ## until the updates have been made or ``timeout`` seconds have passed
        start = rospy.get_time()
        seconds = rospy.get_time()
        while (seconds - start < timeout) and not rospy.is_shutdown():
          # Test if the box is in attached objects
          attached_objects = self.scene.get_attached_objects([box_name])
          is_attached = len(attached_objects.keys()) > 0
          # Test if the box is in the scene.
          # Note that attaching the box will remove it from known_objects
          is_known = box_name in self.scene.get_known_object_names()
          # Test if we are in the expected state
          if (box_is_attached == is_attached) and (box_is_known == is_known):
            return True
          # Sleep so that we give other threads time on the processor
          rospy.sleep(0.1)
          seconds = rospy.get_time()
        # If we exited the while loop without returning then we timed out
        return False

    def open_gripper(self):
        """

        @return:
        """
        tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_socket.connect((tcp_robot_ip, tcp_port))
        tcp_command = "set_digital_out(8,False)\n"
        tcp_socket.send(tcp_command)
        tcp_socket.close()
        time.sleep(0.5)
        play_service = rospy.ServiceProxy('/ur_hardware_interface/dashboard/play', Trigger)
        play = TriggerRequest()
        play_service(play)
        time.sleep(0.5)

    def close_gripper(self):
        """

        @return:
        """
        tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_socket.connect((tcp_robot_ip, tcp_port))
        tcp_command = "set_digital_out(8,True)\n"
        tcp_socket.send(tcp_command)
        tcp_socket.close()
        time.sleep(0.5)
        play_service = rospy.ServiceProxy('/ur_hardware_interface/dashboard/play', Trigger)
        play = TriggerRequest()
        play_service(play)
        time.sleep(0.5)

#
#  Test the different RobotUR methods
#
if __name__ == '__main__':
    myRobot = RobotUR()
    rospy.init_node('robotUR')
    print("Press ENTER to continue")
    raw_input()
    myRobot.open_gripper()
    print("Press ENTER to continue")
    raw_input()
    myRobot.close_gripper()
    # Getting Basic Information
    # ^^^^^^^^^^^^^^^^^^^^^^^^^
    # We can get the name of the reference frame for this robot:
    planning_frame = myRobot.move_group.get_planning_frame()
    print("============ Planning frame: %s" % planning_frame)
    # We can also print the name of the end-effector link for this group:
    eef_link = myRobot.move_group.get_end_effector_link()
    print("============ End effector link: %s" % eef_link)
    # We can get a list of all the groups in the robot:
    group_names = myRobot.robot.get_group_names()
    print("============ Available Planning Groups:", myRobot.robot.get_group_names())
    # Sometimes for debugging it is useful to print the entire state of the robot
    print("============ Printing robot current pose")
    print(myRobot.get_current_pose())
    print("============ Printing robot state")
    print(myRobot.robot.get_current_state())
    print("============ Press `Enter` to execute a movement using a joint state goal ...")
    raw_input()
    # Test of positioning with angular coordinates
    targetReached = myRobot.go_to_joint_state([0, -pi / 4, 0, -pi / 2, 0, pi / 3])
    if targetReached:
        print("Target reached")
    else:
        print("Target not reached")
    print("Press ENTER to continue")
    raw_input()
    # Test of positioning with cartesian coordinates
    print("go_to_pose_goal test")
    pose_goal = Pose()
    pose_goal.position.x = 0.4
    pose_goal.position.y = 0.1
    pose_goal.position.z = 0.4
    myRobot.go_to_pose_goal(pose_goal)
    print("Press ENTER to continue")
    raw_input()
    # Test of positioning with different cartesian waypoints
    print("exec_cartesian_path test")
    waypoints = []
    wpose = myRobot.get_current_pose().pose
    wpose.position.z += 0.1  # First move up (z)
    wpose.position.y += 0.1  # and sideways (y)
    waypoints.append(copy.deepcopy(wpose))
    wpose.position.x += 0.1  # Second move forward/backwards in (x)
    waypoints.append(copy.deepcopy(wpose))
    wpose.position.y -= 0.1  # Third move sideways (y)
    waypoints.append(copy.deepcopy(wpose))
    myRobot.exec_cartesian_path(waypoints)
    print("Press ENTER to continue")
    raw_input()
    pose_goal = Pose()
    pose_goal.position.x = 0.4
    pose_goal.position.y = 0.1
    pose_goal.position.z = 0.4
    myRobot.go_to_pose_goal(pose_goal)
    print("The end")

