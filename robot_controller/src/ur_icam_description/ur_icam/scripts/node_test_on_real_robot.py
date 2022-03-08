#!/usr/bin/env python
import time
import roslib
import socket
import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import struct
from math import pi
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerRequest
from moveit_commander.conversions import pose_to_list
from ur_icam_description.robotUR import RobotUR
from tf.transformations import euler_from_quaternion, quaternion_from_euler

# Config
tcp_host_ip = "10.31.56.102"
tcp_port = 30002


def move_joints(joint0, joint1, joint2, joint3, joint4, joint5):
    # We can get the joint values from the group and adjust some of the values:
    joint_goal = move_group.get_current_joint_values() # 0, -pi/4, 0, -pi/2, 0, pi/3
    joint_goal[0] = joint0
    joint_goal[1] = joint1
    joint_goal[2] = joint2
    joint_goal[3] = joint3
    joint_goal[4] = joint4
    joint_goal[5] = joint5
    # The go command can be called with joint values, poses, or without any
    # parameters if you have already set the pose or joint target for the group
    move_group.go(joint_goal, wait=True)
    # Calling ``stop()`` ensures that there is no residual movement
    move_group.stop()
    print "Fin du move_joint"


def go_to_pose_goal(x, y, z, w):
    pose_goal = geometry_msgs.msg.Pose()
    pose_goal.position.x = x
    pose_goal.position.y = y
    pose_goal.position.z = z
    pose_goal.orientation.w = w
    quaternion = (
        pose_goal.position.x,
        pose_goal.position.y,
        pose_goal.position.z,
        pose_goal.orientation.w,
    )
    euler = euler_from_quaternion(quaternion)
    pose_goal.orientation.x = euler[0]
    pose_goal.orientation.y = euler[1]
    pose_goal.orientation.z = euler[2]
    print pose_goal
    print move_group.get_current_pose().pose
    move_group.set_pose_target(pose_goal)
    move_group.go(wait=True)
    move_group.stop()
    move_group.clear_pose_targets()
    current_pose = move_group.get_current_pose().pose
    print "Fin move_to_pose_goal"


def cartesian_path(x, y, z):
    scale = 1
    waypoints = []
    wpose = move_group.get_current_pose().pose
    if (x != None):
        wpose.position.x += scale * x  # First move up (z)
    if (y != None):
        wpose.position.y += scale * y  # First move up (z)
    if (z != None):
        wpose.position.z += scale * z  # First move up (z)
    #wpose.position.y += scale * 0.5  # and sideways (y)
    waypoints.append(copy.deepcopy(wpose))
    #wpose.position.x -= scale * 0.5  # Second move forward/backwards in (x)
    #waypoints.append(copy.deepcopy(wpose))
    #wpose.position.y -= scale * 0.1  # Third move sideways (y)
    #waypoints.append(copy.deepcopy(wpose))
    (plan, fraction) = move_group.compute_cartesian_path(waypoints, 0.01, 0.0)
    move_group.execute(plan, wait=True)
    print "Fin cartesian_path"

def open_gripper():
    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_socket.connect((tcp_host_ip, tcp_port))
    tcp_command = "set_digital_out(8,False)\n"
    tcp_socket.send(tcp_command)
    data = tcp_socket.recv(1024)
    tcp_socket.close()
    time.sleep(0.5)
    play_service = rospy.ServiceProxy('/ur_hardware_interface/dashboard/play', Trigger)
    play = TriggerRequest()
    result = play_service(play)
    time.sleep(0.5)
    print "Fin open_gripper"


def close_gripper():
    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_socket.connect((tcp_host_ip, tcp_port))
    tcp_command = "set_digital_out(8,True)\n"
    tcp_socket.send(tcp_command)
    data = tcp_socket.recv(1024)
    tcp_socket.close()
    time.sleep(0.5)
    play_service = rospy.ServiceProxy('/ur_hardware_interface/dashboard/play', Trigger)
    play = TriggerRequest()
    result = play_service(play)
    time.sleep(0.5)
    print "Fin close_gripper"

if __name__ == '__main__':
    try:
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('move_group_python_interface_tutorial', anonymous=True)
        robot = moveit_commander.RobotCommander()
        scene = moveit_commander.PlanningSceneInterface()
        group_name = "manipulator"
        move_group = moveit_commander.MoveGroupCommander(group_name)
        print "============ Reference frame: %s" % move_group.get_end_effector_link()
        move_joints(pi, -pi/3, pi/4, -pi/2, -pi/2, 0)
        print move_group.get_current_pose().pose
        open_gripper()
        cartesian_path(None, None, -0.15)
        close_gripper()
        cartesian_path(None, None, 0.15)
        move_joints((3*pi)/4, -pi / 3, pi / 4, -pi / 2, -pi / 2, 0)
        cartesian_path(None, None, -0.11)
        open_gripper()
        cartesian_path(None, None, 0.11)
        close_gripper()
        #move_joints(pi, -pi / 3, pi/4, pi, -pi, 0)
        move_joints(0, -pi/4, 0, -pi/2, 0, pi/3)
        go_to_pose_goal(0.4, 0.25, 0.4, 0)
    except rospy.ROSInterruptException:
        print ("Program interrupted before completion")