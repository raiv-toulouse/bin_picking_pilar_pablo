#!/usr/bin/env python
"""
Code used to train the UR3 robot to perform a pick and place task using Reinforcement Learning and Image Recognition.
This code does not perform actions directly into the robot, it just posts actions in a ROS topic and
gathers state information from another ROS topic.
"""

import rospy
import torch

from ai_manager.srv import GetActions, GetActionsResponse
from RLAlgorithm import RLAlgorithm
from Environment import Environment

rospy.init_node('ai_manager', anonymous=True)  # ROS node initialization
# Global Image Controller
RL_ALGORITHM = RLAlgorithm.recover_training(batch_size=256, lr=0.0001,
                                            others='optimal_original_rewards_algorithm1901')
                                            # others = 'optimal_original_rewards_new_model')

def handle_get_actions(req):
    """
    Callback for each Request from the Robot
    :param req: Robot requests has 3 elements: object_gripped, x and y elements
    :return:
    """
    object_gripped = req.object_gripped
    current_coordinates = [req.x, req.y]
    # Next action is calculated from the current state
    action = RL_ALGORITHM.next_training_step(current_coordinates, object_gripped)

    # RL_ALGORITHM.plot()

    return GetActionsResponse(action)


def get_actions_server():
    """
    Service initialization to receive requests of actions from the robot.
    Each time that a request is received, handle_get_actions function will be called
    :return:
    """
    s = rospy.Service('get_actions', GetActions, handle_get_actions)
    rospy.loginfo("Ready to send actions.")
    rospy.spin()
    rospy.on_shutdown(save_training)


def save_training():
    RL_ALGORITHM.save_training()


if __name__ == '__main__':
    try:
        get_actions_server()
    except rospy.ROSInterruptException:
        pass
