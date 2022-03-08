# coding: utf-8
import actionlib
import control_msgs.msg

ACTION_SERVER = '/gripper_controller/gripper_cmd'

class Gripper(object):
    """Gripper controls the robot's gripper.
    """
    def __init__(self):
        self._client = actionlib.SimpleActionClient(ACTION_SERVER, control_msgs.msg.GripperCommandAction)
        # Wait until the action server has been started and is listening for goals
        self._client.wait_for_server()  # rospy.Duration(10)
        self.goal = control_msgs.msg.GripperCommandGoal()


class Robotiq85Gripper(Gripper):
    '''
    Pince Robotiq 85 https://robotiq.com/fr/produits/main-adaptative-a-2-doigts-2f85-140
    '''
    def __init__(self):
        super(Robotiq85Gripper, self).__init__()
        self.posOpen = 0.0 # ATTENTION cette pince marche à l'envers 0 = open et 0.8 = fermée
        self.posClose = 0.8
        self.maxEffort = 40
        self.open()

    def open(self):
        """Opens the gripper.
        """
        self.goal.command.position = self.posOpen
        self._client.send_goal(self.goal)
        self._client.wait_for_result()

    def close(self, close_pos=None, max_effort=None):
        """Closes the gripper.

        The `goal` has type:
            <class 'control_msgs.msg._GripperCommandGoal.GripperCommandGoal'>
        with a single attribute, accessed via `goal.command`, which consists of:
            position: 0.0
            max_effort: 0.0
        by default, and is of type:
            <class 'control_msgs.msg._GripperCommand.GripperCommand'>

        Args:
            max_effort: The maximum effort, in Newtons, to use. Note that this
                should not be less than 35N, or else the gripper may not close.
        """
        self.goal.command.position = close_pos if close_pos != None else self.posClose
        self.goal.command.max_effort = max_effort if max_effort != None else self.maxEffort
        self._client.send_goal(self.goal)
        self._client.wait_for_result()


