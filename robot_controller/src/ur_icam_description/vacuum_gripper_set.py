#!/usr/bin/env python
# coding: utf-8

import rospy
from vacuum_gripper import VacuumGripper
from ur_icam_description.srv import Grasp,GraspResponse

class VacuumGripperSet:
    def __init__(self,nbGrippers):
        self.lesGrippers = []
        for i in range(nbGrippers):
            self.lesGrippers.append(VacuumGripper(i))
        s = rospy.Service('grasp', Grasp, self.grasp)

    def grasp(self,msg):
        if msg.data:
            self.on()
        else:
            self.off()
        return GraspResponse()

    def on(self):
        for g in self.lesGrippers:
            g.gripper_on()

    def off(self):
        for g in self.lesGrippers:
            g.gripper_off()