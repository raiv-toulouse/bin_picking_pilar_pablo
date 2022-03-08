#!/usr/bin/env python
# coding: utf-8

import rospy
from std_srvs.srv import Empty

class VacuumGripper:
    def __init__(self,ind):
        self.serviceOn = "/ur5/vacuum_gripper{}/on".format(ind)
        self.serviceOff = "/ur5/vacuum_gripper{}/off".format(ind)

    def gripper_on(self):
        # Wait till the srv is available
        rospy.wait_for_service(self.serviceOn)
        try:
            # Create a handle for the calling the srv
            turn_on = rospy.ServiceProxy(self.serviceOn, Empty)
            # Use this handle just like a normal function and call it
            resp = turn_on()
            return resp
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e

    def gripper_off(self):
        rospy.wait_for_service(self.serviceOff)
        try:
            turn_off = rospy.ServiceProxy(self.serviceOff, Empty)
            resp = turn_off()
            return resp
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e
