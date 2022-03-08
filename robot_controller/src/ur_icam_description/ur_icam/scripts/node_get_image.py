#!/usr/bin/env python
# coding: utf-8
import rospy
from ur_icam_description.srv import InitDirectory

#
#  Prend une photo à l'aide du service 'record_image' fourni par camera.py
#
if __name__ == '__main__':
    rospy.loginfo("On prend une photo")
    rospy.wait_for_service('record_image')
    try:
        srv_record = rospy.ServiceProxy('record_image', InitDirectory)
        srv_record('/home/philippe')  # Appel au service
    except rospy.ServiceException, e:
        rospy.logerr("Service record_image failed")
