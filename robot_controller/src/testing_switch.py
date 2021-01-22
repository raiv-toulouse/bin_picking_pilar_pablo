#!/usr/bin/env python
import rospy
from std_msgs.msg import Bool
import random

def talker():
    pub = rospy.Publisher('switch_on_off', Bool, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(1) # 10hz
    while not rospy.is_shutdown():
        switch_on_off = random.choice([True,False])
        rospy.loginfo("Switch: {}".format(switch_on_off))
        pub.publish(switch_on_off)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass