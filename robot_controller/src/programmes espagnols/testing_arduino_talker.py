#!/usr/bin/env python
import rospy
from std_msgs.msg import Bool

def talker():
    pub = rospy.Publisher('switch_on_off', Bool, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(0.5) # 10hz
    while not rospy.is_shutdown():
        rospy.loginfo("Enviando mensaje: {}".format(True))
        pub.publish(True)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass