#!/usr/bin/env python
# coding: utf-8
import rospy
from geometry_msgs.msg import *
import random,argparse,sys
from gazebo_msgs.srv import SpawnModel

#
#  Création de plusieurs objets à des positions aléatoires
#

def create_objects(urdfFile,nb_boxes,xc,yc,dx,dy):
    rospy.wait_for_service("gazebo/spawn_urdf_model")
    try:
        spawn_model = rospy.ServiceProxy("gazebo/spawn_urdf_model", SpawnModel)
        boxXML = open(urdfFile,'r').read()
        for i in range(nb_boxes):
            x = random.uniform(xc-dx,xc+dx)
            y = random.uniform(yc-dy,yc+dy)
            spawn_model("object{}".format(i), boxXML,"/object{}".format(i), Pose(position= Point(x,y,0.2),orientation=Quaternion(0,0,0,0)),"world")
    except rospy.ServiceException as e:
        rospy.loginfo("Service call failed: ",e)



if __name__ == '__main__':
    rospy.init_node('create_boxes', anonymous=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--nb_objects", type=int, default=2,help="nb of objects")
    parser.add_argument("-x", "--x", type=float, default=0,help="coord x du centre")
    parser.add_argument("-y", "--y", type=float, default=0,help="oord y du centre")
    parser.add_argument("--dx", type=float, default=1,help="demi-largeur en X")
    parser.add_argument("--dy", type=float, default=1,help="demi-largeur en Y")
    parser.add_argument("file", help="fichier URDF décrivant l'objet")
    args = parser.parse_args(rospy.myargv()[1:])
    #create_objects("/home/philippe/Devel/catkin_ws/src/ur_icam_description/urdf/red_box.urdf",args.nb_objects,args.x,args.y,args.dx,args.dy)
    create_objects(args.file,args.nb_objects,args.x,args.y,args.dx,args.dy)
