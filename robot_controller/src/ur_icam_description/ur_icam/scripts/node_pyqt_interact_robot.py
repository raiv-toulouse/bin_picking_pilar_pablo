#!/usr/bin/env python
# coding: utf-8
from PyQt5 import QtGui, QtCore
from math import pi
from PyQt5.QtWidgets import *
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from ur_icam_description.reglage import Reglage
from ur_icam_description.robotUR import RobotUR
import rospy


class InteractRobotPyQt(QWidget):
    def __init__(self, initialPose):
        super(InteractRobotPyQt, self).__init__()
        # loadUi('reglage.ui', self)
        # Start the ROS node
        rospy.init_node('interact_robot')
        self.robot = RobotUR()
        objectifAtteint = self.robot.go_to_joint_state(initialPose)  # On met le robot en position initiale
        # objectifAtteint = self.robot.go_to_joint_state([0, 0, 0, 0, 0, 0])
        self.pose_goal = self.robot.get_current_pose().pose  # On récupère ses coord articulaires et cartésiennes
        # Définition de l'IHM
        vLayout = QVBoxLayout()
        # Partie cartésienne
        titre = QLabel("Coordonnées cartésiennes")
        titre.setFont(QtGui.QFont("Arial", 14, QtGui.QFont.Black))
        titre.setAlignment(QtCore.Qt.AlignCenter)
        vLayout.addWidget(titre)
        vLayout.addWidget(QLabel("Position"))
        self.reglPoseX = Reglage(-1, 1, "x : ", self.pose_goal.position.x)
        vLayout.addWidget(self.reglPoseX)
        self.reglPoseY = Reglage(-1, 1, "y : ", self.pose_goal.position.y)
        vLayout.addWidget(self.reglPoseY)
        self.reglPoseZ = Reglage(-0.5, 1, "z : ", self.pose_goal.position.z)
        vLayout.addWidget(self.reglPoseZ)
        vLayout.addWidget(QLabel("Orientation"))
        q = self.pose_goal.orientation
        phi, theta, psi = euler_from_quaternion([q.w, q.x, q.y, q.z])
        self.reglOrientPhi = Reglage(-3.14, 3.14, "phi : ", phi)
        vLayout.addWidget(self.reglOrientPhi)
        self.reglOrientTheta = Reglage(-3.14, 3.14, "theta : ", theta)
        vLayout.addWidget(self.reglOrientTheta)
        self.reglOrientPsi = Reglage(-3.14, 3.14, "psi : ", psi)
        vLayout.addWidget(self.reglOrientPsi)
        bouton = QPushButton('Send move cartesian', self)
        bouton.clicked.connect(self.changerPoseCartesian)
        vLayout.addWidget(bouton)
        # Partie angulaire
        titre = QLabel("Coordonnées angulaires")
        titre.setFont(QtGui.QFont("Arial", 14, QtGui.QFont.Black))
        titre.setAlignment(QtCore.Qt.AlignCenter)
        vLayout.addWidget(titre)
        self.lesReglAngulaires = self.creerWidgetsReglageAng(initialPose, vLayout)
        self.setLayout(vLayout)

    def creerWidgetsReglageAng(self, initialPose, vLayout):
        lesReglAngulaires = []
        for i in range(6):
            reglAng = Reglage(-3.14, 3.14, "q{} : ".format(i + 1), initialPose[i])
            reglAng.slider.valueChanged.connect(self.changerPoseAngular)
            vLayout.addWidget(reglAng)
            lesReglAngulaires.append(reglAng)
        return lesReglAngulaires

    def majPoseAngular(self):
        for angle, reglage in zip(self.robot.get_current_joint(), self.lesReglAngulaires):
            reglage.majEditEtSlider(angle)

    def majPoseCartesian(self):
        p = self.robot.get_current_pose().pose
        self.reglPoseX.majEditEtSlider(p.position.x)
        self.reglPoseY.majEditEtSlider(p.position.y)
        self.reglPoseZ.majEditEtSlider(p.position.z)
        o = p.orientation
        phi, theta, psi = euler_from_quaternion([o.w,o.x,o.y,o.z])
        self.reglOrientPhi.majEditEtSlider(phi)
        self.reglOrientTheta.majEditEtSlider(theta)
        self.reglOrientPsi.majEditEtSlider(psi)


    def changerPoseAngular(self):
        angPose = []
        for w in self.lesReglAngulaires:
            angPose.append(w.value())
        self.robot.go_to_joint_state(angPose)
        self.majPoseCartesian()

    def changerPoseCartesian(self):
        self.pose_goal.position.x = self.reglPoseX.value()
        self.pose_goal.position.y = self.reglPoseY.value()
        self.pose_goal.position.z = self.reglPoseZ.value()
        phi = self.reglOrientPhi.value()
        theta = self.reglOrientTheta.value()
        psi = self.reglOrientPsi.value()
        orient = quaternion_from_euler(phi, theta, psi)
        self.pose_goal.orientation.w = orient[0]
        self.pose_goal.orientation.x = orient[1]
        self.pose_goal.orientation.y = orient[2]
        self.pose_goal.orientation.z = orient[3]
        self.robot.go_to_pose_goal(self.pose_goal)
        self.majPoseAngular()


#
# Permet de positionner le robot à un endroit précis à l'aide de coord cartésiennes
#
if __name__ == '__main__':
    app = QApplication([])
    ihm = InteractRobotPyQt([0, -pi / 2, pi / 2, -pi / 2, -pi / 2, 0])
    ihm.resize(800, 800)
    ihm.show()
    app.exec_()
