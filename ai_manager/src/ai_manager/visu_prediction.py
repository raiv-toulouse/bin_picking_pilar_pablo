import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt
from PyQt5 import uic
import random
import rospy
from ai_manager.msg import ListOfPredictions

THRESHOLD = 0.5  # threshold for success grasping prediction

class VisuPrediction(QWidget):

    def __init__(self):
        super().__init__()
        uic.loadUi("visu_prediction.ui",self)
        rospy.init_node('visu_prediction')
        rospy.Subscriber("predictions", ListOfPredictions, self.update_predictions)
        self.preds = None

    def update_predictions(self,data):
        self.preds = data.predictions
        self.repaint()

    def paintEvent(self, event):
        qp = QPainter(self)
        if self.preds:
            for pred in self.preds:
                x = pred.x
                y = pred.y
                if pred.proba > THRESHOLD:
                    qp.setPen(QPen(Qt.green, 3))
                else:
                    qp.setPen(QPen(Qt.red, 3))
                qp.drawPoint(x, y)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = VisuPrediction()
    gui.show()
    sys.exit(app.exec_())