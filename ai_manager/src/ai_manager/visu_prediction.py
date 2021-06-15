import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt
from PyQt5 import uic
import rospy
from ai_manager.msg import ListOfPredictions
from sensor_msgs.msg import Image

THRESHOLD = 0.5  # threshold for success grasping prediction

class VisuPrediction(QWidget):

    def __init__(self):
        super().__init__()
        uic.loadUi("visu_prediction.ui",self)
        rospy.init_node('visu_prediction')
        rospy.Subscriber("predictions", ListOfPredictions, self._update_predictions)
        rospy.Subscriber('new_image', Image, self._change_image)
        self.preds = None
        self.image = None

    def _change_image(self, req):
        format = QImage.Format_RGB888
        image = QImage(req.data, req.width, req.height, format)
        self.image = image

    def _update_predictions(self,data):
        self.preds = data.predictions
        self.repaint()

    def paintEvent(self, event):
        qp = QPainter(self)
        rect = event.rect()
        if self.image:
            qp.drawImage(rect, self.image, rect)
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