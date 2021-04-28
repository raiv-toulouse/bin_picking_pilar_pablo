from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import torch
WIDTH = HEIGHT = 224

class Canvas(QWidget):

    def __init__(self,parent):
        super().__init__(parent)
        self.parent = parent
        self.pressed = self.moving = False
        self.previous_image = None
        self.all_preds = None

    def set_image(self,filename):
        self.image = QImage(filename)
        self.setMinimumSize(self.image.width(), self.image.height())
        self.previous_image = None
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.previous_image:  # A new click, we restore the previous image without the rectangle
                self.image = self.previous_image
                self.setMinimumSize(self.image.width(), self.image.height())
            self.pressed = True
            self.center = event.pos()
            self.update()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.moving = True
            self.center = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.previous_image = self.image.copy()
            qp = QPainter(self.image)
            self._draw_rectangle(qp)
            self.pressed = self.moving = False
            self.update()

    def paintEvent(self, event):
        qp = QPainter(self)
        rect = event.rect()
        qp.drawImage(rect, self.image, rect)
        if self.moving or self.pressed:
            self._draw_rectangle(qp)
        if self.all_preds:
            self._draw_pred(qp)


    def _draw_rectangle(self, qp):
        if self.parent.inference_model:  # A model exists, we can do inference
            top = self.center.x() - WIDTH / 2
            left = self.center.y()-HEIGHT/2
            qp.setRenderHint(QPainter.Antialiasing)
            qp.setPen(QPen(Qt.red, 5))
            qp.drawPoint(self.center)
            preds = self.parent.predict(self.center.x(), self.center.y())  # calculate the prediction wih CNN
            prob,cl = torch.max(preds, 1)
            if cl.item()==0:  # Fail
                prob = 100 * (1 - prob.item())
            else:  # Success
                prob = 100 * prob.item()
            fail_or_success = 'Fail' if cl.item()==0 else 'Success'
            text = f'{fail_or_success} : {prob:.1f}%'
            qp.setPen(Qt.black)
            qp.setFont(QFont('Decorative', 14))
            qp.drawText(top, left, text)
            if cl.item() == 1:  # Success
                qp.setPen(QPen(Qt.green, 1, Qt.DashLine))
            else:  # Fail
                qp.setPen(QPen(Qt.red, 1, Qt.DashLine))
            qp.drawRect(top, left, WIDTH, HEIGHT)

    def _draw_pred(self, qp):
        for pred in self.all_preds:
            prob,cl = torch.max(pred[2], 1)
            if cl.item()==0:  # Fail
                qp.setPen(QPen(Qt.red, 5))
            else:
                qp.setPen(QPen(Qt.green, 5))
            qp.drawPoint(pred[0], pred[1])

