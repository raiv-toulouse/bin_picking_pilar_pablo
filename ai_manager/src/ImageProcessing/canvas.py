from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import torch
WIDTH = HEIGHT = 224

class Canvas(QWidget):

    def __init__(self,parent):
        super().__init__(parent)
        self.parent = parent
        self.image = QImage('photo.png')
        self.setMinimumSize(self.image.width(), self.image.height())
        self.pressed = self.moving = False
        self.revisions = []

    def set_image(self,filename):
        self.image = QImage(filename)
        self.revisions = []
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.reset()
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
            self.revisions.append(self.image.copy())
            qp = QPainter(self.image)
            self.draw_rectangle(qp)
            self.pressed = self.moving = False
            self.update()

    def paintEvent(self, event):
        qp = QPainter(self)
        rect = event.rect()
        qp.drawImage(rect, self.image, rect)
        if self.moving or self.pressed:
            self.draw_rectangle(qp)

    def draw_rectangle(self, qp):
        top = self.center.x() - WIDTH / 2
        left = self.center.y()-HEIGHT/2
        qp.setRenderHint(QPainter.Antialiasing)
        qp.setPen(QPen(Qt.red, 5))
        qp.drawPoint(self.center)
        preds = self.parent.predict()
        prob,cl = torch.max(preds, 1)
        if cl.item()==0:  # Fail
            prob = 100 * (1 - prob.item())
        else:  # Success
            prob = 100 * prob.item()
        fail_or_success = 'Fail' if cl.item()==0 else 'Success'
        text = f'{fail_or_success} : {prob:.1f}%'
        qp.setPen(Qt.black)
        qp.setFont(QFont('Decorative', 10))
        qp.drawText(top, left, text)
        if cl.item() == 1:  # Success
            qp.setPen(QPen(Qt.green, 1, Qt.DashLine))
        else:  # Fail
            qp.setPen(QPen(Qt.red, 1, Qt.DashLine))
        qp.drawRect(top, left, WIDTH, HEIGHT)

    def undo(self):
        if self.revisions:
            self.image = self.revisions.pop()
            self.update()

    def reset(self):
        if self.revisions:
            self.image = self.revisions[0]
            self.revisions.clear()
            self.update()

