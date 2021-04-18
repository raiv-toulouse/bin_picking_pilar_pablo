import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic
from ImageModel import ImageModel
from torchvision.transforms.functional import crop
from torchvision import transforms
from PIL import Image
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Qt5Agg')

WIDTH = HEIGHT = 224

def imshow(images, title=None, pil_image = False):
    """Imshow for Tensor."""
    if pil_image:
        inp = images
    else:
        img_grid = torchvision.utils.make_grid(images).cpu()
        inp = img_grid.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(2)

class MainWindow(QWidget):

    def __init__(self):
        super().__init__()
        uic.loadUi("explore_ihm.ui",self)
        QShortcut(QKeySequence('Ctrl+Z'), self, self.canvas.undo)
        QShortcut(QKeySequence('Ctrl+R'), self, self.canvas.reset)
        self.btn_change_image.clicked.connect(self.change_image)
        self.btn_pick.clicked.connect(self.predict)
        # Load the best model for evaluation ################################################
        self.image_model = ImageModel(model_name='resnet18')
        model_name = self.image_model._find_name_of_best_model()
        self.inference_model = self.image_model.load_model(model_name) # Load the best model
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x : self.crop_xy(x)),
            transforms.Resize(size=256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.change_image()

    def change_image(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', '.',"Image files (*.jpg *.gif *.png)")
        self.set_image(fname[0])

    def set_image(self,filename):
        self.canvas.set_image(filename)
        self.image = Image.open(filename)

    def crop_xy(self, image):
        return crop(image, self.canvas.center.y()-HEIGHT/2, self.canvas.center.x()-WIDTH/2, HEIGHT, WIDTH)  # top, left, height, width

    def predict(self):
        img = self.transform(self.image)
        #imshow(img)
        img = img.unsqueeze(0)  # To have a 4-dim tensor ([nb_of_images, channels, w, h])
        features, preds = self.image_model.evaluate_image(img, self.inference_model, False)  # No processing
        self.lbl_result.setText(str(torch.exp(preds)))
        return torch.exp(preds)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = MainWindow()
    gui.show()
    sys.exit(app.exec_())