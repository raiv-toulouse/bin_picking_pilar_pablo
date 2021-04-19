import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic
from ImageModel import ImageModel
from torchvision.transforms.functional import crop
from torchvision import transforms
from PIL import Image
import os
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
        self.btn_change_image.clicked.connect(self.change_image)
        self.btn_pick.clicked.connect(self.predict)
        self.btn_load_model.clicked.connect(self.load_model)
        # Load the best model for evaluation ################################################
        self.image_model = ImageModel(model_name='resnet18')
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x : self.crop_xy(x)),
            transforms.Resize(size=256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.inference_model = None
        self.change_image()

    def load_model(self):
        fname = QFileDialog.getOpenFileName(self, 'Open model file', '.', "Model files (*.ckpt)", options=QFileDialog.DontUseNativeDialog)
        if fname[0]:
            #model_name = self.image_model._find_name_of_best_model()
            model_name = os.path.basename(fname[0])
            self.inference_model = self.image_model.load_model(model_name) # Load the selected model
            self.lbl_model_name.setText(fname[0])

    def change_image(self):
        fname = QFileDialog.getOpenFileName(self, 'Open image file', '.',"Image files (*.jpg *.gif *.png)", options=QFileDialog.DontUseNativeDialog)
        if fname[0]:
            self.lbl_image_name.setText(fname[0])
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