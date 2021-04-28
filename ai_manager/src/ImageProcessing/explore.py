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
        self.btn_find_best.clicked.connect(self.find_best_solution)
        self.btn_map.clicked.connect(self.compute_map)
        # Load the best model for evaluation ################################################
        self.image_model = ImageModel(model_name='resnet18')
        self.transform = transforms.Compose([
            transforms.Lambda(lambda img : self.crop_xy(img)),
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
            self.lbl_model_name.setText(model_name)

    def change_image(self):
        fname = QFileDialog.getOpenFileName(self, 'Open image file', '.',"Image files (*.jpg *.gif *.png)", options=QFileDialog.DontUseNativeDialog)
        if fname[0]:
            self.lbl_image_name.setText(os.path.basename(fname[0]))
            self.set_image(fname[0])

    def set_image(self,filename):
        self.canvas.set_image(filename)
        self.image = Image.open(filename)

    def crop_xy(self, image):
        return crop(image, self.predict_center_y - HEIGHT/2, self.predict_center_x - WIDTH/2, HEIGHT, WIDTH)  # top, left, height, width

    def predict(self, x, y):
        self.predict_center_x = x
        self.predict_center_y = y
        img = self.transform(self.image)  # Get the cropped transformed image
        #imshow(img)
        img = img.unsqueeze(0)  # To have a 4-dim tensor ([nb_of_images, channels, w, h])
        features, preds = self.image_model.evaluate_image(img, self.inference_model, False)  # No processing
        #self.lbl_result.setText(str(torch.exp(preds)))
        return torch.exp(preds)

    def find_best_solution(self):
        pass

    def compute_map(self):
        all_preds = self._compute_all_preds()
        self.canvas.all_preds = all_preds
        self.canvas.repaint()

    def _compute_all_preds(self):
        all_preds = []
        steps = int(self.edt_nb_pixels_per_step.text())
        (im_width, im_height) = self.image.size
        half_width = int(WIDTH/2)
        half_height = int(HEIGHT/2)
        for x in range(half_width, im_width -half_width, steps):
            for y in range(half_height, im_height - half_height, steps):
                preds = self.predict(x,y)
                all_preds.append([x, y, preds])
        return all_preds


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = MainWindow()
    gui.show()
    sys.exit(app.exec_())