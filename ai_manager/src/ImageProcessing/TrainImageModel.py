#!/usr/bin/env python

# import libraries
import torch
from PIL import Image
from ImageModel import ImageModel
import numpy as np

# Train a CNN. Use a MyImageModule to load the images located in the default train and val folders (default : 'images', see MyImageModule.py)
# The resulting model will be stored in a file which name looks like this : model-epoch=01-val_loss=0.62.ckpt
# and which is located in 'model/<model name>' like 'model/resnet50'
# To view the logs : tensorboard --logdir=tb_logs

# --- MAIN ----
if __name__ == '__main__':

    image_model = ImageModel(model_name='resnet50', num_epochs=20, dataset_size=1000)

    image_model.call_trainer()  # Train model

    print('End of model training')



