#!/usr/bin/env python

# import libraries
import torch
from PIL import Image
from ImageModel import ImageModel
import numpy as np

def config()
    # Config  ################################################
    model = ImageModel(model_name='resnet50', dataset_size=2692)
    return model

# --- MAIN ----
if __name__ == '__main__':
    print("Cuda:", torch.cuda.is_available())
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(dev)
    if dev.type == 'cuda':
        print(torch.cuda.get_device_name(0))

    image_model = config()

    # Train model  ################################################
    image_model.call_trainer()
    print('End of model training')



