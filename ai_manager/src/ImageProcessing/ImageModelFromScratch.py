#!/usr/bin/env python

# import libraries
import torch
from PIL import Image
from ImageModel import ImageModel
import numpy as np

# --- MAIN ----
if __name__ == '__main__':
    print("Cuda:", torch.cuda.is_available())
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(dev)
    if dev.type == 'cuda':
        print(torch.cuda.get_device_name(0))

    # Config  ################################################
    image_model = ImageModel(model_name='resnet50', dataset_size=2692)

    # Train model  ################################################
    image_model.call_trainer()
    print('End of model training')

    # Evaluate the model ################################################
    image_model.setup_data()
    y_true, y_pred = image_model.evaluate_model()
    print(y_true)
    print(np.exp(y_pred))

    # Load model  ################################################
    name_model = 'model-epoch=03-val_loss=0.40-v6.ckpt'
    inference_model = image_model.load_model(name_model)

    #  Evaluate output  ################################################
    #image_sinfichas = Image.open("pruebas_salida/img1611066111.526423.png")
    # image_confichas = Image.open("pruebas_salida/img1611066172.4361975.png")
    image_success = Image.open("images/success/img1607942892.174248.png")

    # image_model.setup_data()
    # preds, targets = image_model.get_all_preds(inference_model, image_model.image_module.test_dataloader())
    # print(preds)

    image_tensor = image_model.image_preprocessing(image_success)
    features, preds = image_model.evaluate_image(image_success, inference_model)
    print(torch.exp(preds))


    # features, pred_sinfichas = image_model.evaluate_image(image_sinfichas, inference_model)
    # features, pred_confichas = image_model.evaluate_image(image_confichas, inference_model)
    # features, pred_success = image_model.evaluate_image(image_success, inference_model)
    # print("Sin fichas:", torch.exp(pred_sinfichas))
    # print("Con fichas:",  torch.exp(pred_confichas))
    # print("Success:",  torch.exp(pred_success))

    # image_model.test_predictions(inference_model)

    # feature_size = image_model.get_size_features(inference_model)
    # print(feature_size)

    # Evaluate model  ################################################
    # inference_model = image_model.inference_model()
    # y_true, y_pred = image_model.evaluate_model()

