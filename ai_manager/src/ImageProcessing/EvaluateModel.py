#!/usr/bin/env python

# import libraries
import torch
from PIL import Image
from ImageModel import ImageModel
import numpy as np
from ImageModelFromScratch import config
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':

    # Evaluate the model ################################################
    image_model = config()
    image_model.setup_data()
    y_true, y_pred = image_model.evaluate_model()
    result_list = []
    # draw graph
    writer = SummaryWriter()
    init = 0
    for i in range(len(np.array(y_true))):
        if np.array(y_true)[i] == 1:
            lab = "Success"

        elif np.array(y_true)[i] == 0:
            lab = "Fail"

        print("label : ", lab)

        a = np.exp(y_pred)[i]
        print("prediction: Fail : ", a[0] * 100, "%", "| Success : ", a[1] * 100, "%")
        if lab == "Fail" and a[0] > a[1] or lab == "Success" and a[1] > a[0]:
            print("GOOD")
            result_list.append(1)
            init = init + 1
            writer.add_scalar('prediction', init, i)
        else:
            print("BAD")
            result_list.append(0)
            init = init - 1
            writer.add_scalar('prediction', init, i)
        print("\n")
    writer.close()
    compt = 0
    for i in result_list:

        if i == 1:
            compt = compt + i

    error = (1 - (compt / len(result_list))) * 100
    print("Precision : ", 100 - error, "%")
    print("Global error : ", error, "%")

    # print(y_true)

    # print(np.exp(y_pred))

    # Load model  ################################################
    name_model = 'model-epoch=06-val_loss=0.42.ckpt'
    inference_model = image_model.load_model(name_model)

    #  Evaluate output  ################################################
    # image_sinfichas = Image.open("pruebas_salida/img1611066111.526423.png")
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