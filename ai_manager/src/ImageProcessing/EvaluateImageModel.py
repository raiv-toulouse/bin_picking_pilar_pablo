#!/usr/bin/env python

# import libraries
import torch
from PIL import Image
from ImageModel import ImageModel
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

def test(writer, y_true, y_pred):
    result_list = []
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

def create_dataloader(batch_size=4):
    transform = transforms.Compose([
        # you can add other transformations in this list
        # transforms.Grayscale(num_output_channels=1),
        transforms.CenterCrop(size=224),
        transforms.Resize(size=256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = torchvision.datasets.ImageFolder('images', transform=transform)
    return DataLoader(dataset, batch_size, shuffle=False, num_workers=0)

def infere_image(dataloader, inference_model, min, max):
    """
    Use inference_model to predict class of images.
    Images come from [min, max] set of image names from dataloader
    :param dataloader:
    :param inference_model:
    :param min:
    :param max:
    :return:
    """
    for name,_ in dataloader.dataset.samples[min:max]:
        img = Image.open(name)
        features, preds = image_model.evaluate_image(img, inference_model)
        print(torch.exp(preds))

def add_figure_to_tensorboard(ind, inputs, classes, writer, image_model):
    output = image_model.model(inputs)
    print(torch.exp(output[1]))
    writer.add_figure(f'predictions vs. actuals {ind}',
                      image_model.plot_classes_preds(inputs, classes))

if __name__ == '__main__':

    # Load the best model for evaluation ################################################
    image_model = ImageModel(model_name='resnet18')
    writer = SummaryWriter()
    model_name = image_model._find_name_of_best_model()
    print("The name of the evaluated model (the one with the smallest loss) is ", model_name)
    # model_name = 'model-epoch=06-val_loss=0.42.ckpt'
    inference_model = image_model.load_model(model_name) # Load the best model
    feature_size = image_model.get_size_features(inference_model)
    print("feature_size = ", feature_size)
    dataloader = create_dataloader(batch_size=8)

    #  Plot prediction for some images
    it_data = iter(dataloader)
    inputs, classes = next(it_data)
    add_figure_to_tensorboard(0, inputs, classes, writer, image_model)
    for i in range(26):  # Skip to bees images
        inputs, classes = next(it_data)
    add_figure_to_tensorboard(1, inputs, classes, writer, image_model)


    print("ants (classe = 0) ============= ")
    infere_image(dataloader, inference_model, 0, 8)
    print("bees (classe = 1) ============= ")
    infere_image(dataloader, inference_model, 208, 216)

    #  Evaluate on an image  #############################################
    image_success = Image.open("images/success/img1607942817.4263833.png")
    features, preds = image_model.evaluate_image(image_success, inference_model)
    print("Prediction for a success image (img1607942817.4263833.png) ", torch.exp(preds))

    # y_true, y_pred = image_model.evaluate_model(dataloader)
    # test(writer, y_true, y_pred)
    writer.close()