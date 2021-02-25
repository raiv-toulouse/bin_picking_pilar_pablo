# import libraries

import os
import torch
import numpy as np
import re
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torchvision
from PIL import Image
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from pytorch_lightning.loggers import TensorBoardLogger
# from ray.tune.integration.pytorch_lightning import TuneReportCallback

from CNN import CNN
from MyImageModule import MyImageModule

torch.set_printoptions(linewidth=120)


class ImageModel:
    def __init__(self,
                 model_name,
                 dataset_size=None,
                 batch_size=8,
                 num_epochs=20,
                 img_size=256,
                 fine_tuning=True):
        super(ImageModel, self).__init__()
        # Parameters
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.img_size = img_size
        self.dataset_size = dataset_size
        # Set a seed  ################################################
        seed_everything(42)
        # Load model  ################################################
        self.model = CNN(backbone=model_name)
        self.model_name = model_name
        # self.image_module = MyImageModule(batch_size=self.batch_size, dataset_size=100)
        self.image_module = MyImageModule(batch_size=self.batch_size, dataset_size=self.dataset_size)
        # For getting the features for the image
        self.activation = {}
        # Save the model after every epoch by monitoring a quantity.
        current_path = os.path.dirname(os.path.realpath(__file__))
        self.MODEL_CKPT_PATH = os.path.join(current_path, f'model/{self.model_name}/')
        self.MODEL_CKPT = os.path.join(self.MODEL_CKPT_PATH, 'model-{epoch:02d}-{val_loss:.2f}')
        # Tensorboard Logger used
        self.logger = TensorBoardLogger('tb_logs', name=f'Model_{self.model_name}')
        # Flag for feature extracting. When False, we finetune the whole model,when True we only update the reshaped
        self.fine_tuning = fine_tuning

    def config_callbacks(self):
        # Checkpoint  ################################################
        # Saves the models so it is possible to access afterwards
        checkpoint_callback = ModelCheckpoint(dirpath=self.MODEL_CKPT_PATH,
                                              filename=self.MODEL_CKPT,
                                              monitor='val_loss',
                                              save_top_k=1,
                                              mode='min',
                                              save_weights_only=True)
        # EarlyStopping  ################################################
        # Monitor a validation metric and stop training when it stops improving.
        early_stop_callback = EarlyStopping(monitor='val_loss',
                                            min_delta=0.0,
                                            patience=2,
                                            verbose=False,
                                            mode='min')
        # tune_report_callback = TuneReportCallback({"loss": "ptl/val_loss",
        #                                            "mean_accuracy": "ptl/val_accuracy"}, on="validation_end")

        return checkpoint_callback, early_stop_callback

    def call_trainer(self):
        # Load images  ################################################
        self.image_module.setup()

        # Samples required by the custom ImagePredictionLogger callback to log image predictions.
        val_samples = next(iter(self.image_module.val_dataloader()))
        # val_imgs, val_labels = val_samples[0], val_samples[1]
        # print(val_imgs.shape)
        # print(val_labels.shape)
        grid = torchvision.utils.make_grid(val_samples[0], nrow=8, padding=2)
        # write to tensorboard
        self.logger.experiment.add_image('prueba', grid)
        self.logger.close()

        # Load callbacks ########################################
        checkpoint_callback, early_stop_callback = self.config_callbacks()

        # Trainer  ################################################
        trainer = pl.Trainer(max_epochs=self.num_epochs,
                             gpus=1,
                             logger=self.logger,
                             deterministic=True,
                             callbacks=[early_stop_callback, checkpoint_callback])

        # Config Hyperparameters ################################################
        if self.fine_tuning:
            self.tune_model(trainer)

        # Train model ################################################
        trainer.fit(model=self.model, datamodule=self.image_module)
        # Test  ################################################
        trainer.test(datamodule=self.image_module)
        # self.save_graph_logger(self.model)

    def evaluate(self, model, loader):
        y_true = []
        y_pred = []
        for imgs, labels in loader:
            features, prediction = model(imgs)
            y_true.extend(labels)
            y_pred.extend(prediction.detach().numpy())
        return np.array(y_true), np.array(y_pred)

    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook

    # Returns the size of features tensor
    def get_size_features(self, model):
        feature_size = model.get_size()
        return feature_size

    # @torch.no_grad()
    # def evaluate_image(self, image, model):
    #     image_tensor = self.image_preprocessing(image)
    #     # model.feature_extractor.classifier[6].register_forward_hook(self.get_activation('classifier[6]'))
    #     features, pred = model(image_tensor)
    #     # print("Features", self.activation['classifier[6]'])
    #     # features_size = output[0].shape
    #     return features.detach().numpy(), pred.detach().numpy()

    def evaluate_image(self, image, model):
        image_tensor = self.image_preprocessing(image)
        features, prediction = model(image_tensor)
        return features.detach().numpy(), prediction.detach()

    def image_preprocessing(self, image):
        transform = transforms.Compose([
            # you can add other transformations in this list
            # transforms.Grayscale(num_output_channels=1),
            transforms.CenterCrop(size=224),
            transforms.Resize(size=256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image).float()
        # image = Variable(image_tensor, requires_grad=True)
        image = image_tensor.unsqueeze(0)
        # print(image.shape)
        return image

    @torch.no_grad()
    def inference_model(self):
        best_model = self.load_best_model()
        # print(best_model)
        inference_model = self.model.load_from_checkpoint(self.MODEL_CKPT_PATH + best_model)
        return best_model, inference_model

    # Evaluate the model with the test data_loader
    def evaluate_model(self):
        _, inference_model = self.inference_model()
        # print("Inference model:", inference_model)
        # print("Test Dataloader:", self.image_module.test_dataloader())
        y_true, y_pred = self.evaluate(inference_model, self.image_module.test_dataloader())
        return y_true, y_pred

    def load_best_model(self):
        # Load best model  ################################################
        model_ckpts = os.listdir(self.MODEL_CKPT_PATH)
        losses = []
        for model_ckpt in model_ckpts:
            # print(model_ckpt)
            loss = re.findall("\d+\.\d+", model_ckpt)
            # print(loss)
            if not loss:
                losses = losses
            else:
                losses.append(float(loss[0]))

        losses = np.array(losses)
        best_model_index = np.argsort(losses)[0]
        best_model = model_ckpts[best_model_index]
        return best_model

    def load_model(self, name):
        # model_ckpts = os.listdir(self.MODEL_CKPT_PATH)
        model = self.model.load_from_checkpoint(self.MODEL_CKPT_PATH + name)
        model.freeze()
        # print(model)
        return model

    # Find the best learning rate
    def find_lr(self, trainer):
        lr_finder = trainer.tuner.lr_find(model=self.model,
                                          min_lr=1.e-5,
                                          max_lr=0.9,
                                          num_training=30,
                                          mode='exponential',
                                          datamodule=self.image_module)
        # Inspect results
        fig = lr_finder.plot()
        fig.savefig('lr_finder.png', format='png')
        suggested_lr = lr_finder.suggestion()
        print("Learning rate suggested:", suggested_lr)

    def find_optimal_batch_size(self, trainer):
        trainer.tune(model=self.model)

    # TODO: Fuction to finetune model hyperparameters
    def tune_model(self, trainer):
        # Run lr finder
        self.find_lr(trainer)
        self.find_optimal_batch_size(trainer)

    @torch.no_grad()
    def get_all_preds(self, model, loader):
        self.setup_data()
        with torch.no_grad():
            all_preds = torch.tensor([])
            all_targets = torch.tensor([])
            for batch in loader:
                images, labels = batch
                preds = model(images)
                all_preds = torch.cat((all_preds, preds[1]), dim=0)
                all_targets = torch.cat((all_targets, labels), dim=0)
        return torch.exp(all_preds), all_targets

    def setup_data(self):
        self.image_module.setup()


    def test_predictions(self, model):
        dataset = datasets.ImageFolder('pruebas_salida')
        dataloader = DataLoader(dataset, batch_size=1,
                                shuffle=True, num_workers=0)
        test_loader = self.image_module.test_dataloader()
        test_preds, test_targets = self.get_all_preds(model, test_loader)
        print(test_preds)


# --- MAIN ----
if __name__ == '__main__':
    print("Cuda:", torch.cuda.is_available())
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(dev)
    # print("Cuda:", torch.cuda.get_device_name(0))
    if dev.type == 'cuda':
        print(torch.cuda.get_device_name(0))

    # Config  ################################################
    image_model = ImageModel(model_name='resnet50', dataset_size=2692)
    # checkpoint_callback, early_stop_callback = image_model.config_callbacks()

    # Train model  ################################################
    # image_model.call_trainer()
    # y_true, y_pred = evaluate(model, image_module.test_dataloader())

    # Load model  ################################################
    name_model = 'model-epoch=05-val_loss=0.36-weights7y3_unfreeze2.ckpt'
    inference_model = image_model.load_model(name_model)

    #  Evaluate output  ################################################
    image_sinfichas = Image.open("pruebas_salida/img1611066111.526423.png")
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

