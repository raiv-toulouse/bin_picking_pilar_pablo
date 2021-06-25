# import libraries

import os
import torch
import numpy as np
import re
import pytorch_lightning as pl
import torchvision
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torchvision import transforms
from pytorch_lightning.loggers import TensorBoardLogger
from ImageProcessing.CNN import CNN
from ImageProcessing.MyImageModule import MyImageModule
import matplotlib.pyplot as plt
import torch.nn.functional as F

plt.switch_backend('agg')
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
        self.image_module = MyImageModule(batch_size=self.batch_size, dataset_size=self.dataset_size, data_dir='./banque_images/PiecesPlatesx200')
        # Load images  ################################################
        self.image_module.setup()
        # For getting the features for the image
        self.activation = {}
        # Save the model after every epoch by monitoring a quantity.      current_path = os.path.dirname(os.path.realpath(__file__))
        self.MODEL_CKPT_PATH = os.path.join(current_path, f'model/{self.model_name}/')
        self.MODEL_CKPT = os.path.join(self.MODEL_CKPT_PATH, 'model-{epoch:02d}-{val_loss:.2f}')
        # Tensorboard Logger used
        self.logger = TensorBoardLogger('tb_logs2', name=f'Model_{self.model_name}')
        # Flag for feature extracting. When False, we finetune the whole model,when True we only update the reshaped
        self.fine_tuning = fine_tuning

    def call_trainer(self):
        # Samples required by the custom ImagePredictionLogger callback to log image predictions.
        val_samples = next(iter(self.image_module.val_dataloader()))
        grid = self.image_module.inv_trans(torchvision.utils.make_grid(val_samples[0], nrow=8, padding=2))
        # write to tensorboard
        self.logger.experiment.add_image('test', grid)
        self.logger.close()
        # Load callbacks ########################################
        checkpoint_callback, early_stop_callback = self._config_callbacks()
        # Trainer  ################################################
        trainer = pl.Trainer(max_epochs=self.num_epochs,
                             gpus=1,
                             logger=self.logger,
                             deterministic=True,
                             progress_bar_refresh_rate=0,  # To remove the progress bar
                             #callbacks=[early_stop_callback, checkpoint_callback])
                             callbacks = [checkpoint_callback])
        # Config Hyperparameters ################################################
        if self.fine_tuning:
            self._tune_model(trainer)
        # Train model ################################################
        trainer.fit(model=self.model, datamodule=self.image_module)
        # Test  ################################################
        trainer.test(datamodule=self.image_module)


    # Returns the size of features tensor
    def get_size_features(self, model):
        feature_size = model.get_size()
        return feature_size

    @torch.no_grad()
    def evaluate_image(self, image, model, with_processing = True):
        if with_processing:
            image_tensor = self.image_preprocessing(image)
        else:
            image_tensor = image
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
        image = image_tensor.unsqueeze(0)
        return image


    def evaluate_model(self, dataloader = None):
        '''
        Evaluate the model with the test data_loader
        :return: true outputs and predicted outputs
        '''
        _, inference_model = self._inference_model()
        if dataloader:
            y_true, y_pred = self._evaluate(inference_model, dataloader)
        else:  # We use the default test_dataloader
            y_true, y_pred = self._evaluate(inference_model, self.image_module.test_dataloader())
        return y_true, y_pred


    def load_model(self, name):
        """
        Load the model named 'name' from the MODEL_CKPT_PATH folder (ie : model/resnet50)
        :param name: name of the model
        :return: the model freezed to be used for inference
        """
        self.model = self.model.load_from_checkpoint(self.MODEL_CKPT_PATH + name)
        self.model.freeze()
        return self.model


    def plot_classes_preds(self, images, labels):
        '''
        Generates matplotlib Figure using a trained network, along with images
        and labels from a batch, that shows the network's top prediction along
        with its probability, alongside the actual label, coloring this
        information based on whether the prediction was correct or not.
        Uses the "images_to_probs" function.
        '''
        preds, probs = self._images_to_probs(images)
        # plot the images in the batch, along with predicted and true labels
        my_dpi = 96 # For my monitor (see https://www.infobyip.com/detectmonitordpi.php)
        nb_images = len(images)
        fig = plt.figure(figsize=(nb_images * 224/my_dpi, 224/my_dpi), dpi=my_dpi)
        class_names = self.image_module.find_classes()
        for idx in np.arange(nb_images):
            ax = fig.add_subplot(1, nb_images, idx + 1, xticks=[], yticks=[])
            img = self.image_module.inv_trans(images[idx])
            npimg = img.cpu().numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
                class_names[preds[idx]],
                probs[idx] * 100.0,
                class_names[labels[idx]]),
                color=("green" if preds[idx] == labels[idx].item() else "red"))
        return fig


    ##### Private methods #####

    def _config_callbacks(self):
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
                                            patience=5,
                                            verbose=False,
                                            mode='min')
        # tune_report_callback = TuneReportCallback({"loss": "ptl/val_loss",
        #                                            "mean_accuracy": "ptl/val_accuracy"}, on="validation_end")
        return checkpoint_callback, early_stop_callback


    def _evaluate(self, model, loader):
        y_true = []
        y_pred = []
        for imgs, labels in loader:
            features, prediction = model(imgs)
            y_true.extend(labels)
            y_pred.extend(prediction.detach().numpy())
        return np.array(y_true), np.array(y_pred)


    @torch.no_grad()
    def _inference_model(self):
        best_model_name = self._find_name_of_best_model()
        inference_model = self.model.load_from_checkpoint(self.MODEL_CKPT_PATH + best_model_name)
        return best_model_name, inference_model

    def _find_name_of_best_model(self):
        """
        Find the name of the model with the smallest loss from the MODEL_CKPT_PATH folder
        :return: the best model name
        """
        # Load best model  ################################################
        model_ckpts = os.listdir(self.MODEL_CKPT_PATH)
        losses = []
        for model_ckpt in model_ckpts:
            loss = re.findall("\d+\.\d+", model_ckpt)
            if not loss:
                losses = losses
            else:
                losses.append(float(loss[0]))
        losses = np.array(losses)
        best_model_index = np.argsort(losses)[0]
        best_model_name = model_ckpts[best_model_index]
        return best_model_name

    # Find the best learning rate
    def _find_lr(self, trainer):
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

    def _find_optimal_batch_size(self, trainer):
        trainer.tune(model=self.model)

    # TODO: Fuction to finetune model hyperparameters
    def _tune_model(self, trainer):
        # Run lr finder
        self._find_lr(trainer)
        self._find_optimal_batch_size(trainer)

    def _images_to_probs(self, images):
        '''
        Generates predictions and corresponding probabilities from a trained
        network and a list of images
        '''
        output = self.model(images)
        # convert output probabilities to predicted class
        _, preds_tensor = torch.max(output[1], 1)
        # preds = np.squeeze(preds_tensor.cpu().numpy())  CAusait une erreur quand il n'y avait qu'une seule image
        preds = preds_tensor.cpu().numpy()
        return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output[1])]

    # @torch.no_grad()
    # def _get_all_preds(self, model, loader):
    #     self.setup_data()
    #     with torch.no_grad():
    #         all_preds = torch.tensor([])
    #         all_targets = torch.tensor([])
    #         for batch in loader:
    #             images, labels = batch
    #             preds = model(images)
    #             all_preds = torch.cat((all_preds, preds[1]), dim=0)
    #             all_targets = torch.cat((all_targets, labels), dim=0)
    #     return torch.exp(all_preds), all_targets

    # def test_predictions(self, model):
    #     dataset = datasets.ImageFolder('pruebas_salida')
    #     dataloader = DataLoader(dataset, batch_size=1,
    #                             shuffle=True, num_workers=0)
    #     test_loader = self.image_module.test_dataloader()
    #     test_preds, test_targets = self._get_all_preds(model, test_loader)
    #     print(test_preds)

    # def get_activation(self, name):
    #     def hook(model, input, output):
    #         self.activation[name] = output.detach()
    #     return hook

    # def setup_data(self):
    #     self.image_module.setup()
