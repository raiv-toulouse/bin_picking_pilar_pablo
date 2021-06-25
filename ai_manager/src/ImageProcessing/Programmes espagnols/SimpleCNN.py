import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from collections import OrderedDict
import os

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import torch.nn as nn
from torch.nn import Module
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import pytorch_lightning as pl
import torchvision.models as models
from typing import Optional

from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning.metrics.functional import accuracy

from sklearn.metrics import confusion_matrix
# from plotcm import plot_confusion_matrix
import pdb


#  --- Utility functions ---

class SimpleCNN(pl.LightningModule):

    # defines the network
    def __init__(self,
                 input_shape: list = [3, 256, 256],
                 # backbone: str = 'vgg16',
                 train_bn: bool = True,
                 milestones: tuple = (5, 10),
                 batch_size: int = 8,
                 learning_rate: float = 1e-2,
                 lr_scheduler_gamma: float = 1e-1,
                 num_workers: int = 6):
        super(SimpleCNN, self).__init__()
        # parameters
        self.dim = input_shape
        # self.backbone = backbone
        self.train_bn = train_bn
        self.milestones = milestones
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.num_workers = num_workers
        self.num_target_classes = 2

        self.__build_model()

    def __build_model(self):
        """Define model layers & loss."""

        # 1.  Define network:  choose the model
        # Feature extractor
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1)

        self.pool1 = torch.nn.MaxPool2d(2)
        self.pool2 = torch.nn.MaxPool2d(2)

        _feature_extractor_layes = [self.conv1, ]

        self.feauture_extractor = nn.Sequential()


        # Classifier
        n_sizes = self._get_conv_output(self.dim)
        self.fc1 = nn.Linear(n_sizes, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, self.num_target_classes)

        _layers = list(self.feature_extractor.children())[:-1]
        # print(_layers)
        # self.feature_extractor = torch.nn.Sequential(*_layers)

    # returns the size of the output tensor going into Linear layer from the conv block.
    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    # mandatory
    def forward(self, x):
        """Forward pass. Returns logits."""
        # 1. Feature extraction:
        t = self.feature_extractor(t)
        features = t.squeeze(-1).squeeze(-1)
        # 2. Classifier (returns logits):
        t = self.fc(features)

        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)

        return features, x

        # returns the feature tensor from the conv block
    def _forward_features(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool2(F.relu(self.conv4(x)))
        return x

    def get_size(self):
        n_sizes = self._get_conv_output(self.dim)
        return n_sizes

    # loss function
    def loss(self, logits, labels):
        return self.loss_func(input=logits, target=labels)

    # training loop
    def training_step(self, batch, batch_idx):
        # x = images , y = batch, logits = labels
        x, y = batch
        logits = self(x)

        # 2. Compute loss & accuracy:
        train_loss = F.cross_entropy(logits, y)
        # train_loss = self.loss(logits, y)
        print(train_loss)
        preds = torch.argmax(logits, dim=1)
        # num_correct = torch.sum(preds == y).float() / preds.size(0)
        num_correct = torch.eq(preds.view(-1), y.view(-1)).sum()
        acc = accuracy(preds, y)
        self.log('train_loss', train_loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)
        # self.log('num_correct', num_correct, on_step=True, on_epoch=True, logger=True)

        # 3. Outputs:
        tqdm_dict = {'train_loss': train_loss}
        output = OrderedDict({'loss': train_loss,
                              'num_correct': num_correct,
                              'log': tqdm_dict,
                              'progress_bar': tqdm_dict})
        return output
        # return train_loss

    # If you need to do something with all the outputs of each training_step
    def training_epoch_end(self, outputs):
        """Compute and log training loss and accuracy at the epoch level."""

        train_loss_mean = torch.stack([output['loss']
                                       for output in outputs]).mean()
        train_acc_mean = torch.stack([output['num_correct']
                                      for output in outputs]).sum().float()
        train_acc_mean /= (len(outputs) * self.batch_size)
        return {'log': {'train_loss': train_loss_mean,
                        'train_acc': train_acc_mean,
                        'step': self.current_epoch}}

    # validation loop
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        # 2. Compute loss & accuracy:
        # val_loss = self.loss(logits, y)
        val_loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        # num_correct = torch.sum(preds == y).float() / preds.size(0)
        num_correct = torch.eq(preds.view(-1), y.view(-1)).sum()
        acc = accuracy(preds, y)
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, logger=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, logger=True)
        # self.log('num_correct', num_correct, on_step=True, on_epoch=True, logger=True)

        return {'val_loss': val_loss,
                'num_correct': num_correct}

    def validation_epoch_end(self, outputs):
        """Compute and log validation loss and accuracy at the epoch level."""

        val_loss_mean = torch.stack([output['val_loss']
                                     for output in outputs]).mean()
        val_acc_mean = torch.stack([output['num_correct']
                                    for output in outputs]).sum().float()
        val_acc_mean /= (len(outputs) * self.batch_size)
        return {'log': {'val_loss': val_loss_mean,
                        'val_acc': val_acc_mean,
                        'step': self.current_epoch}}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        # 2. Compute loss & accuracy:
        # test_loss = self.loss(logits, y)
        test_loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        # num_correct = torch.sum(preds == y).float() / preds.size(0)
        num_correct = torch.eq(preds.view(-1), y.view(-1)).sum()

        acc = accuracy(preds, y)
        self.log('test_loss', test_loss, on_step=True, on_epoch=True, logger=True)
        self.log('test_acc', acc, on_step=True, on_epoch=True, logger=True)
        # self.log('num_correct', num_correct, on_step=True, on_epoch=True, logger=True)
        return {'test_loss': test_loss,
                'num_correct': num_correct}

    # define optimizers
    def configure_optimizers(self):
        # optimizer2 = torch.optim.Adam(self.feature_extractor.parameters(), lr=self.learning_rate)
        # optimizer1 = torch.optim.SGD(self.feature_extractor.parameters(), lr=0.002, momentum=0.9)
        optimizer2 = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # Decay LR by a factor of 0.1 every 7 epochs
         #scheduler1 = lr_scheduler.StepLR(optimizer1, step_size=7, gamma=0.1)
        # return torch.optim.SGD(self.feature_extractor.parameters(), lr=self.learning_rate, momentum=0.9)
        return optimizer2
        # return (
        #     # {'optimizer': optimizer1, 'lr_scheduler': scheduler1, 'monitor': 'metric_to_track'}
        #     {'optimizer': optimizer1, 'lr_scheduler': scheduler1}
        #     # {'optimizer': optimizer2, 'lr_scheduler': scheduler2},
        # )
