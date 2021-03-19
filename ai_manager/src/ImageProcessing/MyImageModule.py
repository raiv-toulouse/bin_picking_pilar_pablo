# import libraries

import math
import random
from collections import Counter
from copy import copy

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import os

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import pytorch_lightning as pl
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, Subset, WeightedRandomSampler
from pytorch_lightning import Trainer, seed_everything


class MyImageModule(pl.LightningDataModule):

    def __init__(self, batch_size, dataset_size=None):
        super().__init__()
        self.trains_dims = None
        self.batch_size = batch_size
        # self.data_dir = './images/'

        # test with CIFAR Pictures
        self.data_dir = './cifar/'

        self.dataset_size = dataset_size

    def _calculate_weights(self, dataset):
        class_count = Counter(dataset.targets)
        print("Class fail:", class_count[0])
        print("Class success:", class_count[1])
        count = np.array([class_count[0], class_count[1]])
        weight = 1. / torch.Tensor(count)
        weight_samples = np.array([weight[t] for t in dataset.targets])
        return weight_samples

    def setup(self, stage=None):
        self.transform = transforms.Compose([
            # you can add other transformations in this list
            # transforms.Grayscale(num_output_channels=1),
            transforms.CenterCrop(size=224),
            transforms.Resize(size=256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.augmentation = transforms.Compose([
            transforms.CenterCrop(size=224),
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Build Dataset
        dataset = datasets.ImageFolder(self.data_dir)
        weight_samples = self._calculate_weights(dataset)

        # Select a subset of the images
        indices = list(range(len(dataset)))
        dataset_size = len(dataset) if self.dataset_size is None else min(len(dataset), self.dataset_size)

        samples = list(WeightedRandomSampler(weight_samples, len(weight_samples),
                                             replacement=False,
                                             generator=torch.Generator().manual_seed(42)))[:dataset_size]
        # samples = list(SubsetRandomSampler(indices, generator=torch.Generator().manual_seed(42)))[:dataset_size]
        subset = Subset(dataset, indices=samples)

        train_size = int(0.7 * len(subset))
        val_size = int(0.5 * (len(subset) - train_size))
        test_size = int(len(subset) - train_size - val_size)

        train_data, val_data, test_data = random_split(subset,
                                                       [train_size, val_size, test_size],
                                                       generator=torch.Generator().manual_seed(42))

        print("Len Train Data", len(train_data))
        print("Len Val Data", len(val_data))
        print("Len Test Data", len(test_data))

        # Esto es una guarrada : https://stackoverflow.com/questions/51782021/how-to-use-different-data-augmentation-for-subsets-in-pytorch
        # train_data.dataset = copy(dataset)
        # val_data.dataset = copy(dataset)
        # test_data.dataset = copy(dataset)

        # Data Augmentation for Training
        # train_data.dataset.transform = self.augmentation
        # # Transform Data
        # val_data.dataset.transform = self.transform
        # test_data.dataset.transform = self.transform

        # self.train_data = TransformSubset(train_data, transform=self.augmentation)
        self.train_data = TransformSubset(train_data, transform=self.transform)
        self.val_data = TransformSubset(val_data, transform=self.transform)
        self.test_data = TransformSubset(test_data, transform=self.transform)

        print('Targets Train:', TransformSubset(self.train_data).count_targets())
        print('Targets Val:', TransformSubset(self.val_data).count_targets())
        print('Targets Test:', TransformSubset(self.test_data).count_targets())

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(self.train_data, num_workers=16, batch_size=self.batch_size)
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(self.val_data, num_workers=16, batch_size=self.batch_size)
        return val_loader

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(self.test_data, num_workers=16, batch_size=self.batch_size)
        return test_loader

    # TODO: Método para acceder a las clases
    def find_classes(self):
        classes = [d.name for d in os.scandir(self.data_dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes

    @staticmethod
    def _count_targets(subset):
        class_0 = 0
        class_1 = 0
        for tensor, target in subset:
            if target == 0:
                class_0 += 1
            else:
                class_1 += 1
        print('Count class 0:', class_0)
        print('Count class 1:', class_1)


class TransformSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

    def count_targets(self):
        count_class = [0, 0]
        for tensor, target in self.subset:
            if target == 0:
                count_class[0] += 1
            else:
                count_class[1] += 1
        return count_class
