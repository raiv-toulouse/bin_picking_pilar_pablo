# import libraries

from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import os
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, random_split, Subset, WeightedRandomSampler


class MyImageModule(pl.LightningDataModule):

    def __init__(self, batch_size, dataset_size=None, data_dir="./images3/"):
        super().__init__()
        self.trains_dims = None
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.dataset_size = dataset_size
        self.transform = transforms.Compose([
            # you can add other transformations in this list
            # transforms.Grayscale(num_output_channels=1),
            transforms.CenterCrop(size=224),  # Ancien code, les images font 256x256
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

        # Used to correctly display images
        self.inv_trans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                                 std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                            transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                                 std=[1., 1., 1.]),
                                            ])


    def _calculate_weights(self, dataset):
        class_count = Counter(dataset.targets)
        print("Class fail:", class_count[0])
        print("Class success:", class_count[1])
        count = np.array([class_count[0], class_count[1]])
        weight = 1. / torch.Tensor(count)
        weight_samples = np.array([weight[t] for t in dataset.targets])
        return weight_samples

    def setup(self, stage=None):
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

        # self.train_data = TransformSubset(train_data, transform=self.augmentation)
        self.train_data = TransformSubset(train_data, transform=self.augmentation)
        self.val_data = TransformSubset(val_data, transform=self.transform)
        self.test_data = TransformSubset(test_data, transform=self.transform)

        # Ce code prend beaucoup de temps à s'exécuter.
        # print('Targets Train:', TransformSubset(self.train_data).count_targets())
        # print('Targets Val:', TransformSubset(self.val_data).count_targets())
        # print('Targets Test:', TransformSubset(self.test_data).count_targets())

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

    def plot_classes_images(self, images, labels):
        '''
        Generates matplotlib Figure along with images and labels from a batch
        To be used with : writer.add_figure('Title', plot_classes_preds(images, classes))
        '''
        # plot the images in the batch, along with predicted and true labels
        class_names = self.find_classes()
        nb_images = len(images)
        my_dpi = 96  # For my monitor (see https://www.infobyip.com/detectmonitordpi.php)
        fig = plt.figure(figsize=(nb_images * 256 / my_dpi, 256 / my_dpi), dpi=my_dpi)
        for idx in np.arange(nb_images):
            ax = fig.add_subplot(1, nb_images, idx + 1, xticks=[], yticks=[])
            img = self.inv_trans(images[idx])
            npimg = img.cpu().numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            ax.set_title(class_names[labels[idx]])
        return fig


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


# --- MAIN ----
if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter

    image_module = MyImageModule(batch_size=8)
    image_module.setup()
    images, labels = next(iter(image_module.val_dataloader()))
    #print(images[0].shape)
    grid = torchvision.utils.make_grid(images, nrow=8, padding=2)
    writer = SummaryWriter()
    writer.add_figure('images with labels', image_module.plot_classes_images(images, labels))
    writer.add_image('some_images', image_module.inv_trans(grid))
    writer.close()
