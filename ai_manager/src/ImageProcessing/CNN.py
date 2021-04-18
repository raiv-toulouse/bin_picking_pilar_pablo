
from typing import Generator
import torch
from torch.nn import Module
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.models as models
from typing import Optional
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.optimizer import Optimizer
from pytorch_lightning.metrics.functional import accuracy, precision, recall, confusion_matrix, f1, fbeta

BN_TYPES = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)

# --- UTILITY FUNCTIONS ----
# Extract from :
# https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/domain_templates/computer_vision_fine_tuning.py

def _make_trainable(module: Module) -> None:
    """Unfreezes a given module.
    Args:
        module: The module to unfreeze
    """
    for param in module.parameters():
        param.requires_grad = True
    module.train()


def freeze(module: Module,
           n: Optional[int] = None,
           train_bn: bool = True) -> None:
    """Freezes the layers up to index n (if n is not None).
    Args:
        module: The module to freeze (at least partially)
        n: Max depth at which we stop freezing the layers. If None, all
            the layers of the given module will be frozen.
        train_bn: If True, leave the BatchNorm layers in training mode
    """
    children = list(module.children())
    n_max = len(children) if n is None else int(n)

    for child in children[:n_max]:
        _recursive_freeze(module=child, train_bn=train_bn)

    for child in children[n_max:]:
        _make_trainable(module=child)


def _recursive_freeze(module: Module,
                      train_bn: bool = True) -> None:
    """Freezes the layers of a given module.
    Args:
        module: The module to freeze
        train_bn: If True, leave the BatchNorm layers in training mode
    """
    children = list(module.children())
    if not children:
        if not (isinstance(module, BN_TYPES) and train_bn):
            for param in module.parameters():
                param.requires_grad = False
            module.eval()
        else:
            # Make the BN layers trainable
            _make_trainable(module)
    else:
        for child in children:
            _recursive_freeze(module=child, train_bn=train_bn)


def filter_params(module: Module,
                  train_bn: bool = True) -> Generator:
    """Yields the trainable parameters of a given module.
    Args:
        module: A given module
        train_bn: If True, leave the BatchNorm layers in training mode
    Returns:
        Generator
    """
    children = list(module.children())
    if not children:
        if not (isinstance(module, BN_TYPES) and train_bn):
            for param in module.parameters():
                if param.requires_grad:
                    yield param
    else:
        for child in children:
            for param in filter_params(module=child, train_bn=train_bn):
                yield param


def _unfreeze_and_add_param_group(module: Module,
                                  optimizer: Optimizer,
                                  lr: Optional[float] = None,
                                  train_bn: bool = True):
    """Unfreezes a module and adds its parameters to an optimizer."""
    _make_trainable(module)
    params_lr = optimizer.param_groups[0]['lr'] if lr is None else float(lr)
    optimizer.add_param_group(
        {'params': filter_params(module=module, train_bn=train_bn),
         'lr': params_lr / 10.,
         })


# --- PYTORCH LIGHTNING MODULE ----
class CNN(pl.LightningModule):

    # defines the network
    def __init__(self,
                 learning_rate: float = 1e-3,
                 batch_size: int = 8,
                 input_shape: list = [3, 256, 256],
                 backbone: str = 'resnet18',
                 train_bn: bool = True,
                 milestones: tuple = (5, 10),
                 lr_scheduler_gamma: float = 1e-1,
                 num_workers: int = 6):

        super(CNN, self).__init__()
        # parameters
        self.save_hyperparameters()
        self.dim = input_shape
        # 'vgg16', 'resnet50', 'alexnet', 'resnet18', 'resnet34', 'squeezenet1_1', 'googlenet'
        self.backbone = backbone
        self.train_bn = train_bn
        self.milestones = milestones
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.num_workers = num_workers
        # self.lr = config["lr"]
        # self.batch_size = config["batch_size"]
        # build the model
        self.__build_model()

    def __build_model(self):
        """Define model layers & loss."""

        # 1. Load pre-trained network: choose the model for the pretrained network
        model_func = getattr(models, self.backbone)
        backbone = model_func(pretrained=True)
        # self.feature_extractor = model_func(pretrained=True)
        # print("BEFORE CUT")
        # _layers = list(backbone.children())
        # print(_layers)
        # print("AFTER CUT")
        _layers = list(backbone.children())[:-1]
        # print(_layers)
        self.feature_extractor = torch.nn.Sequential(*_layers)
        # print(self.feature_extractor)
        # If.eval() is used, then the layers are frozen.
        # self.feature_extractor.eval()
        # freeze(module=self.feature_extractor, train_bn=self.train_bn)
        # si queremos descongelar últimas capas
        freeze(module=self.feature_extractor, n=-2, train_bn=self.train_bn)
        # _unfreeze_and_add_param_group(
        #     module=self.feature_extractor[:-2], optimizer=optimizer, train_bn=self.train_bn
        #     )

        # 2. Adaptive layer:
        self.adaptive_layer = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.feature_extractor.add_module("adaptive_layer", self.adaptive_layer)
        # print(self.feature_extractor)

        # classes are two: success or failure
        num_target_classes = 2
        n_sizes = self._get_conv_output(self.dim)

        # 3. Classifier
        _fc_layers = [torch.nn.Linear(n_sizes, 256),
                      torch.nn.Linear(256, 32),
                      torch.nn.Linear(32, num_target_classes)]
        # self.fc = torch.nn.Sequential(*_fc_layers)
        # _fc_layers = [nn.Linear(n_sizes, num_target_classes)]
        self.fc = torch.nn.Sequential(*_fc_layers)

        # 4. Loss:
        # self.loss_func = F.cross_entropy

    # mandatory
    def forward(self, t):
        """Forward pass. Returns logits."""
        # 1. Feature extraction:
        t = self.feature_extractor(t)
        # print("t:", t.size())
        features = t.squeeze(-1).squeeze(-1)
        # print("Features", features.size())
        # 2. Classifier (returns logits):
        t = self.fc(features)
        # We want the probability to sum 1
        t = F.log_softmax(t, dim=1)
        return features, t


    # trainning loop
    def training_step(self, batch, batch_idx):
        # x = images , y = batch, logits = labels
        x, y = batch
        logits = self(x)
        # 2. Compute loss & metrics:
        return self._calculate_step_metrics(logits, y)

    def training_epoch_end(self, outputs):
        """Compute and log training loss and accuracy at the epoch level."""
        # computation graph add it during the first epoch only
        if self.current_epoch == 1:
            # sampleImg
            sampleImg = torch.rand((1, 3, 256, 256))
            self.logger.experiment.add_graph(CNN(), sampleImg)
        # logging histograms
        #self.custom_histogram_adder()
        # Calculate metrics
        loss_mean = self._calculate_epoch_metrics(outputs, name='Train')
        print(f"Epoch {self.current_epoch} : training_epoch_end : loss_mean = ", loss_mean.item())

    # validation loop
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        # 2. Compute loss & metrics:
        outputs = self._calculate_step_metrics(logits, y)
        self.log("val_loss", outputs["loss"])
        return outputs

    def validation_epoch_end(self, outputs):
        """Compute and log validation loss and accuracy at the epoch level."""
        loss_mean = self._calculate_epoch_metrics(outputs, name='Val')
        print("==> validation_epoch_end : loss_mean = ", loss_mean.item())

    # test loop
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        # 2. Compute loss & metrics:
        return self._calculate_step_metrics(logits, y)

    def test_epoch_end(self, outputs):
        loss_mean = self._calculate_epoch_metrics(outputs, name='Test')
        print("test_epoch_end : loss_mean = ", loss_mean.item())

    # define optimizers
    def configure_optimizers(self):
        # optimizer2 = torch.optim.Adam(self.feature_extractor.parameters(), lr=self.learning_rate)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)
        optimizer1 = torch.optim.SGD(self.parameters(), lr=0.002, momentum=0.9)
        scheduler = MultiStepLR(optimizer, milestones=self.milestones, gamma=self.lr_scheduler_gamma)
        # Decay LR by a factor of 0.1 every 7 epochs
        scheduler1 = lr_scheduler.StepLR(optimizer1, step_size=7, gamma=0.1)
        # return torch.optim.SGD(self.feature_extractor.parameters(), lr=self.learning_rate, momentum=0.9)
        return (
            # {'optimizer': optimizer1, 'lr_scheduler': scheduler1, 'monitor': 'metric_to_track'}
            {'optimizer': optimizer1, 'lr_scheduler': scheduler1}
            # {'optimizer': optimizer2, 'lr_scheduler': scheduler2},
        )


    # returns the size of the output tensor going into the Linear layer from the conv block.
    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def get_size(self):
        n_sizes = self._get_conv_output(self.dim)
        return n_sizes

    # returns the feature tensor from the conv block
    def _forward_features(self, x):
        x = self.feature_extractor(x)
        # print("Size last_layer", x.size())
        return x

    # loss function, weights modified to give more importance to class 1
    @staticmethod
    def _loss_function(logits, labels):

        weights = torch.tensor([7.0, 3.0]).cuda()
        loss = F.cross_entropy(logits, labels, weight=weights, reduction='mean')
        return loss

    def custom_histogram_adder(self):
        # A custom defined function that adds Histogram to TensorBoard
        # Iterating over all parameters and logging them
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

    # TODO: Refactor internal metrics
    def _calculate_step_metrics(self, logits, y):
        # prepare the metrics
        loss = self._loss_function(logits[1], y)
        # loss = F.cross_entropy(logits[1], y)
        preds = torch.argmax(logits[1], dim=1)
        num_correct = torch.eq(preds.view(-1), y.view(-1)).sum()
        acc = accuracy(preds, y)
        f1_score = f1(preds, y, num_classes=2, average='weighted')
        fb05_score = fbeta(preds, y, num_classes=2, average='weighted', beta=0.5)
        fb2_score = fbeta(preds, y, num_classes=2, average='weighted', beta=2)
        cm = confusion_matrix(preds, y, num_classes=2)
        prec = precision(preds, y, num_classes=2, class_reduction='weighted')
        rec = recall(preds, y, num_classes=2, class_reduction='weighted')
        # au_roc = auroc(preds, y, pos_label=1)
        return {'loss': loss,
                'acc': acc,
                'f1_score': f1_score,
                'f05_score': fb05_score,
                'f2_score': fb2_score,
                'precision': prec,
                'recall': rec,
                # 'auroc': au_roc,
                'confusion_matrix': cm,
                'num_correct': num_correct}

    def _calculate_epoch_metrics(self, outputs, name):

        # Logging activations
        loss_mean = torch.stack([output['loss']
                                 for output in outputs]).mean()
        acc_mean = torch.stack([output['num_correct']
                                for output in outputs]).sum().float()
        acc_mean /= (len(outputs) * self.batch_size)

        f1_score = torch.stack([output['f1_score']
                                for output in outputs]).mean()

        f05_score = torch.stack([output['f05_score']
                                 for output in outputs]).mean()
        f2_score = torch.stack([output['f2_score']
                                for output in outputs]).mean()
        precision = torch.stack([output['precision']
                                 for output in outputs]).mean()
        recall = torch.stack([output['recall']
                              for output in outputs]).mean()
        # Logging scalars
        self.logger.experiment.add_scalar(f'Loss/{name}',
                                          loss_mean,
                                          self.current_epoch)

        self.logger.experiment.add_scalar(f'Accuracy/{name}',
                                          acc_mean,
                                          self.current_epoch)

        self.logger.experiment.add_scalar(f'F1_Score/{name}',
                                          f1_score,
                                          self.current_epoch)
        self.logger.experiment.add_scalar(f'F05_Score/{name}',
                                          f05_score,
                                          self.current_epoch)
        self.logger.experiment.add_scalar(f'F2_Score/{name}',
                                          f2_score,
                                          self.current_epoch)
        self.logger.experiment.add_scalar(f'Precision/{name}',
                                          precision,
                                          self.current_epoch)
        self.logger.experiment.add_scalar(f'Recall/{name}',
                                          recall,
                                          self.current_epoch)
        return loss_mean
