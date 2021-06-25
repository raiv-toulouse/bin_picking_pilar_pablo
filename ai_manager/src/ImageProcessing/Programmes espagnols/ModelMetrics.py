# import libraries
import itertools
import os
import io
from collections import Iterable
import pathlib
from typing import Optional

import torch
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torchvision
from pytorch_lightning import seed_everything, metrics
from pytorch_lightning.metrics import classification
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.metrics.functional import stat_scores, f1, confusion_matrix, \
    precision, precision_recall, fbeta
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve
import torch.nn.functional as F
from torch.nn import Module
BN_TYPES = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)
from torch.utils.tensorboard import SummaryWriter

from CNN import CNN
from MyImageModule import MyImageModule

torch.set_printoptions(linewidth=120)
MODEL_CKPT_PATH = 'model/'
MODEL_CKPT = 'model/model-{epoch:02d}-{val_loss:.2f}'

# --- FUNCTIONS ----
from MyImageModule import MyImageModule
from ImageModel import ImageModel


class ModelMetrics():
    def __init__(self, preds, targets, name_model, parent_model):
        super(ModelMetrics, self).__init__()
        self.preds = preds
        self.targets = targets
        self.parent_model = parent_model
        self.name_model = name_model
        current_path = os.path.dirname(os.path.realpath(__file__))
        # define directory for the metrics file
        self.CSV_PATH = os.path.join(current_path,
                                     f'models_metrics/{self.parent_model}/%s_metrics/%s_metrics.csv' % (
                                     self.name_model, self.name_model))
        self.METRICS_FIGURES_PATH = os.path.join(current_path,
                                                 f'models_metrics/{self.parent_model}/%s_metrics' % self.name_model)
        # define the directory for the images
        pathlib.Path(self.METRICS_FIGURES_PATH).mkdir(parents=True, exist_ok=True)
        # define the directory for the plots
        self.ROC_PATH = os.path.join(self.METRICS_FIGURES_PATH, 'ROC_Curve.png')
        self.Precision_Recall_PATH = os.path.join(self.METRICS_FIGURES_PATH, 'Precision_Recall_Curve.png')
        self.CM_PATH = os.path.join(self.METRICS_FIGURES_PATH, 'Confusion_Matrix.png')

    # --- PLOTS ----
    # Plot ROC Curve
    def plot_ROC_curve(self, pos_label):
        # Compute ROC curve and ROC area for each class
        y_true = self.targets.detach().numpy()
        y_pred = (self.preds.argmax(dim=1)).detach().numpy()

        fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=pos_label)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Curve ROC Label {}'.format(pos_label))
        plt.legend(loc="lower right")
        plt.savefig(self.ROC_PATH, format='png')
        plt.close()
        return roc_auc

    # Plot confusion matrix
    def plot_confusion_matrix(self, cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        # print(cm)
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(self.CM_PATH, format='png')
        plt.close()

    # Plot Precision-Recall Curve
    def plot_precision_recall_curve(self, recall, precision):
        plt.figure()
        plt.plot(recall, precision, color='r', alpha=0.99)
        plt.fill_between(recall, precision, alpha=0.2, color='b', step='post')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        # plt.legend(loc="lower right")
        plt.title('Precision Recall Curve')
        plt.savefig(self.Precision_Recall_PATH, format='png')
        plt.close()

    # Get number of predictions correct
    def get_num_correct(self):
        return self.preds.argmax(dim=1).eq(self.targets).sum().item()

    # Get Precision_recall_curve and plot
    def get_precision_recall_curve(self, pos_label=1, display=True):
        precision, recall, _ = precision_recall_curve(torch.tensor(self.preds.argmax(dim=1)),
                                                      torch.tensor(self.targets),
                                                      pos_label)
        if display:
            self.plot_precision_recall_curve(recall, precision)
        return precision, recall

    # ROC Metric and ROC Curve
    def get_ROC_curve(self, pos_label=1):
        roc_auc = self.plot_ROC_curve(pos_label)
        return roc_auc

    # Get Stats_score
    def get_stats_score(self, class_index=1):
        tp, fp, tn, fn, sup = stat_scores(self.preds, self.targets, class_index)
        return tp, fp, tn, fn, sup

    # F1 Score
    def get_f1_score(self):
        f1_score = f1(self.preds, self.targets, num_classes=2, average='none')
        return f1_score

    # Confusion matrix and plot
    def get_confusion_matrix(self, display=True):
        cm = confusion_matrix(self.preds.argmax(dim=1), self.targets, num_classes=2)
        classes = find_classes(dir='./images/')
        if display:
            self.plot_confusion_matrix(cm.int(), classes)
        return cm.int()

    # Get and display all the metrics. The metrics are saved in a file
    def get_test_metrics(self, display=True):
        # Get Precision - Recall
        output = precision_recall(self.preds, self.targets, num_classes=2, class_reduction='none')
        precision = output[0].numpy()
        recall = output[1].numpy()
        # Get Precision-Recall Curve
        precision_curve, recall_curve = self.get_precision_recall_curve(pos_label=1, display=display)
        # Confusion Matrix
        cm = self.get_confusion_matrix(display=display)
        # F1 Score
        f1_score = self.get_f1_score()
        # F0.5 score
        f05_score = fbeta(self.preds, self.targets, num_classes=2, beta=0.5, threshold=0.5, average='none',
                          multilabel=False)
        # F2 Score
        f2_score = fbeta(self.preds, self.targets, num_classes=2, beta=2, threshold=0.5, average='none',
                         multilabel=False)
        # Stats_score - Class 0
        tp_0, fp_0, tn_0, fn_0, sup_0 = self.get_stats_score(class_index=0)
        # Stats_score - Class 1
        tp_1, fp_1, tn_1, fn_1, sup_1 = self.get_stats_score(class_index=1)
        # ROC Curve
        roc_auc_0 = self.get_ROC_curve(pos_label=0)
        roc_auc_1 = self.get_ROC_curve(pos_label=1)
        # Classification Report
        report = classification_report(self.targets.detach().numpy(), (self.preds.argmax(dim=1)).detach().numpy(),
                                       output_dict=True)
        print("Confusion Matrix")
        print(cm)
        print("Classification Report")
        print(report)

        # Variables are saved in a file
        # List of metric, value for class 0, value for class 1
        metric = ['Precision', 'Recall', 'F1 Score', 'F0.5 Score', 'F2_Score', 'TP', 'FP', 'TN', 'FN', 'ROC']
        value_class0 = [precision[0], recall[0], f1_score[0].numpy(), f05_score[0].numpy(), f2_score[0].numpy(), tp_0,
                        fp_0, tn_0, fn_0, roc_auc_0]
        value_class1 = [precision[1], recall[1], f1_score[1].numpy(), f05_score[1].numpy(), f2_score[1].numpy(), tp_1,
                        tp_1, tn_1, fn_1, roc_auc_1]
        # Dictionary of lists
        dict = {'Metric': metric, 'Class 0': value_class0, 'Class1': value_class1}
        df = pd.DataFrame(dict)
        # dictionary of report
        df_report = pd.DataFrame(report)
        # Saving the dataframe
        df.to_csv(self.CSV_PATH, header=True, index=False)
        df_report.to_csv(self.CSV_PATH, mode='a', header=True, index=False)


@torch.no_grad()
def evaluate(self, model, loader):
    y_true = []
    y_pred = []
    for imgs, labels in loader:
        logits = model(imgs)
        y_true.extend(labels)
        y_pred.extend(logits.detach().numpy())
    return np.array(y_true), np.array(y_pred)


# make a class prediction for one row of data
@torch.no_grad()
def predict(model, loader):
    # convert row to data
    y_pred = []
    y_true = []
    for img, labels in loader:
        logits = model(img)
        # print(logits[1])
        y_pred.extend(logits[1].detach().numpy())
        y_true.extend(labels)
    return np.array(y_true), np.array(y_pred)


@torch.no_grad()
def get_all_preds(model, loader):
    with torch.no_grad():
        all_preds = torch.tensor([])
        all_targets = torch.tensor([])
        for batch in loader:
            images, labels = batch
            preds = model(images)
            all_preds = torch.cat((all_preds, preds[1]), dim=0)
            all_targets = torch.cat((all_targets, labels), dim=0)
    print(torch.exp(all_preds))
    return all_preds, all_targets

# def get_single_pred()


def find_classes(dir):
    classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes


def add_pr_curve_tensorboard(writer, class_index, test_probs, test_preds, classes, global_step=0):
    '''
    Takes in a "class_index" from 0 to 9 and plots the corresponding
    precision-recall curve
    '''
    tensorboard_preds = test_preds == class_index
    tensorboard_probs = test_probs[:, class_index]

    writer.add_pr_curve(classes[class_index],
                        tensorboard_preds,
                        tensorboard_probs,
                        global_step=global_step)
    writer.close()


@torch.no_grad()
def get_probabilities(model, testloader):
    class_probs = []
    class_preds = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            output = model(images)
            class_probs_batch = [F.softmax(el, dim=0) for el in output[1]]
            _, class_preds_batch = torch.max(output[1], 1)

            class_probs.append(class_probs_batch)
            class_preds.append(class_preds_batch)

        test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
        test_preds = torch.cat(class_preds)

    return test_probs, test_preds


def show_activations(model):
    # _layers = list(model.children())[:-1]
    _layers = list(model.feature_extractor.children())[:-1]
   #  _sublayers = list((model.feature_extractor.children()).children())
    # _layers = _layers[:-1]
    print(_layers)
    # print(_sublayers)
    # for layer in _layers:
    #     # print("layer", layer)
    #     if isinstance(layer, Iterable):
    #         for i in layer:
    #             # _sublayers = list(i)
    #             print("%i", i)
    #             if isinstance(i, Iterable):
    #                 for sublayer in _sublayers:
    #                     print("sub_sublayer", sublayer)

    # else:
    #     print(layer)


# --MAIN ------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # instantiate class to handle model
    image_model = ImageModel(model_name='resnet50', dataset_size=2692)
    # Initialize Image Module
    # image_module = MyImageModule(dataset_size=100, batch_size=32)
    image_module = MyImageModule(dataset_size=2692, batch_size=1)
    image_module.setup()

    # --- PREDICT RESULTS ---
    # Get name and model used for testing
    # name_model, inference_model = image_model.inference_model()
    name_model = 'model-epoch=05-val_loss=0.36-weights7y3_unfreeze2.ckpt'
    inference_model = image_model.load_model(name_model)
    # print("Inference model:", inference_model)
    print("Name:", name_model)

    # Prediction with no tensors
    # y_true, y_pred = predict(inference_model, image_module.test_dataloader())
    # print("y_true", y_true)
    # print("y_pred", y_true)

    # Predictions with tensors
    test_preds, test_targets = get_all_preds(inference_model, image_module.test_dataloader())
    # print("Test preds:", test_preds)
    # print("Test_targets", test_targets)

    # --- TESTING METRICS ---
    metrics = ModelMetrics(test_preds, test_targets, name_model, parent_model='resnet50')
    # preds_correct = metrics.get_num_correct()
    # print('total correct:', preds_correct)
    # print('accuracy:', preds_correct / len(image_module.test_data))
    metrics.get_test_metrics(display=True)

    # # Without tensors
    # preds_correct = get_num_correct(torch.Tensor(y_pred), torch.Tensor(y_true))
    # print("--Without tensors--")
    # print('total correct:', preds_correct)
    # print('accuracy:', preds_correct / len(image_module.test_data))

    # Confusion Matrix
    # cm = confusion_matrix(test_targets, test_preds.argmax(dim=1))
    # cm = confusion_matrix(torch.Tensor(y_true), torch.Tensor(y_pred).argmax(dim=1))
    # class_names = find_classes('./images/')
    # print(class_names)
    # plot_confusion_matrix(cm, class_names, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues)

    # test_probs, test_preds = get_probabilities(inference_model, image_module.test_dataloader())
    # print("test_preds", test_preds.shape)
    # print("test_probs", test_probs.shape)
    # plot all the pr curves
    # for i in range(len(class_names)):
    #     add_pr_curve_tensorboard(image_model.writer, i, test_probs, class_names, test_preds)

    # show_activations(inference_model)




