"""
Includes all utils related to training
"""

import torch

from typing import Dict
from torch import Tensor
from omegaconf import DictConfig
import numpy as np



def compute_score_with_logits(logits: Tensor, labels: Tensor) -> Tensor:
    """
    Calculate multiclass accuracy with logits (one class also works)
    :param logits: tensor with logits from the model
    :param labels: tensor holds all the labels
    :return: score for each sample
    """
    logits = torch.max(logits, 1)[1].data  # argmax

    logits_one_hots = torch.zeros(*labels.size())
    if torch.cuda.is_available():
        logits_one_hots = logits_one_hots.cuda()
    logits_one_hots.scatter_(1, logits.view(-1, 1), 1)

    scores = (logits_one_hots * labels)

    return scores


def get_zeroed_metrics_dict() -> Dict:
    """
    :return: dictionary to store all relevant metrics for training
    """
    return {'train': {'label_loss': [], 'bbox_loss': [], 'acc': [], 'iou': [], 'combined_loss': []},
            'test' : {'label_loss': [], 'bbox_loss': [], 'acc': [], 'iou': [], 'combined_loss': []}}

class TrainParams:
    """
    This class holds all train parameters.
    Add here variable in case configuration file is modified.
    """
    num_epochs: int
    lr: float
    lr_decay: float
    lr_gamma: float
    lr_step_size: int
    grad_clip: float
    save_model: bool

    def __init__(self, **kwargs):
        """
        :param kwargs: configuration file
        """
        self.num_epochs = kwargs['num_epochs']

        self.lr = kwargs['lr']['lr_value']
        self.lr_decay = kwargs['lr']['lr_decay']
        self.lr_gamma = kwargs['lr']['lr_gamma']
        self.lr_step_size = kwargs['lr']['lr_step_size']

        self.grad_clip = kwargs['grad_clip']
        self.save_model = kwargs['save_model']


def get_train_params(cfg: DictConfig) -> TrainParams:
    """
    Return a TrainParams instance for a given configuration file
    :param cfg: configuration file
    :return:
    """
    return TrainParams(**cfg['train'])


def calc_batch_iou(a, b, epsilon=1e-5):
    """
    code from http://ronny.rest/tutorials/module/localization_001/iou/
    :param a: tensor of [batch_size, 4]
    :param b: tensor of [batch_size, 4]
    :param epsilon: (float) Small value to prevent division by zero
    :return: sum of ious over batch (float)
    """

    #convert to numpy
    a = a.cpu().detach().numpy()
    b = b.cpu().detach().numpy()

    # COORDINATES OF THE INTERSECTION BOXES
    x1 = np.array([a[:, 0], b[:, 0]]).max(axis=0)
    y1 = np.array([a[:, 1], b[:, 1]]).max(axis=0)
    x2 = np.array([a[:, 0]+a[:, 2], b[:, 0]+b[:, 2]]).min(axis=0)
    y2 = np.array([a[:, 1]+a[:, 3], b[:, 1]+b[:, 3]]).min(axis=0)

    # AREAS OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)

    # handle case where there is NO overlap
    width[width < 0] = 0
    height[height < 0] = 0

    area_overlap = width * height

    # COMBINED AREAS
    area_a = a[:, 2] * a[:, 3]
    area_b = b[:, 2] * b[:, 3]
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined + epsilon)
    return iou.sum()