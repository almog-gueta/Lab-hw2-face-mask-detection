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
            'test': {'label_loss': [], 'bbox_loss': [], 'acc': [], 'iou': [], 'combined_loss': []}}


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
        self.save_model = kwargs['save_model']


def get_train_params(cfg: DictConfig) -> TrainParams:
    """
    Return a TrainParams instance for a given configuration file
    :param cfg: configuration file
    :return:
    """
    return TrainParams(**cfg['train'])


# def calc_batch_iou(a, b, epsilon=1e-5):
#     # NOTICE: a and b should be bboxes with [x,y,w,h] and not [x1,y1,x2,y2]
#     """
#     code from http://ronny.rest/tutorials/module/localization_001/iou/
#     :param a: tensor of [batch_size, 4]
#     :param b: tensor of [batch_size, 4]
#     :param epsilon: (float) Small value to prevent division by zero
#     :return: sum of ious over batch (float)
#     """
#
#     #convert to numpy
#     a = a.cpu().detach().numpy()
#     b = b.cpu().detach().numpy()
#
#     # COORDINATES OF THE INTERSECTION BOXES
#     x1 = np.array([a[:, 0], b[:, 0]]).max(axis=0)
#     y1 = np.array([a[:, 1], b[:, 1]]).max(axis=0)
#     x2 = np.array([a[:, 0]+a[:, 2], b[:, 0]+b[:, 2]]).min(axis=0)
#     y2 = np.array([a[:, 1]+a[:, 3], b[:, 1]+b[:, 3]]).min(axis=0)
#
#     # AREAS OF OVERLAP - Area where the boxes intersect
#     width = (x2 - x1)
#     height = (y2 - y1)
#
#     # handle case where there is NO overlap
#     width[width < 0] = 0
#     height[height < 0] = 0
#
#     area_overlap = width * height
#
#     # COMBINED AREAS
#     area_a = a[:, 2] * a[:, 3]
#     area_b = b[:, 2] * b[:, 3]
#     area_combined = area_a + area_b - area_overlap
#
#     # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
#     iou = area_overlap / (area_combined + epsilon)
#     return iou.sum()
#

def calc_iou(bbox_a, bbox_b):
    """
    Calculate intersection over union (IoU) between two bounding boxes with a (x, y, w, h) format.
    :param bbox_a: Bounding box A. 4-tuple/list.
    :param bbox_b: Bounding box B. 4-tuple/list.
    :return: Intersection over union (IoU) between bbox_a and bbox_b, between 0 and 1.
    """
    x1, y1, w1, h1 = bbox_a
    x2, y2, w2, h2 = bbox_b
    w_intersection = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_intersection = min(y1 + h1, y2 + h2) - max(y1, y2)
    if w_intersection <= 0.0 or h_intersection <= 0.0:  # No overlap
        return 0.0
    intersection = w_intersection * h_intersection
    union = w1 * h1 + w2 * h2 - intersection  # Union = Total Area - Intersection
    return intersection / union


def calc_metrics(loss, detections, targets):
    """
    calc accuracy, iou and loss metrics
    loss is a dictionary of losses returned from model
    """
    sum_loss = sum(loss.values())
    iou = 0
    acc = 0
    for detection, target in zip(detections, targets):
        pred_bbox = detection['boxes']
        pred_label = detection['labels']
        if pred_bbox.numel():
            pred_bbox = pred_bbox[0]
            pred_bbox[2] = pred_bbox[2] - pred_bbox[0]
            pred_bbox[3] = pred_bbox[3] - pred_bbox[1]
            true_bbox = target['boxes'][0].tolist()
            true_bbox[2] = true_bbox[2] - true_bbox[0]
            true_bbox[3] = true_bbox[3] - true_bbox[1]
            iou += calc_iou(pred_bbox, true_bbox)

            if pred_label == target['labels']:
                acc += 1

    if isinstance(iou, torch.Tensor):
        iou = iou.item()
    return sum_loss, iou, acc
