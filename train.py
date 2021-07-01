"""
Here, we will run everything that is related to the training procedure.
"""

import time
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from tqdm import tqdm
from my_utils import train_utils
from torch.utils.data import DataLoader
from my_utils.types import Metrics
from my_utils.train_utils import TrainParams
from my_utils.train_logger import TrainLogger


def get_metrics(best_eval_score: float, eval_acc: float, eval_iou: float, eval_loss: float) -> Metrics:
    """
    Example of metrics dictionary to be reported to tensorboard. Change it to your metrics
    :param best_eval_score: average of accuracy and iou
    :param eval_acc: accuracy on eval dataset
    :param eval_iou: iou on eval dataset
    :prarm eval_loss: loss on eval dataset
    :return:
    """
    return {'Metrics/BestAvgScore': best_eval_score,
            'Metrics/LastAccuracy': eval_acc,
            'Metrics/Lastiou': eval_iou,
            'Metrics/LastCombinedLoss': eval_loss}


def train(model: nn.Module, train_loader: DataLoader, eval_dataloaders: dict, datasets_sizes: dict,
          train_params: TrainParams,
          logger: TrainLogger, data_types: list, use_schedular=True) -> Metrics:
    """
    Training procedure.
    :param model: torch initialized model
    :param train_loader: train loader
    :param eval_dataloaders: dict with 2 dataloaders- one with train data and one with test data
    :param datasets_sizes: dictionary with sizes of each dataset
    :param train_params: parameters for training such as lr, gamma, etc.
    :param logger: train loger object
    :param data_types: list with ['train', 'test']
    :param use_schedular: bool parameter that defines wether to use schedular or not
    :return: metrics from the final epoch - used for the logger
    """
    metrics = train_utils.get_zeroed_metrics_dict()
    best_eval_score = 0.0

    # Create optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=1e-3)

    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=train_params.lr_step_size,
                                                gamma=train_params.lr_gamma)

    # Define device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for epoch in tqdm(range(train_params.num_epochs)):
        train_loss = []
        start_epoch = time.time()

        model.train()
        for i, (images, targets) in enumerate(train_loader):
            if i % 100 == 0:
                print(i)

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items() if k != 'image_id'} for t in targets]
            preds, loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            train_loss.append(loss_value)

            # zero the parameter gradients & backward
            optimizer.zero_grad()
            losses.backward()

            # Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)

            # Optimization step
            optimizer.step()

        # Learning rate scheduler step
        if use_schedular:
            scheduler.step()

        losses_to_print = [(k, v.item()) for k, v in loss_dict.items()]
        print(
            f'Epoch number {epoch} mean loss: {np.mean(train_loss)}, final batch sum of losses: {loss_value}, final batch losses: {losses_to_print}')

        # Evaluation of epoch
        model.eval()
        with torch.no_grad():
            for data_type in data_types:
                epoch_loss = 0.0
                epoch_acc = 0.0
                epoch_iou = 0.0

                for images, targets in eval_dataloaders[data_type]:
                    images = list(image.to(device) for image in images)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                    preds, loss_dict = model(images, targets)  # preds: [{boxes, labels, scores}]
                    batch_loss, batch_iou, batch_acc = train_utils.calc_metrics(loss_dict, preds, targets)
                    epoch_loss += batch_loss.item()
                    epoch_acc += batch_acc
                    epoch_iou += batch_iou

                # Calculate metrics
                metrics[f'{data_type}']['combined_loss'].append(epoch_loss / datasets_sizes[data_type])
                metrics[f'{data_type}']['acc'].append(100 * (epoch_acc / datasets_sizes[data_type]))
                metrics[f'{data_type}']['iou'].append(epoch_iou / datasets_sizes[data_type])

        epoch_time = time.time() - start_epoch
        logger.write_epoch_statistics(epoch, epoch_time,
                                      metrics['train']['acc'][-1], metrics['test']['acc'][-1],
                                      metrics['train']['iou'][-1], metrics['test']['iou'][-1],
                                      metrics['train']['combined_loss'][-1], metrics['test']['combined_loss'][-1])

        scalars = {'Accuracy/Train': metrics['train']['acc'][-1],
                   'Accuracy/Validation': metrics['test']['acc'][-1],
                   'IoU/Train': metrics['train']['iou'][-1],
                   'IoU/Validation': metrics['test']['iou'][-1],
                   'Combined_Loss/Train': metrics['train']['combined_loss'][-1],
                   'Combined_Loss/Validation': metrics['test']['combined_loss'][-1],
                   }

        logger.report_scalars(scalars, epoch)

        eval_avg_score = ((metrics['test']['acc'][-1] / 100) + metrics['test']['iou'][-1]) / 2
        if eval_avg_score > best_eval_score:
            best_eval_score = eval_avg_score
            if train_params.save_model:
                logger.save_model(model, epoch, optimizer)

    plot_graphs(metrics, train_params.num_epochs)

    return get_metrics(best_eval_score,
                       metrics['test']['acc'][-1], metrics['test']['iou'][-1],
                       metrics['test']['combined_loss'][-1])


def plot_graphs(metrics, num_epochs):
    loss_df = pd.DataFrame({'train': metrics['train']['combined_loss'], 'test': metrics['test']['combined_loss']})

    train_sum_score = [acc + iou for acc, iou in
                       zip([acc / 100 for acc in metrics['train']['acc']], metrics['train']['iou'])]
    train_score = [sum / 2 for sum in train_sum_score]
    test_sum_score = [acc + iou for acc, iou in
                      zip([acc / 100 for acc in metrics['test']['acc']], metrics['test']['iou'])]
    test_score = [sum / 2 for sum in test_sum_score]
    score_df = pd.DataFrame({'train': train_score, 'test': test_score})

    acc_df = pd.DataFrame({'train': metrics['train']['acc'], 'test': metrics['test']['acc']})

    iou_df = pd.DataFrame({'train': metrics['train']['iou'], 'test': metrics['test']['iou']})

    melted_loss = pd.melt(loss_df.reset_index(), 'index').rename(
        columns={'index': 'Epoch', 'variable': 'Phase', 'value': 'Loss'})
    melted_score = pd.melt(score_df.reset_index(), 'index').rename(
        columns={'index': 'Epoch', 'variable': 'Phase', 'value': 'Score'})
    melted_acc = pd.melt(acc_df.reset_index(), 'index').rename(
        columns={'index': 'Epoch', 'variable': 'Phase', 'value': 'Acc'})
    melted_iou = pd.melt(iou_df.reset_index(), 'index').rename(
        columns={'index': 'Epoch', 'variable': 'Phase', 'value': 'IoU'})

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, figsize=(10, 15))

    sns.lineplot(x='Epoch', y='Loss', hue='Phase', data=melted_loss, ax=ax1).set_xticks(range(0, num_epochs + 1))
    sns.lineplot(x='Epoch', y='Score', hue='Phase', data=melted_score, ax=ax2).set_xticks(range(0, num_epochs + 1))
    sns.lineplot(x='Epoch', y='Acc', hue='Phase', data=melted_acc, ax=ax3).set_xticks(range(0, num_epochs + 1))
    sns.lineplot(x='Epoch', y='IoU', hue='Phase', data=melted_iou, ax=ax4).set_xticks(range(0, num_epochs + 1))

    plt.show()
