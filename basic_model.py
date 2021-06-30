from dataset import FaceMaskDataset
from torch.utils.data import DataLoader
from utils import train_utils
from utils.train_logger import TrainLogger
from omegaconf import DictConfig, OmegaConf
import hydra
import torch.nn as nn
import torch
from tqdm import tqdm
from torchvision import models
import torch.nn.functional as F
from utils.train_utils import calc_batch_iou
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class combined_model(nn.Module):
    def __init__(self):
        super(combined_model, self).__init__()
        resnet = models.resnet34(pretrained=False)
        layers = list(resnet.children())[:8]
        self.features1 = nn.Sequential(*layers[:6])
        self.features2 = nn.Sequential(*layers[6:])
        self.classifier = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 2))
        self.bb = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 4))

    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = F.relu(x)
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = x.view(x.shape[0], -1)
        # print(self.classifier(x).shape)
        return self.classifier(x), self.bb(x)

def update_optimizer(optimizer, lr):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr




@hydra.main(config_path="cfg", config_name='cfg')
def main(cfg: DictConfig) -> None:
    logger = TrainLogger(exp_name_prefix=cfg['main']['experiment_name_prefix'], logs_dir=cfg['main']['paths']['logs'])
    logger.write(OmegaConf.to_yaml(cfg))

    print("strat precess")
    train_dataset = FaceMaskDataset(image_dir=cfg['main']['paths']['train'], img_size=cfg['data']['img_size'],
                                    phase='train')

    eval_datasets = FaceMaskDataset(image_dir=cfg['main']['paths']['test'], img_size=cfg['data']['img_size'], phase='eval')

    batch_size = cfg['train']['batch_size']
    print("dataloader")
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True,
                              num_workers=cfg['main']['num_workers'])
    test_loader = DataLoader(eval_datasets, batch_size, shuffle=False)
    epochs = 40
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = combined_model().cuda()
    print(model)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.006)
    train_epocs(model,optimizer,train_loader,test_loader,logger,epochs)

def train_epocs(model, optimizer, train_dl, valid_dl,logger, epochs=10,C=1000):
    idx = 0
    metrics = train_utils.get_zeroed_metrics_dict()
    best_eval_score = 0.0
    for e in tqdm(range(epochs)):
        model.train()
        total = 0
        sum_loss = 0
        correct = 0
        ious_sum = 0.0
        print("train")
        for x, y_bb,y_class in train_dl:
            batch = y_class.shape[0]
            x = x.cuda().float()
            y_class = y_class.cuda()
            y_bb = y_bb.cuda().float()
            out_class, out_bb = model(x)
            # print(f"pred shape:{out_class.shape}, true shape {y_class.shape}")
            loss_class = F.cross_entropy(out_class, y_class, reduction="sum")
            loss_bb = F.l1_loss(out_bb, y_bb, reduction="none").sum(1)
            loss_bb = loss_bb.sum()
            loss = loss_class + loss_bb/C
            _, pred = torch.max(out_class, 1)
            ious_sum += calc_batch_iou(y_bb, out_bb)
            correct += pred.eq(y_class).sum().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            idx += 1
            total += batch
            sum_loss += loss.item()
        train_loss = sum_loss/total
        train_acc = correct / total
        train_iou = ious_sum / total
        print("test")
        val_loss, val_acc,val_iou,eval_avg_score = val_metrics(model, valid_dl, C)
        print("Train: train_loss %.3f train_acc: %.3f train_iou: %.3f\nVal: val_loss %.3f val_acc %.3f val_iou %.3f eval_avg_score %.3f\n" % (train_loss,train_acc,train_iou, val_loss, val_acc,val_iou,eval_avg_score))
        metrics=update_metrics(metrics, train_loss, val_loss, train_acc, val_acc, train_iou, val_iou)
        if eval_avg_score > best_eval_score:
            best_eval_score = eval_avg_score
            logger.save_model(model, e, optimizer)
            print("model saved")
    plot_graphs(metrics, epochs)
    # return sum_loss/total

def val_metrics(model, valid_dl, C=1000):
    with torch.no_grad():
        model.eval()
        total = 0
        sum_loss = 0
        correct = 0
        ious_sum = 0.0
        for x, y_bb, y_class in valid_dl:
            batch = y_class.shape[0]
            x = x.cuda().float()
            y_class = y_class.cuda()
            y_bb = y_bb.cuda().float()
            out_class, out_bb = model(x)
            loss_class = F.cross_entropy(out_class, y_class, reduction="sum")
            loss_bb = F.l1_loss(out_bb, y_bb, reduction="none").sum(1)
            loss_bb = loss_bb.sum()
            loss = loss_class + loss_bb/C
            _, pred = torch.max(out_class, 1)
            ious_sum += calc_batch_iou(y_bb, out_bb)
            correct += pred.eq(y_class).sum().item()
            sum_loss += loss.item()
            total += batch
        acc=correct/total
        iou=ious_sum/total
        eval_avg_score = (acc + iou) / 2

    return sum_loss/total, acc,iou,eval_avg_score

def update_metrics(metrics,train_loss,val_loss,train_acc,val_acc,train_iou,val_iou):
    #train
    metrics["train"]['combined_loss'].append(train_loss)
    metrics["train"]['acc'].append(train_acc)
    metrics["train"]['iou'].append(train_iou)

    #test
    metrics["test"]['combined_loss'].append(val_loss)
    metrics["test"]['acc'].append(val_acc)
    metrics["test"]['iou'].append(val_iou)
    return metrics

def plot_graphs(metrics, num_epochs):

    loss_df = pd.DataFrame({'train':metrics['train']['combined_loss'], 'test':metrics['test']['combined_loss']})

    train_sum_score = [acc+iou for acc, iou in zip([acc for acc in metrics['train']['acc']], metrics['train']['iou'])]
    train_score = [sum/2 for sum in train_sum_score]
    test_sum_score = [acc+iou for acc, iou in zip([acc for acc in metrics['test']['acc']], metrics['test']['iou'])]
    test_score = [sum / 2 for sum in test_sum_score]
    score_df = pd.DataFrame({'train': train_score, 'test': test_score})

    acc_df = pd.DataFrame({'train':metrics['train']['acc'], 'test':metrics['test']['acc']})

    iou_df = pd.DataFrame({'train':metrics['train']['iou'], 'test':metrics['test']['iou']})

    melted_loss = pd.melt(loss_df.reset_index(), 'index').rename(
        columns={'index': 'Epoch', 'variable': 'Phase', 'value': 'Loss'})
    melted_score = pd.melt(score_df.reset_index(), 'index').rename(
        columns={'index': 'Epoch', 'variable': 'Phase', 'value': 'Score'})
    melted_acc = pd.melt(acc_df.reset_index(), 'index').rename(
        columns={'index': 'Epoch', 'variable': 'Phase', 'value': 'Acc'})
    melted_iou = pd.melt(iou_df.reset_index(), 'index').rename(
        columns={'index': 'Epoch', 'variable': 'Phase', 'value': 'IoU'})

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, figsize=(10, 15))

    sns.lineplot(x='Epoch', y='Loss', hue='Phase', data=melted_loss, ax=ax1).set_xticks(range(0, num_epochs+1))
    sns.lineplot(x='Epoch', y='Score', hue='Phase', data=melted_score, ax=ax2).set_xticks(range(0, num_epochs+1))
    sns.lineplot(x='Epoch', y='Acc', hue='Phase', data=melted_acc, ax=ax3).set_xticks(range(0, num_epochs+1))
    sns.lineplot(x='Epoch', y='IoU', hue='Phase', data=melted_iou, ax=ax4).set_xticks(range(0, num_epochs+1))

    plt.show()


if __name__ == '__main__':
    main()

