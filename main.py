"""
Main file
We will run the whole program from here
"""

import torch
import hydra

from train import train
from dataset import FaceMaskDataset
from torch.utils.data import DataLoader
from my_utils import main_utils, train_utils, data_utils
from my_utils.train_logger import TrainLogger
from omegaconf import DictConfig, OmegaConf

torch.backends.cudnn.benchmark = True


@hydra.main(config_path="cfg", config_name='cfg')
def main(cfg: DictConfig) -> None:
    """
    Run the code following a given configuration
    :param cfg: configuration file retrieved from hydra framework
    """

    scheduler_flag = True
    backbone = 'resnext50_32x4d'

    main_utils.init(cfg)
    logger = TrainLogger(exp_name_prefix=cfg['main']['experiment_name_prefix'], logs_dir=cfg['main']['paths']['logs'])
    logger.write(OmegaConf.to_yaml(cfg))

    # Set seed for results reproduction
    main_utils.set_seed(cfg['main']['seed'])

    # Load dataset
    data_types = ['train', 'test']
    train_dataset = FaceMaskDataset(image_dir=cfg['main']['paths']['train'], img_size=cfg['data']['img_size'],
                                    phase='train')
    eval_datasets = {
        data_type: FaceMaskDataset(image_dir=cfg['main']['paths'][data_type], img_size=cfg['data']['img_size'],
                                   phase='eval') for
        data_type in data_types}
    datasets_sizes = {data_type: len(eval_datasets[data_type]) for data_type in data_types}

    batch_size = cfg['train']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=0, collate_fn=data_utils.collate_fn)
    eval_dataloaders = {data_type: DataLoader(eval_datasets[data_type], batch_size, shuffle=False, num_workers=0,
                                              collate_fn=data_utils.collate_fn) for data_type in data_types}

    # Init model
    model = main_utils.init_rcnn_model(max_size=cfg['main']['max_img_size'], change_backbone=True, backbone=backbone)

    print('is cuda: ', torch.cuda.is_available())
    if torch.cuda.is_available():
        model.to('cuda')

    logger.write(main_utils.get_model_string(model))

    # Run model
    train_params = train_utils.get_train_params(cfg)

    # Report metrics and hyper parameters to tensorboard
    metrics = train(model, train_loader, eval_dataloaders, datasets_sizes, train_params, logger, data_types,
                    use_schedular=scheduler_flag)
    hyper_parameters = main_utils.get_flatten_dict(cfg['train'])

    logger.report_metrics_hyper_params(hyper_parameters, metrics)


if __name__ == '__main__':
    main()
