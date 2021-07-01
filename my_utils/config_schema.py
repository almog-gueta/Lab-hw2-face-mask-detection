"""
Schema for cfg file
"""

CFG_SCHEMA = {
    'main': {
        'experiment_name_prefix': str,
        'seed': int,
        'trains': bool,
        'paths': {
            'train': str,
            'test': str,
            'example': str,
            'logs': str,
        },
        'max_img_size': int,
    },
    'train': {
        'num_epochs': int,
        'batch_size': int,
        'save_model': bool,
        'lr': {
            'lr_value': float,
            'lr_decay': int,
            'lr_gamma': float,
            'lr_step_size': int,
        },
    },
    'data': {
            'img_size': int,
        },
}
