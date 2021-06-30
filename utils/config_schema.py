"""
Schema for config file
"""

CFG_SCHEMA = {
    'main': {
        'experiment_name_prefix': str,
        'seed': int,
        'num_workers': int,
        'trains': bool,
        'paths': {
            'train': str,
            'test': str,
            'example': str,
            'logs': str,
        },
        "model_names": {
            "q_model_name": str,
            "v_model_name": str,
            "vqa_model_name": str,
        },
    },
    'train': {
        'num_epochs': int,
        'grad_clip': float,
        # 'dropout': float,
        # 'num_hid': int,
        'batch_size': int,
        'save_model': bool,
        'lr': {
            'lr_value': float,
            'lr_decay': int,
            'lr_gamma': float,
            'lr_step_size': float,
        },
    },
    'dataset': {
        'resize_h': int,
        'resize_w': int,
    },
    "class_model": {
        "cnn": {
            "dims": list,
            "fc_out": int,
            "is_atten": bool,
            "is_autoencoder": bool,
        },
    },
    "bbox_model": {
        "cnn": {
            "dims": list,
            "fc_out": int,
            "is_atten": bool,
            "is_autoencoder": bool,
        },
    },
    "v_model": {
        "cnn": {
            "dims": list,
            "kernel_size": int,
            "padding": int,
            "pool": int,
            "activation": str,
        },
        "attention_cnn": {
            "dims": list,
            "kernel_size": int,
            "padding": int,
            "pool": int,
            "fc_out": int,
            "activation": str,
            "is_atten": bool,
            "is_autoencoder": bool,
        },
    },
    "atten_model": {
      "projected_dim": int,
    },
    "vqa_model": {
      "basic": {
        "activation": str,
        "num_hid": int,
        "dropout": float,
        "is_concat": bool,
        },
      "basic_lstm_cnn": {
        "activation": str,
        "num_hid": int,
        "dropout": float,
        "is_concat": bool,
        },
      "atten_lstm_cnn": {
        "activation": str,
        "num_hid": int,
        "dropout": float,
        "is_concat": bool,
        },
    },
    'data': {
        'img_size': int,
    },
}