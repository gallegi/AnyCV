from types import SimpleNamespace


def get_default_configs():
    cfg = SimpleNamespace(**{})
    cfg.model_dir = 'models/'

    cfg.device = 'cuda:0'
    cfg.batch_size = 32
    cfg.num_workers = 4
    cfg.base_lr = 5e-5
    cfg.warmup_factor = 10
    cfg.num_epoch = 50
    cfg.folds_to_run = [0]
    cfg.patience = 10
    cfg.seed = 67
    cfg.amp = True

    cfg.ver_note = 'v1'
    cfg.sample = None

    cfg.backbone = 'tf_efficientnet_b3_ns'
    cfg.backbone_pretrained = True

    return cfg
