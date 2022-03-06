from commons.configs import get_default_configs
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from types import SimpleNamespace

cfg = get_default_configs()
cfg.tensor_inp_shape = (512, 512, 3)
cfg.loss = 'adaptive_arcface'
cfg.pool = "avg"
cfg.n_classes = 1000
cfg.arcface_s = 45
cfg.arcface_m = 0.3
cfg.embedding_size = 512

# augmentations
cfg.train_aug = A.Compose([
        A.Resize(cfg.tensor_inp_shape[0], cfg.tensor_inp_shape[1]),
        A.Normalize(),
        ToTensorV2(p=1.0),
    ])

cfg.val_aug = A.Compose([
        A.Resize(cfg.tensor_inp_shape[0], cfg.tensor_inp_shape[1]),
        A.Normalize(),
        ToTensorV2(p=1.0),
    ])