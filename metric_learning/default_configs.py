from commons.configs import get_default_configs
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from types import SimpleNamespace

def get_configs():
    cfg = get_default_configs()

    cfg.tensor_inp_shape = (512, 512, 3)

    cfg.global_pool = "gem"
    cfg.embedding_size = 512

    # arcface
    cfg.margin = 45
    cfg.scale = 0.3
    cfg.adaptive_margin = False
    cfg.sub_center = False
    cfg.arcface_m_x = 0.45
    cfg.arcface_m_y = 0.05

    # dolg
    cfg.dolg_dilations = [6,12,18]

    # augmentations
    cfg.train_trans = A.Compose([
            A.Resize(cfg.tensor_inp_shape[0], cfg.tensor_inp_shape[1]),
            A.Normalize(),
            ToTensorV2(p=1.0),
        ])

    cfg.valid_trans = A.Compose([
            A.Resize(cfg.tensor_inp_shape[0], cfg.tensor_inp_shape[1]),
            A.Normalize(),
            ToTensorV2(p=1.0),
        ])

    return cfg