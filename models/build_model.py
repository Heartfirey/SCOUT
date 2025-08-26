import os
import torch
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Tuple
from torchvision import transforms

from utils import Logger
from config import Config
from .talnet import ModelEMA
from .sinet import SINet_ResNet50
from .sinetv2 import SINet_v2
from .fspnet import FSPNet
# from .active_components import VAE, Discriminator
from models.modules.nn_aug import GeometricAugmentation,ColorAugmentation
from models.modules.augmentation_container import AugmentationContainer

def build_model(config: Config) -> torch.nn.Module:
    if config.model_name == 'Default':
        model = ModelEMA(config=config, bb_pretrained=True)
    elif config.model_name == 'SINet':
        model = SINet_ResNet50(config=config)
    elif config.model_name == 'SINetv2':
        model = SINet_v2(config, channel=32, imagenet_pretrained=True)
    elif config.model_name == 'FSPNet':
        model = FSPNet(config)
    return model

def build_augmentation(config):
    normalizer = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if config.enable_g_aug:
        g_aug = GeometricAugmentation(config.g_scale, n_dim=config.n_dim)
    else:
        g_aug = None
    if config.enable_c_aug:
        c_aug = ColorAugmentation(config.c_scale, n_dim=config.n_dim)
    else:
        c_aug = None
    if not config.enable_g_aug and not config.enable_c_aug:
        g_aug = GeometricAugmentation(config.g_scale, n_dim=config.n_dim)
    augmentation = AugmentationContainer(c_aug, g_aug, config.c_reg_coef, normalizer)
    return augmentation

def build_model_optimizers(config: Config, logger: Logger, device: torch.device, resume: str=None) -> any:
    model = build_model(config)
    aug = build_augmentation(config)
    epoch_st = 0
    if resume is not None:
        if os.path.isfile(resume):
            logger.key_info("[+] Loading model checkpoint from '{}'".format(resume))
            state_dict = torch.load(resume, map_location='cpu')
            model.load_state_dict(state_dict['model'], strict=False)
            aug.load_state_dict(state_dict['aug'], strict=False)
            # epoch_st = state_dict.get('epoch', 0) + 1
        else:
            logger.warn_info("[!] No checkpoint found at '{}'".format(resume))
    
    if config.distributed_train:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        model = DDP(model, device_ids=[device], find_unused_parameters=True)
        aug = torch.nn.SyncBatchNorm.convert_sync_batchnorm(aug).to(device)
        aug = DDP(aug, device_ids=[device])
    else:
        model = model.to(device)
        aug = aug.to(device)
    
    if config.compile_model:
        model = torch.compile(model, mode=['default', 'reduce-overhead', 'max-autotune'][0])
        aug = torch.compile(aug, mode=['default', 'reduce-overhead', 'max-autotune'][0])
    if config.precisionHigh:
        torch.set_float32_matmul_precision('high')
    
    if config.optimizer == 'AdamW':
        optimizer = optim.AdamW(params=model.parameters(), lr=config.lr, weight_decay=1e-2)
    elif config.optimizer == 'Adam':
        optimizer = optim.Adam(params=model.parameters(), lr=config.lr, weight_decay=0)
    optimizer_aug = optim.AdamW(params=aug.parameters(), lr=1e-3, weight_decay=1e-2)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[lde if lde > 0 else config.tot_epochs + lde + 1 for lde in config.lr_decay_epochs],
        gamma=config.lr_decay_rate
    )
    
    logger.freeze_info("Optimizer details: {}".format(str(optimizer)))
    logger.freeze_info("Scheduler details: {}".format(str(lr_scheduler.state_dict())))
    
    return model, aug, optimizer, optimizer_aug, lr_scheduler, epoch_st

def build_model_eval(config: Config, logger: Logger, resume: str, device: torch.device='cpu') -> torch.nn.Module:
    model = build_model(config=config)
    logger.freeze_info("[+] Loading model from {} to evaluate...".format(resume))
    assert os.path.isfile(resume), "[x] target checkpoint not exists!"
    state_dict = torch.load(resume, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model = model.to(device)
    return model
    
