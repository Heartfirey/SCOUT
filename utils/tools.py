import os
import gc
import cv2
import time
import random
import torch
import functools
import numpy as np
from PIL import Image
from torchvision import transforms

from .logger import Logger
logger = Logger(name='Tools')

def retry_if_cuda_oom(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                return func(*args, **kwargs)
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    if attempt < max_attempts - 1:
                        logger.warn_info(f"CUDA OOM: Retrying... (Attempt {attempt + 1}/{max_attempts})")
                        time.sleep(1) 
                        torch.cuda.empty_cache()
                        gc.collect()
                    continue
                else:
                    logger.warn_info("Reached maximum retry attempts for CUDA OOM.")
                    raise
    return wrapper

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def save_checkpoint(state, path, filename="latest.pth"):
    torch.save(state, os.path.join(path, filename))

def save_tensor_img(tenor_im, path):
    im = tenor_im.cpu().clone()
    im = im.squeeze(0)
    tensor2pil = transforms.ToPILImage()
    im = tensor2pil(im)
    im.save(path)
import matplotlib.cm as cm
def save_feat_img(feat, path):
    im = feat.cpu()
    channel_data = im[0, 0, :, :].detach().numpy()
    print(type(channel_data))
    normalized_data = cv2.normalize(channel_data, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    magma_color_mapped = cm.get_cmap('magma')(normalized_data)[:, :, :3]  # 仅取RGB，丢弃透明度
    heatmap_color = np.uint8(magma_color_mapped * 255)
    cv2.imwrite(path, heatmap_color)

def path_to_image(path: str, size=(1024, 1024), color_type=['rgb', 'gray'][0]):
    if color_type.lower() == 'rgb':
        image = cv2.imread(path)
    elif color_type.lower() == 'gray':
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        print('Select the color_type to return, either to RGB or gray image.')
        return
    if size:
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    if color_type.lower() == 'rgb':
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert('RGB')
    else:
        image = Image.fromarray(image).convert('L')
    return image

def check_state_dict(state_dict: dict, unwanted_prefix='_orig_mod.'):
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    return state_dict

def generate_smoothed_gt(gts):
    epsilon = 0.001
    new_gts = (1-epsilon)*gts+epsilon/2
    return new_gts
