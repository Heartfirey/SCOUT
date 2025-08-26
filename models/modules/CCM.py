import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from models.modules.mlp import Block
from utils.submit import vis_attn 
from torch.distributed import get_rank

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = preds.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        intersection = (preds * targets).sum()
        dice_coefficient = (2.0 * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)

        return 1.0 - dice_coefficient

class CCM(nn.Module):
    def __init__(self, img_dim: int, text_dim: int, config):
        super(CCM, self).__init__()
        self.config = config
        self.proj = nn.Linear(img_dim, text_dim)
        self.diceloss = nn.BCELoss()
        self.mseloss = nn.MSELoss()
        self.block = Block(
            dim=text_dim,
            num_heads=1,
            attn_drop=config.fusion_attn_dropout,
            use_learnable_query=True,
            return_attn_map=True
        )
    
    def forward(self, x, gt=None):
        B, C, H, W = x.shape
        x_flatten = x.reshape(B,C,-1).permute(0,2,1)
        x_flatten = self.proj(x_flatten)
        outs, attn = self.block(x_flatten, H, W)
        loss = 0
        if gt!=None:
            gt = (F.interpolate(gt, size=(H, W), mode='bilinear', align_corners=True)>0.5).float()
            attn = attn.reshape(B, 1, H, W)
            loss = self.diceloss(attn.sigmoid(), gt) + self.config.ccm_alpha * self.mseloss(outs.squeeze(1), self.avg_pooling(x_flatten.permute(0,2,1).reshape(B,-1,H,W), gt))
            # if not self.config.distributed_train or get_rank() == 0:
            #     vis_attn(attn, gt)
        return outs, loss

    def avg_pooling(self, preds, mask):
        mask = (mask > 0).float()
        masked_preds = preds * mask

        masked_sum = masked_preds.sum(dim=(2, 3))
        mask_area = mask.sum(dim=(2, 3))
        avg_pooling = masked_sum / (mask_area + 1e-6)

        return avg_pooling
