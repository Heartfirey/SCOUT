import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.VQ import Vector_Quantizer
from models.modules.mlp import Block
from models.modules.CCM import CCM

class Text_Aug(nn.Module):
    def __init__(self, config, text_dim, img_dim, dropout, codebook_num, codebook_dim, TF_block_num=1):
        super(Text_Aug, self).__init__()

        self.config = config
        self.text_dim = text_dim
        self.img_dim = img_dim
        self.VQ = Vector_Quantizer(
            codebook_num=codebook_num,
            codebook_dim=codebook_dim,
            e_dim=text_dim
        )

        self.ccm = CCM(
            img_dim=img_dim,
            text_dim=text_dim,
            config=config
        )
    
    def forward(self, text_x, img_features, hash_labels=None, gt=None):
        text_features, text_attention_mask = text_x['text_features'], text_x['text_attention_mask']
        B, C, H, W = img_features.shape
        text_shape = text_features.shape
        text_mask_shape = text_attention_mask.shape
        text_features = text_features.reshape(B, -1, self.text_dim)
        text_attention_mask = text_attention_mask.reshape(B, -1)
        
        c, ccm_loss = self.ccm(img_features, gt)
        text_features, text_attention_mask, vq_loss = self.VQ(c, text_features.detach(), text_attention_mask.detach(), hash_labels=hash_labels, update=(gt!=None))
        text_features = text_features.reshape(text_shape)
        text_attention_mask = text_attention_mask.reshape(text_mask_shape)
        return{
            'text_features':text_features,
            'text_attention_mask':text_attention_mask
        }, {
            'ccm_loss':ccm_loss,
            'vq_loss':vq_loss
        }