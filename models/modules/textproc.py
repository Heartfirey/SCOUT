import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, List, Dict

class TextProcessor(nn.Module):
    """
    Module for processing referring text data
    """
    def __init__(self, in_dim: int, out_dim: int):
        super(TextProcessor, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim, eps=1e-12)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, in_text_dict: Dict[str, Tensor]) -> dict:
        # Extract from the input dict
        # encoded_text = in_text_dict['encoded_text']
        text_features = in_text_dict['text_features'].squeeze(1)
        text_attention_mask = in_text_dict['text_attention_mask'].squeeze(1)
        
        # Perform projection on text features
        text_features = self.proj(text_features)
        text_features = self.layer_norm(text_features)
        text_features = self.dropout(text_features)
        text_masks = text_attention_mask
        text_features = NestedTensor(text_features, text_masks)
        
        # Post-process the text features
        text_word_features, text_word_masks = text_features.decompose()
        # except embedded shape: [B, LEN, C]
        embedded = text_word_features * text_word_masks.unsqueeze(-1)
        # except text_sentence_features shape: [B, C]
        text_sentence_features = embedded.sum(1) / (text_word_masks.ne(1).sum(-1).unsqueeze(-1).float())

        feature_dict = {
            "refs": text_word_features,             # [B, SEQ_LEN, C]
            # "ref_values": text_word_features,     # [B, SEQ_LEN, C]
            "masks": text_word_masks,               # [B, SEQ_LEN]
            "ref_embeds": text_sentence_features,   # [B, C]
        }
        return feature_dict

class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device, non_blocking=False):
        cast_tensor = self.tensors.to(device, non_blocking=non_blocking)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device, non_blocking=non_blocking)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def record_stream(self, *args, **kwargs):
        self.tensors.record_stream(*args, **kwargs)
        if self.mask is not None:
            self.mask.record_stream(*args, **kwargs)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)
