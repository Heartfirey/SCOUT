import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.init import xavier_uniform_
from einops import rearrange
from flash_attn.bert_padding import unpad_input, pad_input
from typing import List, Dict
try:
    from .flash_attn import FlashCrossAttention
except ImportError:
    from flash_attn import FlashCrossAttention

class MultiModalFusion(nn.Module):
    def __init__(self, img_dim: int, text_dim: int, config) -> None:
        super(MultiModalFusion, self).__init__()
        
        self.config = config
        
        self.img_norm = nn.LayerNorm(img_dim)
        self.text_norm = nn.LayerNorm(text_dim)
        
        self.cross_modal_attention = MultiModalAttention(
            query_dim=img_dim,
            key_dim=text_dim,
            value_dim=text_dim,
            head_dim=config.fusion_head_dim,
            num_heads=config.fusion_num_heads,
            dropout=config.fusion_attn_dropout
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(img_dim, img_dim * 3, bias=True)
        )
        
        self._init_parameters()
        
    def _init_parameters(self):
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
    
    def forward(self, img_x_lst: List[Tensor], text_x: Dict[str, Tensor]) -> any:
        """Input
        Args:
            img_x_lst (List[Tensor]): list of image features, each shape like [B, C, H, W]
            text_x (Dict[str, Tensor]): text feature list.
        Returns:
            ...
        """
        new_img_x_lst = []
        for img_x in img_x_lst:
            B, C, H, W = img_x.size()
            img_x = img_x.flatten(-2).permute(0, 2, 1)  # (B, C, H, W) -> (B, H*W, C)
            mask = (~text_x['masks']).to(torch.long)
            attn_w = self.cross_modal_attention(self.img_norm(img_x), self.text_norm(text_x['refs']), self.text_norm(text_x['refs']), attention_mask=mask)
            ref_embeds = text_x['ref_embeds']
            shift, scale, gate = self.adaLN_modulation(ref_embeds).chunk(3, dim=1)
            attn_w = gate.unsqueeze(1) * (attn_w * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1))
            # img_x = img_x + attn_w
            img_x = attn_w + img_x
            img_x = img_x.permute(0, 2, 1).reshape(B, C, H, W)
            new_img_x_lst.append(img_x)
        
        return new_img_x_lst

class MultiModalAttention(nn.Module):
    def __init__(self, query_dim: int, key_dim: int, head_dim: int, num_heads: int, value_dim: int=None, dropout: float=0.1) -> None:
        super(MultiModalAttention, self).__init__()
        
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.embed_dim = head_dim * num_heads
        
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim if value_dim is not None else key_dim
        
        self.query_proj = nn.Linear(self.query_dim, self.embed_dim)
        self.key_proj = nn.Linear(self.key_dim, self.embed_dim)
        self.value_proj = nn.Linear(self.value_dim, self.embed_dim)

        self.mh_crossattn = FlashCrossAttention(attention_dropout=dropout)
        # self.mh_crossattn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        self.out_proj = nn.Linear(self.embed_dim, query_dim)
        
        self._init_parameters()
        
    def _init_parameters(self):
        xavier_uniform_(self.query_proj.weight); self.query_proj.bias.data.zero_()
        xavier_uniform_(self.key_proj.weight); self.key_proj.bias.data.zero_()
        xavier_uniform_(self.value_proj.weight); self.value_proj.bias.data.zero_()
        xavier_uniform_(self.out_proj.weight); self.out_proj.bias.data.zero_()
        
    def forward(self, q, k, v=None, attention_mask=None):
        B, QLEN, QDIM = q.shape
        B, KLEN, KDIM = k.shape
        
        q_proj = self.query_proj(q) # (B, QLEN, QDIM) -> (B, QLEN, embed_dim)
        k_proj = self.key_proj(k)   # (B, KLEN, KDIM) -> (B, KLEN, embed_dim)
        v_proj = self.value_proj(v if v is not None else k)
        
        if 1:
            # flash_attn_varlen_kvpacked_func
            # q: (total_q, nheads, headdim)
            # kv: (total_k, 2, nheads_k, headdim)
            # cu_seqlens_q: (batch_size + 1,), dtype torch.int32. 
            # cu_seqlens_k: (batch_size + 1,), dtype torch.int32.
            # max_seqlen_q: int. 
            # max_seqlen_k: int
            origin_dtype = q_proj.dtype
            # -> (B, LEN, num_heads, head_dim)
            q_proj = AttnTensorTools.to_multi_head_shape(q_proj, B, QLEN, self.num_heads, self.head_dim).to(torch.bfloat16)
            k_proj = AttnTensorTools.to_multi_head_shape(k_proj, B, KLEN, self.num_heads, self.head_dim).to(torch.bfloat16)
            v_proj = AttnTensorTools.to_multi_head_shape(v_proj, B, KLEN, self.num_heads, self.head_dim).to(torch.bfloat16)
            # Img_query process
            max_seqlen_q = q_proj.shape[1]
            cu_seqlens_q = torch.arange(0, (B + 1) * max_seqlen_q, step=max_seqlen_q, dtype=torch.int32, device=q_proj.device)
            q_proj = q_proj.flatten(0, 1)   # (B * QLEN, n_heads, head_dim)
            # Text key-value process
            kv_proj = torch.stack([k_proj, v_proj], dim=2)
            kv_proj = kv_proj.flatten(2)
            kv_unpad, indices, cu_seqlens_k, max_seqlen_k, batch_size = unpad_input(kv_proj, attention_mask)
            kv_proj = rearrange(kv_unpad, 'nnz (two h d) -> nnz two h d', two=2, h=self.num_heads)  # [total_k, 2, nheads, head_dim]

            out = self.mh_crossattn(q_proj, kv_proj, cu_seqlens=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen=max_seqlen_q, max_seqlen_k=max_seqlen_k)
            
            out = out.reshape(B, QLEN, self.num_heads, self.head_dim).flatten(2)
            out = out.to(origin_dtype)

            out = self.out_proj(out)
        
        # out = self.mh_crossattn(
        #     query=q_proj, key=k_proj, value=v_proj, key_padding_mask=attention_mask.bool()
        # )
        # out = out[0].reshape(B, QLEN, self.num_heads, self.head_dim).flatten(2)
        # out = self.out_proj(out)
        
        return out
        
class AttnTensorTools:
    @classmethod
    def to_multi_head_shape(cls, tensor: Tensor, bs: int, seq_len: int, num_heads: int, head_dim: int) -> Tensor:
        return tensor.view(bs, seq_len, num_heads, head_dim).contiguous()


# if __name__ == "__main__":
#     # Test MultiModalAttention
#     ref_text = "Just a sentence for test!"
#     from transformers import CLIPTextModel, CLIPTokenizerFast
#     # tokenizer = CLIPTokenizerFast("openai/clip-vit-base-patch32")
#     text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
#     tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
#     for p in text_encoder.parameters():
#         p.requires_grad_(False)
#     tokenized = tokenizer(ref_text, padding="max_length", max_length=64, truncation=True, return_tensors="pt")
#     encoded_text = text_encoder(**tokenized)
#     text_attention_mask = tokenized['attention_mask'].ne(1).bool()

#     text_feat_dict = {
#         'text_features': encoded_text.last_hidden_state,
#         'text_attention_mask': text_attention_mask,
#     }

#     from textproc import TextProcesser
#     text_proc = TextProcesser(512, 512)
#     text_feat_dict = text_proc(text_feat_dict)
    
#     # print(text_feat_dict)
#     # print('*' * 20)

#     mm_attention = MultiModalAttention(512, 512, 64, 8, 512).cuda()
#     img_x = torch.randn(1, 512, 22, 22)
#     img_x = img_x.flatten(-2).permute(0, 2, 1)
#     print("Image reshape: ", img_x.shape)
#     mask = (~text_feat_dict['masks']).to(torch.long)
#     result = mm_attention(img_x.cuda(), text_feat_dict['refs'].cuda(), text_feat_dict['refs'].cuda(), mask.cuda())
#     print(result.cpu().shape)
