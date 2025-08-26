# refcod settings
enable_ref_text = True
text_encoders = ['roberta-base', 'bert-base', 'bert-large', 'clip-base', 'clip-large'][-1]
text_encoders_out_dim = {
    'roberta-base': 768, 
    'bert-base': 768, 
    'bert-large': 1024, 
    'clip-base': 512,
    'clip-large': 768
}[text_encoders]
text_embed_dim = 256
img_channel_dim = 256
ref_max_len = 64
fusion_head_dim = 256
fusion_num_heads = 8
fusion_attn_dropout = 0.1
text_replace=None
using_ref_cache=False

#text_aug
vq_attn_dropout = 0.1
TF_block_num=1
codebook_num=64
codebook_dim=text_encoders_out_dim
VQ_ckp=None
VQ_num_heads=8
ccm_alpha=1
enable_text_aug=False


enable_g_aug=True
enable_c_aug=True
enable_img_aug=enable_g_aug or enable_c_aug

enable_fixtext=False