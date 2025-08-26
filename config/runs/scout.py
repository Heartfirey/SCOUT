import os
# training settings
ckpt_dir = "./works/scout"
active_learning =  False
tot_epochs = 30
sup_only_train_epoch=15
distributed_train = True
device_map = {
    "model": "*"
} # Only available for non distributed training
rand_seed = 7
lr = 1e-4

IoU_finetune_last_epochs = [0, -6][1]
eval_epoch = 10
# model settings
compile_model = False
precisionHigh = True
img_size = 640
clip_img_size = 336

backbone = [
    'vgg16', 'vgg16bn', 'resnet50',         # 0, 1, 2
    'pvt_v2_b2', 'pvt_v2_b5',               # 3-bs10, 4-bs5
    'swin_v1_b', 'swin_v1_l',               # 5-bs9, 6-bs6
    'swin_v1_t', 'swin_v1_s',               # 7, 8
    'pvt_v2_b0', 'pvt_v2_b1',               # 9, 10
][5]
lateral_channels_in_collection = {
    'vgg16': [512, 256, 128, 64], 'vgg16bn': [512, 256, 128, 64], 'resnet50': [1024, 512, 256, 64],
    'pvt_v2_b2': [512, 320, 128, 64], 'pvt_v2_b5': [512, 320, 128, 64],
    'swin_v1_b': [1024, 512, 256, 128], 'swin_v1_l': [1536, 768, 384, 192],
    'swin_v1_t': [768, 384, 192, 96], 'swin_v1_s': [768, 384, 192, 96],
    'pvt_v2_b0': [256, 160, 64, 32], 'pvt_v2_b1': [512, 320, 128, 64],
}[backbone]
lateral_channels_in_collection = [channel * 2 for channel in lateral_channels_in_collection]
cxt_num = [0, 3][1]
cxt = lateral_channels_in_collection[1:][::-1][-cxt_num:] if cxt_num else []

# data settings
load_all = False
batch_size = 6
data_split = [0.05] #[0.01, 0.05, 0.1]
data_split_indices_file_format = "data/cache/labeled_indices/split{}_labeled_indices.pt"
task = ["COD", "RefCOD"][1]
training_set = {
    'COD': 'TR-COD10K+TR-CAMO',
    'RefCOD': 'TR-COD10K+TR-CAMO'
}[task]
enable_ref_text = (task == "RefCOD")
training_set = "TR-COD10K+TR-CAMO"
testing_sets = "CHAMELEON+"

# Reference COD Settings
text_replace = "camouflaged object; concealed object; object hidden in background; CLUE_Token"
use_text_replace = True
text_encoders = ['roberta-base', 'bert-base', 'bert-large', 'clip-base', 'clip-large'][-1]
text_encoders_out_dim = {
    'roberta-base': 768, 
    'bert-base': 768, 
    'bert-large': 1024, 
    'clip-base': 512,
    'clip-large': 768
}[text_encoders]
text_embed_dim = 64
img_channel_dim = 64
ref_max_len = 64
fusion_head_dim = 64
fusion_num_heads = 8
fusion_attn_dropout = 0.1

# Evaluate settings
pred_save_root = os.path.join(ckpt_dir, "training_preds")
eval_epoch = 90
eval_step = 2
save_step = 10

#img_aug settings
sampling_freq=5
inner_iter = 2
g_scale = 0.4
c_scale = 0.3
c_reg_coef = 10
n_dim = 128

#text_aug setting
enable_text_aug=False
using_ref_cache=True
VQ_ckp='data/cache/labeled_indices/codebook_{}.pt'


#wandb
ModelName="SCOUT"
others={
    'sup_epoch':sup_only_train_epoch,
    'total_epoch':tot_epochs
}