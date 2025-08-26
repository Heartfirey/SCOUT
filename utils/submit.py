import wandb
import os
import torch

writer = None
cnt=0
now_step=0
update=True
def create_wandb(config):
    global writer
    wandb.init(
        project="SCOUT",
        name=os.path.basename(config.ckpt_dir),
        config={
            "ModelName": config.ModelName,
            "DataSplit": config.data_split[0],
        }|config.others
    )
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=config.ckpt_dir)
    wandb.tensorboard.patch(root_logdir=config.ckpt_dir)
    return writer

def get_writer():
    global writer
    return writer

def vis_attn(attn_map, gt):
    global update, cnt, now_step, writer
    if not update:
        if now_step != wandb.run.step:
            update=True
            cnt=0
        else:
            return
    img = torch.cat((gt, attn_map), dim=-1)
    writer.add_image(str(cnt), img[0], global_step=wandb.run.step)
    cnt += 1
    if cnt == 6:
        update=False
        now_step=wandb.run.step