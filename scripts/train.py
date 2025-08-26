import os
import torch
import datetime
import wandb

from torch.distributed import init_process_group, destroy_process_group, get_rank


from config import Config
from data import init_trainloader, init_testloaders
from engine import ActiveTrainer, SemiSupervisedTrainer
from utils import Logger, set_seed
from utils.submit import create_wandb
from models import build_model_optimizers

from scripts.args import get_train_parser

torch.autograd.set_detect_anomaly(True)
parser = get_train_parser()
args = parser.parse_args()

config = Config(run_cfg = args.config)
os.makedirs(config.ckpt_dir, exist_ok=True)
logger = Logger(name='TalNet Train Script', path=os.path.join(config.ckpt_dir, "log.txt"))

    
if config.distributed_train:
    init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=3600*10))
    device = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(device)
else:
    __config_device = config.device_map.get('model', '*')
    if __config_device == '*':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(__config_device)

if config.rand_seed is not None:
    seed = config.rand_seed
    if config.distributed_train:
        seed += get_rank()
        print(f'Rank {get_rank()} seed: {seed}')
    set_seed(seed)
writer=None
if (not config.distributed_train) or (config.distributed_train and get_rank() == 0):
    writer = create_wandb(config)
    

epoch_st = 1

trainloader = init_trainloader(config)
testloaders = init_testloaders(config)

def active_training_pipeline():
    __config_device = config.device_map.get('active_components', '*')
    if __config_device == '*':
        active_components_device = device
    else:
        active_components_device = torch.device()
    model_lrsch = build_model_optimizers(config, logger, device, resume=args.resume)
    trainer = ActiveTrainer(
        data_loaders=(trainloader, testloaders),
        config=config, device=device, logger=logger, writer=writer
    )
    for split in config.data_split:
        labeled_indices_file = config.data_split_indices_file_format.format(split)
        if os.path.isfile(labeled_indices_file):
            labeled_indices = torch.load(labeled_indices_file)
        else:
            raise FileNotFoundError("Labeled indices file not found at '{}'".format(labeled_indices_file))
        trainer.reset_trainer(
            model_lrsch=model_lrsch,
            labeled_indices=labeled_indices
        )
        trainer.launch_select(
            split=split, 
            target_split=next((config.active_sample_split[i + 1] 
                               for i in range(len(config.active_sample_split) - 1) 
                               if config.active_sample_split[i] == split), None),
            total_epochs=config.active_aug_epoch
        )
    
def semi_supervised_training_pipeline():
    if config.enable_text_aug:
        config.ckpt_dir = os.path.join(config.ckpt_dir, 'text_aug')
        os.makedirs(config.ckpt_dir, exist_ok=True)
    trainer = SemiSupervisedTrainer(
        data_loaders=(trainloader, testloaders),
        config=config, device=device, logger=logger, writer=writer
    )
    for split in config.data_split:
        model_lrsch = build_model_optimizers(config, logger, device)
        labeled_indices_file = config.data_split_indices_file_format.format(split)
        
        if os.path.isfile(labeled_indices_file):
            labeled_indices = torch.load(labeled_indices_file)
        else:
            raise FileNotFoundError("Labeled indices file not found at '{}'".format(labeled_indices_file))
        trainer.reset_trainer(
            split=split,
            model_lrsch=model_lrsch,
            labeled_indices=labeled_indices
        )
        trainer.launch_train(
            split=split, 
            total_epochs=config.tot_epochs, 
        )

def main():
    if args.active_learning:
        config.active_learning = True
        active_training_pipeline()
    else:
        semi_supervised_training_pipeline()
    if config.distributed_train:
        destroy_process_group()
    logger.freeze_info("Training script execution completed at {}".format(
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    
if __name__ == '__main__':
    main()
    


