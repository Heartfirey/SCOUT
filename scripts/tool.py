import os
import sys
import torch
import datetime

from torch.distributed import init_process_group, destroy_process_group, get_rank

from tqdm import tqdm
from config import Config
from data import init_trainloader, init_testloaders
from engine import ActiveTrainer, SemiSupervisedTrainer
from utils import Logger, set_seed
from models import build_model_optimizers

from scripts.args import get_tool_parser

# set the root path to the project
os.chdir(os.environ['PYTHONPATH'])

parser = get_tool_parser()
args = parser.parse_args()

config = Config(run_cfg = args.config)
config.distributed_train = False
logger = Logger(name='TalNet Tool Script', path="/tmp/tool.log")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def extract_pseudo_labels(split, model_dir, save_dir):
    cache_body = {
        "split": split,
        "soft_labels": dict()
    }
    model = build_model_optimizers(
        config=config,
        logger=logger,
        device=device,
        resume=model_dir
    )[0]
    config.batch_size = 1
    train_loader = init_trainloader(config)
    model.eval()
    for batch in tqdm(train_loader):
        inputs = batch[0].to(device)
        label_file_path = batch[-2][0]
        if type(label_file_path) is not str:
            label_file_path = label_file_path.item()
        if config.enable_ref_text:
            text_features, text_attention_mask = batch[2].to(device), batch[3].to(device)
            ref_text_dict = {
                'text_features': text_features,
                'text_attention_mask': text_attention_mask
            }
            with torch.no_grad():
                scaled_preds = model(inputs, ref_text_dict)
                for each in scaled_preds:
                    each.to('cpu')
        else:
            with torch.no_grad():
                scaled_preds = model(inputs)[-1].sigmoid().cpu()
                # for each in scaled_preds:
                #     each.to('cpu')
        cache_body['soft_labels'].update({label_file_path:scaled_preds})
        del(inputs)
        torch.cuda.empty_cache()
    logger.freeze_info("[+] Saving to dir {}...".format(save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(
        cache_body, 
        os.path.join(save_dir, "split{}_pseudo_labels.pt".format(split))
    )
    logger.success_info("[o] Done at {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    
def sugery_of_exists(pt_dir: str):
    old_pt = torch.load(pt_dir)
    new_pt = {
        "split": old_pt['split'],
        "soft_labels": dict()
    }
    for key, value in tqdm(old_pt['soft_labels'].items()):
        new_pt['soft_labels'].update({str(key.split('/')[-1].split('.')[0]):value})
    torch.save(new_pt, pt_dir)

def main():
    extract_pseudo_labels(args.split, args.model_dir, args.save_dir)
    # sugery_of_exists('works/semi_split0.01_cod_dist/split0.01_pseudo_labels.pt')
    # sugery_of_exists('works/semi_split0.05_cod_dist/split0.05_pseudo_labels.pt')

if __name__ == "__main__":
    main()
