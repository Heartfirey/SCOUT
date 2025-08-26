import os
import torch
import datetime

from config import Config
from utils import Logger, set_seed
from data import init_testloaders
from models import build_model_eval
from engine.evaluator import Evaluator

from scripts.args import get_eval_parser

parser = get_eval_parser()
args = parser.parse_args()

# merge the config from file
config = Config(run_cfg=args.config)
config.distributed_train=False
logger = Logger(name='TalNet Eval Script', path=os.path.join(config.ckpt_dir, "eval_log.txt"))

if config.rand_seed is not None:
    set_seed(config.rand_seed)
    
device = torch.device('cuda')

# prepare models
model = build_model_eval(config=config, logger=logger, resume=args.model_dir, device=device)
model.eval()
logger.key_info("[o] Model loaded done. Current eval plan: {}".format(config.task))

config.use_text_replace = (args.text_type=='fixed')

# replace the evaluation dataset settings.
config.testing_sets = args.testset
config.pred_save_root = os.path.join(config.ckpt_dir, "test_preds")
config.using_ref_cache = False
# prepare data loaders
testloaders = init_testloaders(config)

# preapre evaluator
evaluator = Evaluator.from_exists(
    config=config,
    logger=logger,
    device=device,
    model=model
)

def evaluate_pipeline():
    for testset_name, testloader in testloaders.items():
        evaluator.inference_on_dataset(testloader, testset_name)
        evaluator.evaluate_inference_result(testloader, testset_name)
    logger.success_info("[o] Evaluation done at {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

if __name__ == "__main__":
    evaluate_pipeline()
