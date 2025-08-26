import os
import cv2
import argparse
import torch
import torch.nn as nn
import torch.utils.data as data
import prettytable as pt

from glob import glob
from tqdm import tqdm

from config import Config
from utils import Logger, save_tensor_img
from data import init_testloaders
from models import build_model_eval

from .metrics import calculate


class Evaluator:
    def __init__(self, config: Config, logger: Logger, resume: str, device: torch.device='cpu', model: nn.Module=None):
        self.config = config
        self.logger = logger
        self.device = device
        self.model = build_model_eval(config, logger, resume, device) if model is None else model
        self.logger.success_info("[o] Evaluator is ready!")

    @classmethod
    def from_exists(cls, config: Config, logger: Logger, resume: str=None, device: torch.device='cpu', model: nn.Module=None):
        logger.freeze_info("[+] Evaluator will be initialize with existing model...")
        return cls(config, logger, resume, device, model)
    
    def inference_on_dataset(self, dataloader: data.DataLoader, testset_name: str, ema=False) -> None:
        model_training_state = self.model.training
        # if model_training_state:
        #     self.model.eval()
        if self.config.use_text_replace:
            text_type = 'const'
        else:
            text_type = 'accurate'
        current_save_dir = os.path.join(self.config.pred_save_root, self.config.task, testset_name, text_type)
        os.makedirs(current_save_dir, exist_ok=True)
        
        for batch in tqdm(dataloader, total=len(dataloader)):
            inputs = batch[0].to(self.device)
            label_paths = batch[-2]
            
            if self.config.enable_ref_text:
                text_features, text_attention_mask = batch[2].to(self.device), batch[3].to(self.device)
                ref_text_dict = {
                    'text_features': text_features,
                    'text_attention_mask': text_attention_mask
                }
                with torch.no_grad():
                    scaled_preds = self.model(inputs, ref_text_dict, ema=ema)[-1].sigmoid()
            else:
                with torch.no_grad():
                    scaled_preds = self.model(inputs, ema=ema)[-1].sigmoid()

            for idx_sample in range(scaled_preds.shape[0]):
                res = nn.functional.interpolate(
                    scaled_preds[idx_sample].unsqueeze(0),
                    size=cv2.imread(label_paths[idx_sample], cv2.IMREAD_GRAYSCALE).shape[:2],
                    mode='bilinear',
                    align_corners=True
                )
                save_imgfile_name = label_paths[idx_sample].replace('\\', '/').split('/')[-1]
                save_tensor_img(res, os.path.join(current_save_dir, save_imgfile_name))
                
    def evaluate_inference_result(self, dataloader: data.DataLoader, testset_name: str, save_dir_replace: str=None) -> dict:
        log_save_file_id_suffix = 0
        log_savedir = os.path.join(self.config.pred_save_root, self.config.task, 'results', '{}_{}.log'.format(testset_name, log_save_file_id_suffix))
        log_savedir = log_savedir if save_dir_replace is None else save_dir_replace
        if not os.path.exists(os.path.join(self.config.pred_save_root, self.config.task, 'results')):
            os.makedirs(os.path.join(self.config.pred_save_root, self.config.task, 'results'), exist_ok=True)
        while os.path.exists(log_savedir):
            log_save_file_id_suffix += 1
            log_savedir = os.path.join(self.config.pred_save_root, self.config.task, 'results', '{}_{}.log'.format(testset_name, log_save_file_id_suffix))
        
        if self.config.use_text_replace:
            text_type = 'const'
        else:
            text_type = 'accurate'

        current_result_dir = os.path.join(self.config.pred_save_root, self.config.task, testset_name, text_type)
        gt_path = os.path.join(self.config.data_root_dir, self.config.task, testset_name)
        assert os.path.isdir(current_result_dir), f"[x] {current_result_dir} not exists!"
        
        gt_paths = sorted([
            os.path.join(gt_path, 'gt', file_name)
            for file_name in os.listdir(os.path.join(gt_path, 'gt'))
        ])
        
        pred_paths = sorted([
            os.path.join(current_result_dir, file_name)
            for file_name in os.listdir(current_result_dir)
        ])
        
        with open(log_savedir, 'a+') as fw:
            tb = pt.PrettyTable()
            tb.field_names = [
                "Dataset", "Task", "maxFm", "wFmeasure", 'MAE', "Smeasure", "meanEm", "maxEm", "meanFm",
                "adpEm", "adpFm",
            ]
            self.logger.key_info("[+] Starting calculate the evaluation result...")
            em, sm, fm, mae, wfm = calculate(gt_paths, pred_paths, metrics=['S', 'MAE', 'E', 'F', 'WF'], verbose=True)
            
            e_max, e_mean, e_adp = em['curve'].max(), em['curve'].mean(), em['adp'].mean()
            f_max, f_mean, f_wfm, f_adp = fm['curve'].max(), fm['curve'].mean(), wfm, fm['adp']
            
            tb.add_row([
                testset_name, self.config.task,
                f_max.round(3), f_wfm.round(3), mae.round(3), sm.round(3), e_mean.round(3), e_max.round(3), f_mean.round(3),
                e_adp.round(3), f_adp.round(3)
            ])
            
            self.logger.success_info('\n'+str(tb)+'\n')
            fw.write(str(tb).replace('+', '|') + '\n')
            fw.close()
        
        return dict(e_max=e_max, e_mean=e_mean, e_adp=e_adp, f_max=f_max, f_mean=f_mean, f_wfm=f_wfm, f_adp=f_adp, mae=mae, sm=sm)
    
            
