import os
import datetime

import torch
import torch.distributed
import torch.nn as nn
from torch.distributed import get_rank
import numpy as np
from PIL import Image
import random
import wandb
from torchvision.transforms import functional as TF
from torchvision import transforms

from data import prepare_dataloader
from utils import AverageMeter, retry_if_cuda_oom
from utils.maker import make_codebook
from .loss import PixLoss, AugLoss
from .evaluator import Evaluator
from .calculator import ActiveCalculator

class ActiveTrainer:
    def __init__(self, data_loaders, config, device, logger=None, writer=None):
        self.train_loader, self.test_loaders = data_loaders

        self.config = config
        self.aug_loss = AugLoss(config)
        self.loss_log = AverageMeter()

        self.global_step = 0
        self.device = device
        self.logger = logger
        self.writer = writer

    def random_bool(self, probability=0.5):
        return random.random() < probability

    def launch_select(self, split: int, target_split: int, total_epochs: int):
        target_labeled_indices = list(self.current_labeled_indices)
        target_count = int(len(self.train_loader.dataset) * target_split) - len(target_labeled_indices)
        score_list = []
        self.labeled_dataloader = prepare_dataloader(
            dataset = self.train_loader.dataset, batch_size=self.config.batch_size, num_workers=self.config.num_workers,
            to_be_distributed=self.config.distributed_train, is_train=True, labeled_indices=self.current_labeled_indices
        )
    
        self.unlabeled_dataloader = prepare_dataloader(
            dataset = self.train_loader.dataset, batch_size=self.config.batch_size, num_workers=self.config.num_workers,
            to_be_distributed=self.config.distributed_train, is_train=True, labeled_indices=self.current_labeled_indices, is_unsup=True
        )
        self.model.eval()
        self.aug.eval()
        if not self.config.distributed_train or get_rank()==0:
            self.active_calculator = ActiveCalculator(
                config=self.config,
                logger=self.logger,
                device=self.device,
                aug=self.aug if not self.config.distributed_train else self.aug.module,
                model=self.model if not self.config.distributed_train else self.model.module,
                dataloaders=(self.labeled_dataloader,self.unlabeled_dataloader)
            )
            score_list = self.active_calculator.cal_active_score()
            target_labeled_indices = list(self.current_labeled_indices)
            mid = int(len(score_list) * 0.5)
            start = max(0, mid - (target_count // 2))
            end = start + target_count
            target_labeled_indices.extend([item[0] for item in score_list[start:end]])
            code_book = make_codebook(target_labeled_indices)
            indices_file = self.config.data_split_indices_file_format.format(target_split)
            torch.save(target_labeled_indices, indices_file)
            self.logger.freeze_info("Saving split{} indices:{}".format(target_split, indices_file))
            code_book_file= self.config.VQ_ckp.format(target_split)
            torch.save(code_book, code_book_file)
            self.logger.freeze_info("Saving split{} codebook:{}".format(target_split, code_book_file))
        if self.config.distributed_train:
            torch.distributed.barrier()

    def _destory_model(self):
        try:
            del self.model, self.model_optimizer, self.model_lr_scheduler
        except:
            pass
        
    def reset_trainer(self, model_lrsch, labeled_indices):
        self._destory_model()
        torch.cuda.empty_cache()
        self.model, self.aug, self.model_optimizer, self.model_optimizer_aug, self.model_lr_scheduler, self.epoch_st = model_lrsch
        self.current_labeled_indices = labeled_indices

def TorchColorEnhance(img_tensor: torch.Tensor, factor: float) -> torch.Tensor:
    
    if img_tensor.shape[0] != 3:
        raise ValueError("Input image tensor must have 3 channels (RGB).")

    img_tensor_normalized = img_tensor / 255.0
    
    gray_tensor = (
        0.2989 * img_tensor_normalized[0] + # R
        0.5870 * img_tensor_normalized[1] + # G
        0.1140 * img_tensor_normalized[2]   # B
    ).unsqueeze(0)
    
    gray_tensor = gray_tensor.repeat(3, 1, 1)
    enhanced_tensor = factor * img_tensor_normalized + (1 - factor) * gray_tensor
    enhanced_tensor = (enhanced_tensor * 255.0).clamp(0, 255)
    
    return enhanced_tensor
    

class SemiSupervisedTrainer:
    def __init__(self, data_loaders, config, device, logger=None, writer=None):
        self.train_loader, self.test_loaders = data_loaders

        self.config = config
        self.writer = writer
        self.cnt=0
        if self.config.out_ref:
            self.criterion_gdt = nn.BCELoss()

        # Setting loss
        self.pix_loss = PixLoss(config)
        self.aug_loss = AugLoss(config)
        self.loss_log = AverageMeter()

        self.global_step = 0
        self.device = device
        self.logger = logger
        
    def _destory_model(self):
        try:
            del self.model, self.model_optimizer, self.model_lr_scheduler
        except:
            pass
        
    def reset_trainer(self, split, model_lrsch, labeled_indices):
        self._destory_model()
        torch.cuda.empty_cache()
        self.model, self.aug, self.model_optimizer, self.model_optimizer_aug, self.model_lr_scheduler, self.epoch_st = model_lrsch
        self.current_labeled_indices = labeled_indices
        if self.config.distributed_train:
            self.model.module.load_codebook(self.config.VQ_ckp.format(split))
        else:
            self.model.load_codebook(self.config.VQ_ckp.format(split))
    
    @retry_if_cuda_oom
    def _train_batch(self, batch, gt_replace=None, loss_alpha=1.):
        inputs = batch[0].to(self.device)
        gts = batch[1].to(self.device) if gt_replace is None else gt_replace

        if self.config.enable_ref_text:
            text_features, text_attention_mask = batch[2].to(self.device), batch[3].to(self.device)
            ref_text_dict = {
                'text_features': text_features,
                'text_attention_mask': text_attention_mask
            }
            scaled_preds = self.model(inputs, ref_text_dict, batch[-2], gts if gt_replace==None else None)
        else:
            scaled_preds = self.model(inputs)

        loss = 0
        if gt_replace == None and self.config.enable_text_aug:
            scaled_preds, text_loss = scaled_preds
            for v in text_loss.values():
                loss = v + loss
        if self.config.out_ref:
            (outs_gdt_pred, outs_gdt_label), scaled_preds = scaled_preds
            for _idx, (_gdt_pred, _gdt_label) in enumerate(zip(outs_gdt_pred, outs_gdt_label)):
                _gdt_pred = nn.functional.interpolate(_gdt_pred, size=_gdt_label.shape[2:], mode='bilinear', align_corners=True).sigmoid()
                _gdt_label = _gdt_label.sigmoid()
                loss_gdt = self.criterion_gdt(_gdt_pred, _gdt_label) if _idx == 0 else self.criterion_gdt(_gdt_pred, _gdt_label) + loss_gdt
        
        loss_pix = self.pix_loss(scaled_preds, torch.clamp(gts, 0, 1)) * loss_alpha
        self.loss_dict['loss_pix'] = loss_pix.item()
        if gt_replace == None and self.config.enable_text_aug:
            for k,v in text_loss.items():
                self.loss_dict[k] = v.item()
        
        loss = loss_pix + loss
        if self.config.out_ref:
            loss = loss + loss_gdt * 1.0
            
        self.loss_log.update(loss.item(), inputs.size(0))
        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()

    def stop_bn_track_running_stats(self, model):
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.track_running_stats = False

    def activate_bn_track_running_stats(self, model):
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.track_running_stats = True

    def start_text_aug(self):
        self.config.enable_text_aug = True
        if self.config.distributed_train:
            self.model.module.student.freeze()
        else:
            self.model.student.freeze()
    
    @retry_if_cuda_oom
    def aug_forward(self, batch, ema=False):
        batch[0], batch[1] = batch[0].to(self.device), batch[1].to(self.device)
        inputs, gts, c_reg = self.aug(batch[0], batch[1], update=True)
        inputs, gts = inputs.to(self.device), gts.to(self.device)
        loss = 0
        if self.config.enable_ref_text:
            text_features, text_attention_mask = batch[2].to(self.device), batch[3].to(self.device)
            ref_text_dict = {
                'text_features': text_features,
                'text_attention_mask': text_attention_mask
            }
            scaled_preds = self.model(inputs, ref_text_dict=ref_text_dict, ema=ema)
        else:
            scaled_preds = self.model(inputs, ema=ema)
        if self.config.out_ref and not ema:
            (outs_gdt_pred, outs_gdt_label), scaled_preds = scaled_preds
            for x,y in zip(outs_gdt_pred, outs_gdt_label):
                loss = x.sum()*0+y.sum()*0 + loss
        loss = self.aug_loss(scaled_preds, torch.clamp(gts, 0, 1))
        if not ema:
            loss = loss * -1
        loss = 0.5*c_reg + loss
        loss.backward()
        return loss.detach()
    
    @retry_if_cuda_oom
    def _train_batch_aug(self, batch):
        if self.config.distributed_train:
            for para in self.model.module.student.parameters():
                para.requires_grad = False
        else:
            for para in self.model.student.parameters():
                para.requires_grad = False
        self.model_optimizer_aug.zero_grad()
        torch.cuda.empty_cache()
        if self.config.distributed_train:
            self.stop_bn_track_running_stats(self.model.module.student)
        else:
            self.stop_bn_track_running_stats(self.model.student)

        loss = self.aug_forward(batch)
        loss = self.aug_forward(batch, ema=True) + loss
        
        self.model_optimizer_aug.step()
        if self.config.distributed_train:
            self.activate_bn_track_running_stats(self.model.module.student)
        else:
            self.activate_bn_track_running_stats(self.model.student)
        if self.config.distributed_train:
            for name, para in self.model.module.student.named_parameters():
                para.requires_grad = True
        else:
            for name, para in self.model.student.named_parameters():
                para.requires_grad = True
        return loss

    @retry_if_cuda_oom
    def train_epoch_aug(self, epoch, total_epochs):
        self.logger.key_info("[+] Training aug at epoch {} ...".format(epoch))
        if self.config.distributed_train:
            self.aug.module.reset()
        else:
            self.aug.reset()
        self.aug.train()
        self.model.train()
        for batch_idx, sup_batch in enumerate(self.labeled_dataloader):
            loss = self._train_batch_aug(sup_batch)
            if batch_idx % 20 == 0:
                info_progress = 'Epoch[{0}/{1}] Iter[{2}/{3}].'.format(epoch, total_epochs, batch_idx, len(self.labeled_dataloader))
                info_loss = 'aug Training Losses'
                info_loss += ', aug loss: {:.3f}'.format(loss.item())
                self.logger.info(' '.join((info_progress, info_loss)))
                
    @retry_if_cuda_oom
    def train_epoch(self, epoch, total_epochs):
        self.logger.key_info("[+] Training epoch {} ...".format(epoch))
        self.model.train()
        self.loss_dict = {}

        if epoch > total_epochs + self.config.IoU_finetune_last_epochs:
            self.pix_loss.lambdas_pix_last['bce'] *= 0
            self.pix_loss.lambdas_pix_last['ssim'] *= 1
            self.pix_loss.lambdas_pix_last['iou'] *= 0.5

        for batch_idx, (sup_batch, unsup_batch) in enumerate(zip(self.labeled_dataloader, self.unlabeled_dataloader)):
            if self.config.enable_img_aug:
                with torch.no_grad():
                    sup_batch[0], sup_batch[1], _ = self.aug(sup_batch[0].to(self.device), sup_batch[1].to(self.device))
                    unsup_batch[0], _, _ = self.aug(unsup_batch[0].to(self.device), unsup_batch[1].to(self.device))
            
            if self.writer and (not self.config.distributed_train or get_rank()==0):
                if batch_idx < 25:
                    self.writer.add_image(sup_batch[-2][0], torch.cat((sup_batch[0][0], sup_batch[1][0].repeat(3,1,1)), dim=-1), global_step=self.cnt)
                elif batch_idx == 25:
                    self.cnt+=1

            self._train_batch(sup_batch)
            if batch_idx % 20 == 0:
                info_progress = 'Epoch[{0}/{1}] Iter[{2}/{3}].'.format(epoch, total_epochs, batch_idx, len(self.labeled_dataloader))
                info_loss = 'Semi-Supervised Training Losses'
                for loss_name, loss_value in self.loss_dict.items():
                    info_loss += ', {}: {:.3f}'.format(loss_name, loss_value)
                if (not self.config.distributed_train) or (self.config.distributed_train and get_rank() == 0):
                    wandb.log(
                        {"Sup-"+k:v for k,v in self.loss_dict.items()},
                        step=self.global_step
                    )
                self.logger.info(' '.join((info_progress, info_loss)))

            if epoch >= self.config.sup_only_train_epoch:
                if self.config.distributed_train:
                    self.model.module.teacher.eval()
                else:
                    self.model.teacher.eval()
                inputs = unsup_batch[0].to(self.device)
                if self.config.enable_ref_text:
                    text_features, text_attention_mask = unsup_batch[2].to(self.device), unsup_batch[3].to(self.device)
                    ref_text_dict = {
                        'text_features': text_features,
                        'text_attention_mask': text_attention_mask
                    }
                    with torch.no_grad():
                        p_labels = self.model(inputs, ref_text_dict, ema=True)[-1].sigmoid()
                else:
                    with torch.no_grad():
                        p_labels = self.model(inputs, ema=True)[-1].sigmoid()
                self._train_batch(unsup_batch, gt_replace=p_labels, loss_alpha=0.1)

                if batch_idx % 20 == 0:
                    info_progress = 'Unsueprvised Training Epoch[{0}/{1}] Iter[{2}/{3}].'.format(epoch, total_epochs, batch_idx, len(self.unlabeled_dataloader))
                    info_loss = 'Unsueprvised Training Losses'
                    for loss_name, loss_value in self.loss_dict.items():
                        info_loss += ', {}: {:.3f}'.format(loss_name, loss_value)
                    self.logger.info(' '.join((info_progress, info_loss)))
                    if (not self.config.distributed_train) or (self.config.distributed_train and get_rank() == 0):
                        wandb.log(
                            {"Unsup-loss":self.loss_dict['loss_pix']},
                            step=self.global_step
                        )
            self.global_step += 1

            if epoch < self.config.sup_only_train_epoch:
                if self.config.distributed_train:
                    self.model.module.ema_update(self.global_step, 0)
                else:
                    self.model.ema_update(self.global_step, 0)
            else:
                if self.config.distributed_train:
                    self.model.module.ema_update(self.global_step)
                else:
                    self.model.ema_update(self.global_step)
                
        
        return self.loss_log.avg
        
    def launch_train(self, split, total_epochs: int):
        self.labeled_dataloader = prepare_dataloader(
            dataset = self.train_loader.dataset, batch_size=self.config.batch_size, num_workers=self.config.num_workers,
            to_be_distributed=self.config.distributed_train, is_train=True, labeled_indices=self.current_labeled_indices
        )
    
        self.unlabeled_dataloader = prepare_dataloader(
            dataset = self.train_loader.dataset, batch_size=self.config.batch_size, num_workers=self.config.num_workers,
            to_be_distributed=self.config.distributed_train, is_train=True, labeled_indices=self.current_labeled_indices, is_unsup=True
        )
        assert len(self.labeled_dataloader) == len(self.unlabeled_dataloader),"The lenth between labeled_dataloader and unlabeled_dataloader is not equal!"
        
        for epoch in range(self.epoch_st, total_epochs + 1):
            if epoch == total_epochs:
                self.start_text_aug()
            if self.config.distributed_train:
                self.unlabeled_dataloader.sampler.set_epoch(epoch)
                self.labeled_dataloader.sampler.set_epoch(epoch)
            self.train_epoch(epoch, total_epochs)
            if epoch % self.config.inner_iter ==0 and not self.config.enable_text_aug and self.config.enable_img_aug:
                self.train_epoch_aug(epoch, total_epochs)
            self.logger.success_info("[*] Epoch {} done.".format(epoch))
            self.logger.key_info("[*] Training Loss: {:.3f}".format(self.loss_log.avg))
            
            if epoch >= total_epochs - self.config.save_last and epoch % self.config.save_step == 0 and ((not self.config.distributed_train) or torch.distributed.get_rank() == 0):
                model_dict = {
                    'aug': self.aug.module.state_dict() if self.config.distributed_train else self.aug.state_dict(),
                    'model': self.model.module.state_dict() if self.config.distributed_train else self.model.state_dict(),
                    'optimizer': self.model_optimizer.state_dict(),
                    'lr_scheduler': self.model_lr_scheduler.state_dict(),
                    'epoch': epoch
                }
                self.logger.freeze_info("[*] Saving model...")
                torch.save(
                    model_dict,
                    os.path.join(self.config.ckpt_dir, 'split{}_model_{}.pth'.format(split, epoch))
                )
                self.logger.success_info("[*] Model saved.")
            if self.config.distributed_train:
                torch.distributed.barrier()
            if epoch % self.config.eval_step == 0:
                if (self.config.distributed_train and get_rank() == 0) or (not self.config.distributed_train):
                    self.evaluate_online(epoch, is_last=(epoch==total_epochs))

    def evaluate_online(self, epoch, is_last=False):
        if self.config.distributed_train and get_rank() != 0:
            return
        self.logger.key_info("[+] Online evaluation created, model epoch: {}...".format(epoch))
        self.model.eval()
        evaluator = Evaluator.from_exists(
            config=self.config,
            logger=self.logger,
            device=self.device,
            model=self.model if not self.config.distributed_train else self.model.module
        )
        for testset_name, testloader in self.test_loaders.items():
            evaluator.inference_on_dataset(testloader, testset_name)
            result = evaluator.evaluate_inference_result(testloader, testset_name)
            wandb.log(
                {
                    'T-MAE': result['mae'],
                    'T-maxFm': result['f_max'],
                    'T-wFmeasure': result['f_wfm'],
                    'T-SMeasure': result['sm'],
                    'T-meanEm': result['e_mean'],
                    'T-meanFm': result['f_mean']
                },step=wandb.run.step
            )
            if is_last:
                wandb.log(
                {
                    'F-MAE': result['mae'],
                    'F-maxFm': result['f_max'],
                    'F-wFmeasure': result['f_wfm'],
                    'F-SMeasure': result['sm'],
                    'F-meanEm': result['e_mean'],
                    'F-meanFm': result['f_mean']
                }
            )
        self.logger.key_info("[+] Online evaluation done...")


