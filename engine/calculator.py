import os
import cv2
import argparse
import torch
import torch.nn as nn
import torch.utils.data as data
import prettytable as pt
import numpy as np

from sklearn.neighbors import KernelDensity
from glob import glob
from tqdm import tqdm

from config import Config
from utils import Logger, save_tensor_img
from utils.maker import make_codebook
from data import init_testloaders
from models import build_model_eval

from .metrics import statistics

class ActiveCalculator:
    def __init__(self, config, logger, device, model, aug, dataloaders):
        self.labeled_dataloader, self.unlabeled_dataloader = dataloaders
        self.config = config
        self.logger = logger
        self.device = device
        self.model = model
        self.aug = aug
        self.logger.success_info("[o] ActiveCalculator is ready!")
        self.statistics = statistics()
        self.sms = None
        self.maes = None

    def IUE_score(self, preds, preds_ema, p_labels):
        sm, mae = self.statistics.step(gts=preds_ema, preds=preds)
        if self.sms != None:
            self.sms = torch.cat((self.sms, sm), dim=0)
            self.maes = torch.cat((self.maes, mae), dim=0)
        else:
            self.sms = sm
            self.maes = mae

    def kde_norm(self, data, bandwidth=0.2, kernel='gaussian', num_points=1000):
        data = np.asarray(data).reshape(-1, 1)

        kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
        kde.fit(data)

        log_density = kde.score_samples(data)
        density = np.exp(log_density)
        cdf = np.cumsum(density) / np.sum(density)
        normalized_data = np.interp(np.linspace(0, 1, num_points), cdf, np.sort(data.flatten()))

        return normalized_data

    def pca(self, X, n_components):
        X = X.float()

        X_mean = torch.mean(X, dim=0)
        X_centered = X - X_mean
        U, S, V = torch.linalg.svd(X_centered)
        principal_components = V[:n_components].T
        X_reduced = torch.mm(X_centered, principal_components)

        return X_reduced

    def cal_active_score(self):
        img_scores = None
        file_name = []
        score_list = list()
        for batch in tqdm(self.unlabeled_dataloader, total=len(self.unlabeled_dataloader),desc="Calculate score:"):
            with torch.no_grad():
                inputs = batch[0].to(self.device)
                
                text_features, text_attention_mask = batch[2].to(self.device), batch[3].to(self.device)
                ref_text_dict = {
                    'text_features': text_features,
                    'text_attention_mask': text_attention_mask
                }
                p_labels = (self.model(inputs, ref_text_dict=ref_text_dict)[-1].sigmoid()>0.5).float()
                inputs, p_labels, _ = self.aug(inputs, p_labels)
                scaled_preds = self.model(inputs, ref_text_dict=ref_text_dict)[-1].sigmoid()
                scaled_preds_ema = self.model(inputs, ref_text_dict=ref_text_dict, ema=True)[-1].sigmoid()
                self.IUE_score(scaled_preds, scaled_preds_ema, p_labels)
                file_name += batch[-2]
        self.kde_norm(self.sms)
        self.kde_norm(self.maes)
        img_scores = self.sms - self.maes
        score_list += [(name, img_score.item()) for name, img_score in zip(file_name, img_scores)]
        score_list = sorted(score_list, key=lambda x: x[1])
        return score_list

