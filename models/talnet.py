import torch
import torch.nn as nn
import torch.distributed as dist
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torchvision.transforms import functional as TF
from torchvision.models import vgg16, vgg16_bn
from torchvision.models import resnet50
from kornia.filters import laplacian
from torchvision import transforms

from models.backbones.build_backbone import build_backbone
from models.modules.decoder_blocks import BasicDecBlk, ResBlk, HierarAttDecBlk
from models.modules.lateral_blocks import BasicLatBlk
from models.modules.aspp import ASPP, ASPPDeformable
from models.modules.ing import *
from models.refinement.refiner import Refiner, RefinerPVTInChannels4, RefUNet
from models.refinement.stem_layer import StemLayer
from models.modules.mmfusion import MultiModalFusion
from models.modules.textproc import TextProcessor
from models.modules.text_aug import Text_Aug

class ModelEMA(nn.Module):
    def __init__(self, config, bb_pretrained=True, alpha=0.999):
        super(ModelEMA, self).__init__()
        self.alpha = alpha
        self.student = TalNet(config=config, bb_pretrained=bb_pretrained)
        self.teacher = TalNet(config=config, bb_pretrained=bb_pretrained, ema=True)
        self._init_teacher_params()

    def forward(self, x, ref_text_dict=None, hash_labels=None, gt=None, ema=False):
        if ema:
            return self.teacher(x, ref_text_dict, hash_labels, gt)
        else:
            return self.student(x, ref_text_dict, hash_labels, gt)
    
    def load_codebook(self, ckp_path):
        self.student.text_aug.VQ.load_ckp(ckp_path)
        self.teacher.text_aug.VQ.load_ckp(ckp_path)
        
    def ema_update(self, global_step, alpha=None):
        with torch.no_grad():
            if alpha == None:
                alpha = min(1 - 1 / (global_step + 1), self.alpha)
            for ema_param, param in zip(self.teacher.parameters(), self.student.parameters()):
                ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
            for ema_buff, buff in zip(self.teacher.buffers(), self.student.buffers()):
                ema_buff = ema_buff.float()
                buff = buff.float()
                ema_buff.data.mul_(alpha).add_(buff.data, alpha=1 - alpha)

    def sync(self):
        for param in self.teacher.parameters():
            dist.all_reduce(param.data) 
            param.data /= dist.get_world_size()  

    def _init_teacher_params(self):
        for param_t, param_s in zip(self.teacher.parameters(), self.student.parameters()):
            param_t.data.copy_(param_s.data)
    def train(self, mode=True):
        super().train(mode)
        self.student.train(mode)
        self.teacher.eval()  

    def eval(self):
        super().eval()
        self.student.eval()
        self.teacher.eval()

class TalNet(nn.Module):
    def __init__(self, config, bb_pretrained=True, ema=False):
        super(TalNet, self).__init__()
        self.config = config
        self.epoch = 1
        self.ema = ema
        self.bb = build_backbone(self.config.backbone, config=config, pretrained=bb_pretrained)
        if self.config.clip_img_size:
            self.prepare_vision_encoder()
        channels = self.config.lateral_channels_in_collection
        if self.config.squeeze_block:
            self.squeeze_module = nn.Sequential(*[
                eval(self.config.squeeze_block.split('_x')[0])(config, channels[0]+sum(self.config.cxt), channels[0])
                for _ in range(eval(self.config.squeeze_block.split('_x')[1]))
            ])
        
        self.decoder = Decoder(config, channels)

        # Referring text module
        channels_rev = channels[:]
        channels_rev.reverse()
        if self.config.enable_ref_text:
            # scale_factor = 2 if self.config.mul_scl_ipt == "cat" else 1
            self.in_img_proj = nn.ModuleList([
                nn.Conv2d(channels_rev[i] , self.config.img_channel_dim, 1, 1, 0)
                for i in range(0, len(channels_rev))
            ])
            if self.config.clip_img_size:
                self.clip_vision_proj =nn.Conv2d(1024 , self.config.img_channel_dim, 1, 1, 0)

            self.out_img_proj = nn.ModuleList([
                nn.Conv2d(self.config.img_channel_dim, channels_rev[i], 1, 1, 0)
                for i in range(0, len(channels_rev))
            ])
            self.text_processor = TextProcessor(
                in_dim=self.config.text_encoders_out_dim,
                out_dim=self.config.text_embed_dim
            )
            self.fushion_layer = MultiModalFusion(
                img_dim=self.config.img_channel_dim,
                text_dim=self.config.text_embed_dim,
                config=self.config
            )
            self.text_aug = Text_Aug(
                config=config,
                text_dim=config.text_encoders_out_dim,
                img_dim=config.lateral_channels_in_collection[0],
                dropout=config.vq_attn_dropout,
                codebook_num=config.codebook_num,
                codebook_dim=config.codebook_dim,
                TF_block_num=config.TF_block_num
            )

        if ema:
            for param in self.parameters():
                param.requires_grad=False

    def freeze(self):
        for name, param in self.named_parameters():
            if not 'text_aug' in name:# and not 'fushion_layer' in name:
                param.requires_grad = False

    def prepare_vision_encoder(self):
        from transformers import CLIPVisionModel
        if self.config.text_encoders == 'clip-base':
            self.vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        elif self.config.text_encoders == 'clip-large':
            self.vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336")
        else:
            raise NotImplementedError(f"Text encoder {self.config.text_encoders} not supported.")
        assert self.vision_encoder is not None, f"Vision encoder {self.config.vision_encoders} not found."
        
        for p in self.vision_encoder.parameters():
            p.requires_grad_(False)
        self.vision_encoder = self.vision_encoder.to('cuda')

    def denormalize(self, tensor, imgsize=336, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):

        mean = torch.tensor(mean).view(1, -1, 1, 1).to(tensor.device)
        std = torch.tensor(std).view(1, -1, 1, 1).to(tensor.device)
    
        tensor = tensor * std + mean
        tensor = F.interpolate(tensor, size=imgsize, mode='bilinear', align_corners=True)
        tensor = (tensor - mean) / std
        return tensor

    def forward_enc(self, x, ref_text_dict=None, hash_labels=None, gt=None):
        if self.config.backbone in ['vgg16', 'vgg16bn', 'resnet50']:
            x1 = self.bb.conv1(x); x2 = self.bb.conv2(x1); x3 = self.bb.conv3(x2); x4 = self.bb.conv4(x3)
        else:
            x1, x2, x3, x4 = self.bb(x)
            B, C, H, W = x.shape
            x1_, x2_, x3_, x4_ = self.bb(F.interpolate(x, size=(H//2, W//2), mode='bilinear', align_corners=True))
            x1 = torch.cat([x1, F.interpolate(x1_, size=x1.shape[2:], mode='bilinear', align_corners=True)], dim=1)
            x2 = torch.cat([x2, F.interpolate(x2_, size=x2.shape[2:], mode='bilinear', align_corners=True)], dim=1)
            x3 = torch.cat([x3, F.interpolate(x3_, size=x3.shape[2:], mode='bilinear', align_corners=True)], dim=1)
            x4 = torch.cat([x4, F.interpolate(x4_, size=x4.shape[2:], mode='bilinear', align_corners=True)], dim=1)
            
            text_loss = 0
            if self.config.enable_ref_text:
                assert ref_text_dict is not None, "Referring text dict is not provided!"

                if self.config.enable_text_aug:
                    ref_text_dict, text_loss = self.text_aug(ref_text_dict, x4.detach(), hash_labels, gt)
                ref_text_dict = self.text_processor(ref_text_dict)
                x1_proj, x2_proj, x3_proj, x4_proj = [self.in_img_proj[i](x) for i, x in enumerate([x1, x2, x3, x4])]
                if self.config.clip_img_size:
                    vision_features = self.denormalize(x)
                    vision_features = self.vision_encoder(vision_features).last_hidden_state
                    vision_features = vision_features[:,1:,:].reshape(vision_features.shape[0], int(vision_features.shape[1]**0.5), int(vision_features.shape[1]**0.5), vision_features.shape[2]).permute(0, 3, 1, 2)
                    vision_features = self.clip_vision_proj(vision_features)
                    x1_proj, x2_proj, x3_proj, x4_proj = (x_proj + F.interpolate(vision_features, size=x_proj.shape[2:], mode='bilinear', align_corners=True) for x_proj in (x1_proj, x2_proj, x3_proj, x4_proj))

                (x1_proj, x2_proj, x3_proj, x4_proj) = self.fushion_layer([x1_proj, x2_proj, x3_proj, x4_proj], ref_text_dict)
                x1, x2, x3, x4 = [self.out_img_proj[i](x) for i, x in enumerate([x1_proj, x2_proj, x3_proj, x4_proj])]
        
        if self.config.cxt:
            x4 = torch.cat(
                (
                    *[
                        F.interpolate(x1, size=x4.shape[2:], mode='bilinear', align_corners=True),
                        F.interpolate(x2, size=x4.shape[2:], mode='bilinear', align_corners=True),
                        F.interpolate(x3, size=x4.shape[2:], mode='bilinear', align_corners=True),
                    ][-len(self.config.cxt):],
                    x4
                ),
                dim=1
            )
        return (x1, x2, x3, x4), text_loss

    def forward_ori(self, x, ref_text_dict=None, hash_labels=None, gt=None):
        ########## Encoder ##########
        (x1, x2, x3, x4), text_loss = self.forward_enc(x, ref_text_dict, hash_labels, gt)
        if self.config.squeeze_block:
            x4 = self.squeeze_module(x4)
        ########## Decoder ##########
        features = [x, x1, x2, x3, x4]
        if self.training and self.config.out_ref:
            features.append(laplacian(torch.mean(x, dim=1).unsqueeze(1), kernel_size=5))
        scaled_preds = self.decoder(features)
        return scaled_preds, text_loss

    def forward(self, x, ref_text_dict=None, hash_labels=None, gt=None):
        scaled_preds, text_loss = self.forward_ori(x, ref_text_dict, hash_labels, gt)
        if gt != None and self.config.enable_text_aug:
            return scaled_preds, text_loss
        return scaled_preds


class Decoder(nn.Module):
    def __init__(self, config, channels):
        super(Decoder, self).__init__()
        self.config = config
        DecoderBlock = eval(self.config.dec_blk)
        LateralBlock = BasicLatBlk

        if self.config.dec_ipt:
            self.split = self.config.dec_ipt_split
            N_dec_ipt = 64
            DBlock = SimpleConvs
            ic = 64
            ipt_cha_opt = 1
            self.ipt_blk4 = DBlock(2**8*3 if self.split else 3, [N_dec_ipt, channels[0]//8][ipt_cha_opt], inter_channels=ic)
            self.ipt_blk3 = DBlock(2**6*3 if self.split else 3, [N_dec_ipt, channels[1]//8][ipt_cha_opt], inter_channels=ic)
            self.ipt_blk2 = DBlock(2**4*3 if self.split else 3, [N_dec_ipt, channels[2]//8][ipt_cha_opt], inter_channels=ic)
            self.ipt_blk1 = DBlock(2**0*3 if self.split else 3, [N_dec_ipt, channels[3]//8][ipt_cha_opt], inter_channels=ic)
        else:
            self.split = None

        self.decoder_block4 = DecoderBlock(config,channels[0], channels[1])
        self.decoder_block3 = DecoderBlock(config,channels[1]+([N_dec_ipt, channels[0]//8][ipt_cha_opt] if self.config.dec_ipt else 0), channels[2])
        self.decoder_block2 = DecoderBlock(config,channels[2]+([N_dec_ipt, channels[1]//8][ipt_cha_opt] if self.config.dec_ipt else 0), channels[3])
        self.decoder_block1 = DecoderBlock(config,channels[3]+([N_dec_ipt, channels[2]//8][ipt_cha_opt] if self.config.dec_ipt else 0), channels[3]//2)
        self.conv_out1 = nn.Sequential(nn.Conv2d(channels[3]//2+([N_dec_ipt, channels[3]//8][ipt_cha_opt] if self.config.dec_ipt else 0), 1, 1, 1, 0))

        self.lateral_block4 = LateralBlock(channels[1], channels[1])
        self.lateral_block3 = LateralBlock(channels[2], channels[2])
        self.lateral_block2 = LateralBlock(channels[3], channels[3])

        if self.config.ms_supervision:
            self.conv_ms_spvn_4 = nn.Conv2d(channels[1], 1, 1, 1, 0)
            self.conv_ms_spvn_3 = nn.Conv2d(channels[2], 1, 1, 1, 0)
            self.conv_ms_spvn_2 = nn.Conv2d(channels[3], 1, 1, 1, 0)

            if self.config.out_ref:
                _N = 16
                # self.gdt_convs_4 = nn.Sequential(nn.Conv2d(channels[1], _N, 3, 1, 1), nn.BatchNorm2d(_N), nn.ReLU(inplace=True))
                self.gdt_convs_3 = nn.Sequential(nn.Conv2d(channels[2], _N, 3, 1, 1), nn.BatchNorm2d(_N), nn.ReLU(inplace=True))
                self.gdt_convs_2 = nn.Sequential(nn.Conv2d(channels[3], _N, 3, 1, 1), nn.BatchNorm2d(_N), nn.ReLU(inplace=True))

                # self.gdt_convs_pred_4 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
                self.gdt_convs_pred_3 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
                self.gdt_convs_pred_2 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
                
                # self.gdt_convs_attn_4 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
                self.gdt_convs_attn_3 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
                self.gdt_convs_attn_2 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))


    def get_patches_batch(self, x, p):
        _size_h, _size_w = p.shape[2:]
        patches_batch = []
        for idx in range(x.shape[0]):
            columns_x = torch.split(x[idx], split_size_or_sections=_size_w, dim=-1)
            patches_x = []
            for column_x in columns_x:
                patches_x += [p.unsqueeze(0) for p in torch.split(column_x, split_size_or_sections=_size_h, dim=-2)]
            patch_sample = torch.cat(patches_x, dim=1)
            patches_batch.append(patch_sample)
        return torch.cat(patches_batch, dim=0)

    def forward(self, features):
        if self.training and self.config.out_ref:
            outs_gdt_pred = []
            outs_gdt_label = []
            x, x1, x2, x3, x4, gdt_gt = features
        else:
            x, x1, x2, x3, x4 = features
        outs = []
        p4 = self.decoder_block4(x4)
        m4 = self.conv_ms_spvn_4(p4) if self.config.ms_supervision else None
        _p4 = F.interpolate(p4, size=x3.shape[2:], mode='bilinear', align_corners=True)
        _p3 = _p4 + self.lateral_block4(x3)
        if self.config.dec_ipt:
            patches_batch = self.get_patches_batch(x, _p3) if self.split else x
            _p3 = torch.cat((_p3, self.ipt_blk4(F.interpolate(patches_batch, size=x3.shape[2:], mode='bilinear', align_corners=True))), 1)

        p3 = self.decoder_block3(_p3)
        m3 = self.conv_ms_spvn_3(p3) if self.config.ms_supervision else None
        if self.config.out_ref:
            p3_gdt = self.gdt_convs_3(p3)
            if self.training:
                # >> GT:
                # m3 --dilation--> m3_dia
                # G_3^gt * m3_dia --> G_3^m, which is the label of gradient
                m3_dia = m3
                gdt_label_main_3 = gdt_gt * F.interpolate(m3_dia, size=gdt_gt.shape[2:], mode='bilinear', align_corners=True)
                outs_gdt_label.append(gdt_label_main_3)
                # >> Pred:
                # p3 --conv--BN--> F_3^G, where F_3^G predicts the \hat{G_3} with xx
                # F_3^G --sigmoid--> A_3^G
                gdt_pred_3 = self.gdt_convs_pred_3(p3_gdt)
                outs_gdt_pred.append(gdt_pred_3)
            gdt_attn_3 = self.gdt_convs_attn_3(p3_gdt).sigmoid()
            # >> Finally:
            # p3 = p3 * A_3^G
            p3 = p3 * gdt_attn_3
        _p3 = F.interpolate(p3, size=x2.shape[2:], mode='bilinear', align_corners=True)
        _p2 = _p3 + self.lateral_block3(x2)
        if self.config.dec_ipt:
            patches_batch = self.get_patches_batch(x, _p2) if self.split else x
            _p2 = torch.cat((_p2, self.ipt_blk3(F.interpolate(patches_batch, size=x2.shape[2:], mode='bilinear', align_corners=True))), 1)

        p2 = self.decoder_block2(_p2)
        m2 = self.conv_ms_spvn_2(p2) if self.config.ms_supervision else None
        if self.config.out_ref:
            p2_gdt = self.gdt_convs_2(p2)
            if self.training:
                # >> GT:
                m2_dia = m2
                gdt_label_main_2 = gdt_gt * F.interpolate(m2_dia, size=gdt_gt.shape[2:], mode='bilinear', align_corners=True)
                outs_gdt_label.append(gdt_label_main_2)
                # >> Pred:
                gdt_pred_2 = self.gdt_convs_pred_2(p2_gdt)
                outs_gdt_pred.append(gdt_pred_2)
            gdt_attn_2 = self.gdt_convs_attn_2(p2_gdt).sigmoid()
            # >> Finally:
            p2 = p2 * gdt_attn_2
        _p2 = F.interpolate(p2, size=x1.shape[2:], mode='bilinear', align_corners=True)
        _p1 = _p2 + self.lateral_block2(x1)
        if self.config.dec_ipt:
            patches_batch = self.get_patches_batch(x, _p1) if self.split else x
            _p1 = torch.cat((_p1, self.ipt_blk2(F.interpolate(patches_batch, size=x1.shape[2:], mode='bilinear', align_corners=True))), 1)

        _p1 = self.decoder_block1(_p1)
        _p1 = F.interpolate(_p1, size=x.shape[2:], mode='bilinear', align_corners=True)
        if self.config.dec_ipt:
            patches_batch = self.get_patches_batch(x, _p1) if self.split else x
            _p1 = torch.cat((_p1, self.ipt_blk1(F.interpolate(patches_batch, size=x.shape[2:], mode='bilinear', align_corners=True))), 1)
        p1_out = self.conv_out1(_p1)

        if self.config.ms_supervision:
            outs.append(m4)
            outs.append(m3)
            outs.append(m2)
        outs.append(p1_out)
        return outs if not (self.config.out_ref and self.training) else ([outs_gdt_pred, outs_gdt_label], outs)


class SimpleConvs(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, inter_channels=64
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, inter_channels, 3, 1, 1)
        self.conv_out = nn.Conv2d(inter_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        return self.conv_out(self.conv1(x))
