import torch
import torch.nn as nn


def slicd_Wasserstein_distance(x1, x2, n_projection=128):
    x1 = x1.flatten(-2).transpose(1, 2).contiguous() # (b, 3, h, w) -> (b, n, 3)
    x2 = x2.flatten(-2).transpose(1, 2).contiguous()
    rand_proj = torch.randn(3, n_projection, device=x1.device)
    rand_proj = rand_proj / (rand_proj.norm(2, dim=0, keepdim=True) + 1e-12)
    sorted_proj_x1 = torch.matmul(x1, rand_proj).sort(0)[0]
    sorted_proj_x2 = torch.matmul(x2, rand_proj).sort(0)[0]
    return (sorted_proj_x1 - sorted_proj_x2).pow(2).mean()


class AugmentationContainer(nn.Module):
    def __init__(
            self, c_aug, g_aug, c_reg_coef=0,
            normalizer=None, replay_buffer=None, n_chunk=16):
        super().__init__()
        self.c_aug = c_aug
        self.g_aug = g_aug
        self.c_reg_coef = c_reg_coef
        self.normalizer = normalizer
        self.replay_buffer = replay_buffer
        self.n_chunk = n_chunk

    def get_params(self, x, c, c_aug, g_aug):
        # sample noise vector from unit gauss
        noise = x.new(x.shape[0], self.g_aug.n_dim if self.g_aug else self.c_aug.n_dim).normal_()
        target = self.normalizer(x) if self.normalizer is not None else x
        # sample augmentation parameters
        if g_aug != None:
            grid = g_aug(target, noise, c)
        else:
            grid = None
        if c_aug:
            scale, shift = c_aug(target, noise, c)
        else :
            scale, shift = None, None
        return (scale, shift), grid

    def augmentation(self, x, c, c_aug, g_aug, y=None, update=False):
        c_param, g_param = self.get_params(x, c, c_aug, g_aug)
        aug_y = y
        # color augmentation
        if c_aug != None:
            aug_x = c_aug.transform(x, *c_param)
        else:
            aug_x = x
        # color regularization
        if update and self.c_reg_coef > 0 and c_aug != None:
            if self.normalizer is not None:
                swd = self.c_reg_coef * slicd_Wasserstein_distance(self.normalizer(x), self.normalizer(aug_x))
            else:
                swd = self.c_reg_coef * slicd_Wasserstein_distance(x, aug_x)
        else:
            swd = torch.zeros(1, device=x.device)
        # geometric augmentation
        if g_aug != None:
            aug_x = g_aug.transform(aug_x, g_param)
            if y != None:
                with torch.no_grad():
                    aug_y = g_aug.transform(y, g_param)
        return aug_x, aug_y, swd

    def forward(self, x, y=None, c=None, update=False):
        x, y, swd = self.augmentation(x=x, y=y, c=c, c_aug=self.c_aug, g_aug=self.g_aug, update=update)
        x = self.normalizer(x)
        if y != None:
            return x, y, swd
        else:
            return x, swd

    def get_augmentation_model(self):
        return nn.ModuleList([self.c_aug, self.g_aug])

    def reset(self):
        # initialize parameters
        if self.c_aug:  
            self.c_aug.reset()
        if self.g_aug:
            self.g_aug.reset()
