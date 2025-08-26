import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding

class Vector_Quantizer(nn.Module):
    def __init__(self, codebook_num, codebook_dim, e_dim):
        super().__init__()
        self.codebook_dim = codebook_dim
        self.e_dim = e_dim
        self.codebook_num = codebook_num
        self.codebook = Embedding(self.codebook_num, self.codebook_dim)
        self.proj = nn.Linear(self.e_dim, self.codebook_dim)
        self.soft_max = nn.Softmax(dim=-1)
        self.mseloss = nn.MSELoss(reduction='none')

    def load_ckp(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        weight = checkpoint['weight'].cuda()
        category = checkpoint['category']
        self.codebook_num = weight.shape[0]
        self.codebook_dim = weight.shape[1]
        del self.codebook
        print('reinit codebook from {}'.format(path))
        self.codebook = Embedding(self.codebook_num, self.codebook_dim)
        self.codebook.weight.data = weight.float()
        self.codebook.weight.requires_grad = True
        self.ori_weight = weight.clone().float()
        self.category = category

    def sim(self, x1, x2, dim=-1, eps=1e-8):
        x1_norm = torch.norm(x1, p=2, dim=dim, keepdim=True).clamp(min=eps)
        x2_norm = torch.norm(x2, p=2, dim=dim, keepdim=True).clamp(min=eps)
        x1_normalized = x1 / x1_norm
        x2_normalized = x2 / x2_norm
        
        cosine_sim = torch.sum(x1_normalized.unsqueeze(2) * x2_normalized.unsqueeze(1), dim=dim)

        return 1 - cosine_sim

    def dist(self, x1, x2):
        assert x1.shape == x2.shape, "Shapes of x1 and x2 must match."
        dot_product = torch.sum(x1 * x2, dim=-1)

        norm_x1 = torch.norm(x1, p=2, dim=-1)
        norm_x2 = torch.norm(x2, p=2, dim=-1)
        epsilon = 1e-8
        cosine_similarity = dot_product / (norm_x1 * norm_x2 + epsilon)

        return 1 - cosine_similarity

    def soft_match(self, inputs):
        d = self.sim(inputs, self.codebook.weight.unsqueeze(0))
        d = self.soft_max(d)
        c = (d.unsqueeze(-1) * self.codebook.weight[None,None,...]).sum(dim=2)
        return c
    
    def cal_loss(self, clue, text_emb, attn_mask):
        text_emb_sotf = self.soft_match(text_emb)
        weight = self.sim(clue, text_emb).clamp(0,1).detach()
        loss = ((weight.squeeze(1) * self.dist(text_emb.detach(), text_emb_sotf)) * ~attn_mask).sum()
        return loss
    
    def cal_ccmloss(self, inputs, hash_labels):
        with torch.no_grad():  
            mask = torch.tensor(['COD10K' in name for name in hash_labels]).to(inputs.device)
            if mask.sum() == 0:
                return torch.tensor(0.).to(inputs.device)
            inputs = inputs[mask]
            hash_labels = [label for label, m in zip(hash_labels, mask) if m]
            target = torch.stack([self.ori_weight[self.category[name.split('-')[-2]]] for name in hash_labels], dim=0).to(inputs.device)
        loss = self.mseloss(inputs, target.unsqueeze(1))
        return loss.mean()

    def get_code_indices(self, flat_x, target=None):
        # flag = self.training
        flat_x = F.normalize(flat_x, p=2, dim=1)
        weight = self.embeddings.weight
        weight = F.normalize(weight, p=2, dim=1)
        distances = (
                torch.sum(flat_x ** 2, dim=1, keepdim=True) +
                torch.sum(weight ** 2, dim=1) -
                2. * torch.matmul(flat_x, weight.t())
        )
        encoding_indices = torch.argmin(distances, dim=1)
        return encoding_indices

    def forward(self, inputs, text_emb, attn_mask, hash_labels=None, update=False):
        bs = inputs.shape[0]
        inputs = self.proj(inputs)
        loss = 0
        if update:
            loss = self.cal_ccmloss(inputs, hash_labels)

        clue = self.soft_match(inputs)
        
        if update:
            loss = self.cal_loss(clue, text_emb, attn_mask) + loss

        
        new_text_emb = text_emb.clone()
        new_attn_mask = attn_mask.clone()

        for i in range(bs):
            valid_length = len(attn_mask[i]) - attn_mask[i].sum().item()
            if valid_length < 64:
                new_text_emb[i][valid_length] = new_text_emb[i][valid_length-1]
                new_text_emb[i][valid_length-1] = clue[i, 0].clone()
                new_attn_mask[i][valid_length] = False
        return new_text_emb, new_attn_mask, loss