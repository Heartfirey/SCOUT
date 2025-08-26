import torch
import numpy as np
from tqdm import tqdm

class AdversarySampler:
    def __init__(self, budget):
        self.budget = budget

    def sample(self, vae, discriminator, data, labeled_indices, device):
        all_preds = []
        all_indices = []

        dataset = data.dataset
        all_indices = list(range(len(dataset)))
        
        vae.to(device)
        discriminator.to(device)
        
        for indice, data in tqdm(enumerate(data)):
            if indice in labeled_indices:
                continue
            images = data[0].to(device)

            with torch.no_grad():
                _, _, mu, _ = vae(images)
                preds = discriminator(mu)

            preds = preds.cpu().data
            if type(preds) is list:
                all_preds.extend(preds)
                all_indices.extend(indice)
            else:
                all_preds.append(preds)
                all_indices.append(indice)

        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        # need to multiply by -1 to be able to use torch.topk 
        all_preds *= -1

        # select the points which the discriminator things are the most likely to be unlabeled
        _, querry_indices = torch.topk(all_preds, int(self.budget))
        querry_pool_indices = np.asarray(all_indices)[querry_indices]

        return querry_pool_indices
        