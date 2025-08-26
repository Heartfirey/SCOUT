import os
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torchvision.transforms import ToTensor, Grayscale
from torch.utils.data import DataLoader, Dataset
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


from tqdm import tqdm

def extract_image_feature(images):
    def to_frequency(tensor):
        img = Grayscale()(tensor)
        img = torch.fft.fftshift(torch.fft.fftn(img))
        return torch.log(torch.abs(img) + 1).flatten()
    
    def to_color_histogram(tensor):
        img = tensor.permute(1, 2, 0).cpu().numpy()     # convert to HWC mode
        hist, _ = np.histogramdd(img.reshape(-1, 3), bins=16, range=[(0, 1), (0, 1), (0, 1)])
        return torch.from_numpy(hist.flatten())
    
    def to_texture_features(tensor):
        img = Grayscale()(tensor).squeeze(0).cpu().numpy().astype(np.uint8)
        gcm = graycomatrix(img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(gcm, 'contrast')
        return torch.from_numpy(contrast.flatten())
    
    def to_edge_features(tensor):
        img = Grayscale()(tensor).squeeze(0).cpu().numpy()
        lbp = local_binary_pattern(img, 8, 1, method='uniform')
        hist, _ = np.histogram(lbp, bins=59, range=(0, 59))
        return torch.from_numpy(hist.flatten())
    
    features = []

    freq_features = to_frequency(images)
    color_histogram = to_color_histogram(images)
    texture_features = to_texture_features(images)
    edge_features = to_edge_features(images)
    features.append(torch.cat([freq_features, color_histogram, texture_features, edge_features]))
    return features

def sample_centers(config, dataset: Dataset, sample_size: int, from_cache: str='./cache/trainset_pre_features.pt') -> list:
    if os.path.isfile(from_cache):
        print('[-] load from cache on disk...')
        all_features = torch.load(from_cache)
    else:
        all_features = []
        for each_input in tqdm(dataset):
            cur_image = each_input[0]
            all_features.append(extract_image_feature(cur_image)[0])
        print('[!] save cache to disk...')
        torch.save(all_features, from_cache)
        
    
    scaler = StandardScaler()
    all_features = scaler.fit_transform(all_features)
    print("Starting perform PCA...")
    pca = PCA(n_components=1024)
    all_features = pca.fit_transform(all_features)
    print('Starting perform KMeans...')
    kmeans = KMeans(n_clusters=sample_size, random_state=config.rand_seed)
    kmeans.fit(all_features)
    
    centers = kmeans.cluster_centers_
    indices = []
    for each_center in tqdm(centers):
        distances = np.linalg.norm(all_features - each_center, axis=1)
        indices.append(np.argmin(distances))
    return indices



# if __name__ == "__main__":
#     fake_image = torch.rand(2, 3, 256, 256)
#     features = extract_image_feature(fake_image)
#     for each_item in features:
#         print(each_item.shape)
