import os
import cv2
import json
import torch
import pickle
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils import data
from torchvision import transforms

from utils import preproc
from utils import path_to_image
from utils import Logger


Image.MAX_IMAGE_PIXELS = None       # remove DecompressionBombWarning

logger = Logger(name='Dataset', path='./data/logs/dataset.txt')

class MyData(data.Dataset):
    def __init__(self, config, datasets, image_size, is_train=True, text_replace=None, use_text_replace=False):
        self.config = config
        self.size_train = image_size
        self.size_test = image_size
        self.keep_size = not config.img_size
        self.data_size = (config.img_size, config.img_size)
        self.is_train = is_train
        self.load_all = config.load_all
        # self.device = config.device
        self.transform_image = transforms.Compose([
            transforms.Resize(self.data_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ][self.load_all or self.keep_size:-1 if self.is_train and config.enable_img_aug else None])
        self.transform_label = transforms.Compose([
            transforms.Resize(self.data_size),
            transforms.ToTensor(),
        ][self.load_all or self.keep_size:])
        dataset_root = os.path.join(config.data_root_dir, config.task)
        # datasets can be a list of different datasets for training on combined sets.
        self.image_paths = []
        for dataset in datasets.split('+'):
            image_root = os.path.join(dataset_root, dataset, 'im')
            self.image_paths += [os.path.join(image_root, p) for p in os.listdir(image_root)]
        self.label_paths = []
        for p in self.image_paths:
            for ext in ['.png', '.jpg', '.PNG', '.JPG', '.JPEG']:
                ## 'im' and 'gt' may need modifying
                p_gt = p.replace('/im/', '/gt/').replace('.'+p.split('.')[-1], ext)
                if os.path.exists(p_gt):
                    self.label_paths.append(p_gt)
                    break
        self.image_to_idx = {os.path.basename(image_path).split('.')[0]: idx for idx, image_path in enumerate(self.image_paths)}
        self.unlabeled_indices = []
        
        if config.enable_ref_text is True:
            self.enc_anno_dict = self.load_dataset_captions(
                os.path.join(dataset_root, datasets.split('+')[0], 'ref_anno.json'), using_cache=config.using_ref_cache, prefix=datasets, replace_all_sentense=(text_replace if use_text_replace else None)
            )
            if text_replace is not None:
                self.set_unlabeled_ref_text(text_replace)
            else:
                self.unlabeled_reference_text = None
        self.text_embed_loaded = []
        if self.load_all:
            self.images_loaded, self.labels_loaded, self.text_embed_loaded = [], [], []
            # for image_path, label_path in zip(self.image_paths, self.label_paths):
            for image_path, label_path in tqdm(zip(self.image_paths, self.label_paths), total=len(self.image_paths)):
                _image = path_to_image(image_path, size=(config.img_size, config.img_size), color_type='rgb')
                _label = path_to_image(label_path, size=(config.img_size, config.img_size), color_type='gray')
                self.images_loaded.append(_image)
                self.labels_loaded.append(_label)
                if config.enable_ref_text is True:
                    image_filename = image_path.split('/')[-1]
                    assert image_filename in self.enc_anno_dict.keys()
                    self.text_embed_loaded.append(self.enc_anno_dict[image_filename])
        else:
            if config.enable_ref_text is True:
                for image_path, label_path in tqdm(zip(self.image_paths, self.label_paths), total=len(self.image_paths)):
                    image_filename = image_path.split('/')[-1]
                    assert image_filename in self.enc_anno_dict.keys()
                    self.text_embed_loaded.append(self.enc_anno_dict[image_filename])

    def random_boolean(self, probability_of_true=0.5):
        return random.random() < probability_of_true
    
    def __getitem__(self, index):
        if self.load_all:
            image = self.images_loaded[index]
            label = self.labels_loaded[index]
        else:
            image = path_to_image(self.image_paths[index], size=(self.config.img_size, self.config.img_size), color_type='rgb')
            label = path_to_image(self.label_paths[index], size=(self.config.img_size, self.config.img_size), color_type='gray')
        if self.config.enable_ref_text:
            text_features = self.text_embed_loaded[index]['text_features']
            text_attention_mask = self.text_embed_loaded[index]['text_attention_mask']
            if index in self.unlabeled_indices or (self.is_train and self.config.enable_fixtext):
                if self.unlabeled_reference_text is not None:
                    text_features = self.unlabeled_reference_text['text_features']
                    text_attention_mask = self.unlabeled_reference_text['text_attention_mask']
                    # print("!!!! DEBUG !!!! [text feature has been replaced] !!!! DEBUG !!!!")
        # loading image and label
        if self.is_train:
            image, label = preproc(image, label, preproc_methods=self.config.preproc_methods)

        image, label = self.transform_image(image), self.transform_label(label)

        #! Preventing the semi-supervised learning from using the labeled data
        # if index in self.unlabeled_indices:
        #     label = 0
        
        hash_label_path = self.label_paths[index].split('/')[-1].split('.')[0]
        
        if self.is_train:
            if self.config.enable_ref_text:
                return image, label, text_features, text_attention_mask, hash_label_path, index
            else:
                return image, label, hash_label_path, index
        else:
            if self.config.enable_ref_text:
                return image, label, text_features, text_attention_mask, self.label_paths[index], hash_label_path
            else:
                return image, label, self.label_paths[index], hash_label_path
    
    def set_unlabeled_data(self, unlabeled_indices):
        self.unlabeled_indices = unlabeled_indices
    
    def unset_unlabled_data(self):
        self.unlabeled_indices = []
    
    def set_unlabeled_ref_text(self, sentense):
        self.prepare_text_encoder()
        self.unlabeled_reference_text = self.text_encode(sentense)
        logger.freeze_info("[o] The reference text of all unlabled data has been set to {}".format(sentense))
        self.unload_model()
        
    def load_dataset_captions(self, json_file, using_cache: False, cache_dir='./data/cache/dataset', prefix=None, replace_all_sentense: str=None):
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        cache_dir = os.path.join(cache_dir, prefix + '_' + json_file.split('/')[-1].split('.')[0]) + '.pickle'
        if using_cache is True:# and not self.config.active_learning:
            logger.success_info("Parameter using_cache is set to True.")
            if os.path.exists(cache_dir):
                logger.success_info(f"Cache is available! Loading dataset from cache: {cache_dir}")
                annotation_list = pickle.load(open(cache_dir, 'rb'))
                return annotation_list
            else:
                logger.warn_info(f"Cache is not available! Running normal process...")
                
        self.prepare_text_encoder()
        if replace_all_sentense is not None:
            logger.warn_info("[!] Warning: replace_all_sentense is set to a fixed sentence: {}.".format(replace_all_sentense))
        anno_list = json.load(open(json_file, 'r'))
        enc_anno_dict = dict()
        for each_item in tqdm(anno_list):
            if "NonCAM" in each_item['image_filename']:
                continue
            enc_anno_dict[each_item['image_filename']] = self.text_encode(each_item['summary'] if replace_all_sentense is None else replace_all_sentense)
            # enc_anno_dict[each_item['image_filename']]['text_attention_mask'] = torch.zeros((1,64)).bool()
            if replace_all_sentense is not None:
                logger.info('[*] Caption "{}" has been replaced to "{}"'.format(each_item['summary'], replace_all_sentense))
        logger.success_info(f"Dataset captions loaded successfully.")
        logger.key_info("Saving datset to cahce: {}".format(cache_dir))
        with open(cache_dir, 'wb') as f:
            pickle.dump(enc_anno_dict, f)
        self.unload_model()
        return enc_anno_dict
        
    def prepare_text_encoder(self):
        if self.config.text_encoders == "roberta-base":
            from transformers import RobertaModel, RobertaTokenizerFast
            self.text_encoder = RobertaModel.from_pretrained("openai/roberta-base-uncased")
            self.tokenizer = RobertaTokenizerFast.from_pretrained("openai/roberta-base-uncased")
            logger.info("Using roberta-base-uncased as text encoder")
        elif self.config.text_encoders == "bert-base":
            from transformers import BertModel, BertTokenizerFast
            self.text_encoder = BertModel.from_pretrained("openai/bert-base-uncased")
            self.tokenizer = BertTokenizerFast.from_pretrained("openai/bert-base-uncased")
            logger.info("Using bert-base-uncased as text encoder")
        elif self.config.text_encoders == 'bert-large':
            from transformers import BertModel, BertTokenizerFast
            self.text_encoder = BertModel.from_pretrained("openai/bert-large-uncased")
            self.tokenizer = BertTokenizerFast.from_pretrained("openai/bert-large-uncased")
            logger.info("Using bert-large-uncased as text encoder")
        elif self.config.text_encoders == 'clip-base':
            from transformers import CLIPTextModel, CLIPTokenizerFast
            self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
            self.tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
            logger.info("Using clip-vit-base-patch32 as text encoder")
        elif self.config.text_encoders == 'clip-large':
            from transformers import CLIPTextModel, CLIPTokenizerFast
            self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14-336")
            self.tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-large-patch14-336")
            logger.info("Using clip-vit-large-patch14-336 as text encoder")
        else:
            raise NotImplementedError(f"Text encoder {self.config.text_encoders} not supported.")
        assert self.text_encoder is not None, f"Text encoder {self.config.text_encoders} not found."
        assert self.tokenizer is not None, f"Tokenizer for text encoder {self.config.text_encoders} not found."
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)
        self.text_encoder = self.text_encoder.to('cuda')
        logger.success_info("Text encoder and tokenizer loaded successfully.")
    
    def text_encode(self, text: str):
        tokenized = self.tokenizer(text, padding="max_length", max_length=self.config.ref_max_len, truncation=True, return_tensors="pt").to('cuda')
        encoded_text = self.text_encoder(**tokenized)
        text_attention_mask = tokenized['attention_mask'].ne(1).bool()
        
        if self.config.active_learning:
            text_feat_dict = {
                # 'encoded_text': encoded_text,
                'text_features_pooler': encoded_text.pooler_output.cpu(),
                'text_features': encoded_text.last_hidden_state.cpu(),
                'text_attention_mask': text_attention_mask.cpu(),
            }
        else:
            text_features = encoded_text.last_hidden_state
            text_feat_dict = {
                # 'encoded_text': encoded_text,
                'text_features': text_features.cpu(),
                'text_attention_mask': text_attention_mask.cpu(),
            }
        
        return text_feat_dict
    def prepare_labeled_text_features(self):
        if self.config.active_learning and self.unlabeled_indices != None:
            labeled_indices = np.setdiff1d(np.arange(len(self.image_paths)), self.unlabeled_indices).tolist()
            labeled_text_features = [self.text_embed_loaded[idx]['text_features_pooler'] for idx in labeled_indices]
            labeled_text_features = torch.cat(labeled_text_features, dim=0)
            return labeled_text_features
        else:
            return None
    def unload_model(self):
        del(self.text_encoder)
        del(self.tokenizer)
        
    def __len__(self):
        return len(self.image_paths)

def prepare_dataloader(dataset: data.Dataset, batch_size: int, num_workers: int, to_be_distributed: bool=False, is_train: bool=False, labeled_indices: list=None, is_unsup=False) -> data.DataLoader:
    if labeled_indices:
        indices = [dataset.image_to_idx[file_name] for file_name in labeled_indices]
        if is_unsup:
            indices = np.setdiff1d(np.arange(len(dataset)), indices).tolist()
            sample_number = len(labeled_indices)
        else:
            sample_number = len(dataset) - len(labeled_indices)
    else:
        indices = None
    if indices and len(indices) < sample_number and is_train:
        indices = indices * ((sample_number + len(indices) - 1) // len(indices))
        indices = random.sample(indices, sample_number)
    if indices is not None and is_unsup:
        dataset.set_unlabeled_data(indices)
    if to_be_distributed:
        if indices is None:
            sampler = data.DistributedSampler(dataset)
        else:
            dataset = data.Subset(dataset, indices)
            logger.key_info("[+] Subset of dataset has been created, length: {}".format(len(dataset)))
            sampler = data.DistributedSampler(dataset)
        return data.DataLoader(
            dataset=dataset, batch_size=batch_size, num_workers=min(num_workers, batch_size), pin_memory=True,
            shuffle=False, sampler=sampler, drop_last=True
        )
    else:
        if indices is None or len(indices) == 0 :
            sampler = data.RandomSampler(dataset)
        else:
            sampler = data.SubsetRandomSampler(indices)
        return data.DataLoader(
            dataset = dataset, batch_size = batch_size, num_workers=min(num_workers, batch_size, 0), pin_memory=True,
            sampler = sampler, drop_last=True
        )

def init_trainloader(config):
    train_loader = prepare_dataloader(
        MyData(config=config, datasets=config.training_set, image_size=config.img_size, is_train=True, text_replace=config.text_replace),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        to_be_distributed=config.distributed_train,
        is_train=True 
    )
    logger.success_info("{} batches of train dataloader {} has been created.".format(len(train_loader), config.training_set))
    return train_loader

def init_testloaders(config, use_text_replace=False):
    test_loaders = {}
    for testset in config.testing_sets.strip('+').split('+'):
        _data_loader_test = prepare_dataloader(
            MyData(config=config, datasets=testset, image_size=config.img_size, is_train=False, text_replace=config.text_replace, use_text_replace=config.use_text_replace or use_text_replace),
            num_workers=config.num_workers,
            to_be_distributed=False,
            batch_size=config.batch_size_valid, is_train=False
        )
        logger.success_info("{} batches of test dataloader {} has been created.".format(len(_data_loader_test), testset))
        test_loaders[testset] = _data_loader_test
    return test_loaders
