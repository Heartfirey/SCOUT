
import os
import torch
from transformers import CLIPTextModel, CLIPTokenizerFast


text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14-336")
tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-large-patch14-336")

def text_encode(text: str):
        tokenized = tokenizer(text, padding="max_length", max_length=64, truncation=True, return_tensors="pt")
        encoded_text = text_encoder(**tokenized)
        text_attention_mask = tokenized['attention_mask'].ne(1).bool()
        text_features = encoded_text.last_hidden_state
        
        return text_features[0][1]
def make_codebook(indices):
    category = []
    for name in indices:
        if 'COD10K' in name:
            category.append(name.split('-')[-2])
    category = set(category)
    print(category)
    weight = []

    for x in category:
        weight.append(text_encode(x))
    category2idx = dict()
    for i,name in enumerate(category):
        category2idx[name]=i
    while(len(weight) < 64):
        weight.append(text_encode(' ')) 
    weight = torch.stack(weight, dim=0)
    saved_data ={
        'weight':weight,
        'category':category2idx
    }  
    return saved_data
