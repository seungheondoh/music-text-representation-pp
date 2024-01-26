import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets
import jsonlines
from mtrpp.utils.audio_utils import int16_to_float32

class FMA(Dataset):
    def __init__(self, data_dir, split, caption_type, sr=22050, duration=10, audio_enc="npy"):
        self.data_dir = os.path.join(data_dir, "fma_large")
        self.split = split
        self.audio_enc = audio_enc
        self.n_samples = int(sr * duration)
        self.caption_type = caption_type
        self.blacklist_tag = {"music", "instrument"}
        self._load_dataset()
        self.prob = 0.09

    def _load_dataset(self):
        enrich_fma = load_dataset("seungheondoh/enrich-fma-large")
        fma_train = enrich_fma["train"]
        fma_valid = enrich_fma["valid"]
        self.dataset = concatenate_datasets([fma_train, fma_valid])
        
    def load_tag(self, item):
        tag_list = item['tag_list']
        k = random.choice(range(1, len(tag_list)+1)) 
        sampled_tag_list = random.sample(tag_list, k)
        text = ", ".join(sampled_tag_list)
        return text.lower()
    
    def load_caption(self, item):
        text = item['pseudo_caption']
        return text
    
    def load_text(self, item):
        text_pool = []
        if "tag" in self.caption_type:
            text_pool.append(self.load_tag(item))
        if "caption" in self.caption_type:
            text_pool.append(self.load_caption(item))
        
        random.shuffle(text_pool)
        k = random.choice(range(1, len(text_pool)+1)) 
        sampled_text_pool = random.sample(text_pool, k)
        text = ". ".join(sampled_text_pool)
        return text


    def get_audio_path(self, track_id):
        tid_str = '{:06d}'.format(int(track_id))
        return os.path.join(tid_str[:3], tid_str + '.npy')

    def load_audio(self, audio_path):
        audio = np.load(audio_path, mmap_mode='r')
        audio = int16_to_float32(audio) # for flaot32 loader
        if len(audio.shape) == 2:
            audio = audio.squeeze(0)
        input_size = int(self.n_samples)
        if audio.shape[-1] < input_size:
            pad = np.zeros(input_size)
            pad[:audio.shape[-1]] = audio
            audio = pad
        random_idx = random.randint(0, audio.shape[-1]-self.n_samples)
        audio_tensor = torch.from_numpy(np.array(audio[random_idx:random_idx+self.n_samples]).astype('float32'))
        return audio_tensor

    def __getitem__(self, index):
        item = self.dataset[index]
        fname = item['track_id']
        text = self.load_text(item)
        path = self.get_audio_path(fname)
        audio_path = os.path.join(self.data_dir,'npy', path)
        audio_tensor = self.load_audio(audio_path)
        return fname, text, audio_tensor

    def __len__(self):
        return len(self.dataset)

# dataset = FMA(data_dir="../../../dataset", split="train", caption_type="tag_caption")
# for idx, i in enumerate(dataset):
#     fname, text, audio_tensor = i
#     print(fname, text)
#     if idx == 30:
#         break