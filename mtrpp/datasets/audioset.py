import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets
import jsonlines
from mtrpp.utils.audio_utils import int16_to_float32

class Audioset(Dataset):
    def __init__(self, data_dir, split, caption_type, sr=22050, duration=10, audio_enc="npy"):
        self.data_dir = os.path.join(data_dir, "audioset")
        self.split = split
        self.audio_enc = audio_enc
        self.n_samples = int(sr * duration)
        self.caption_type = caption_type
        self.prob = 0.25
        self.blacklist_tag = {"music", "instrument"}
        self._load_dataset()

    def _load_dataset(self):
        enrich_audioset = load_dataset("seungheondoh/enrich-audioset-music")
        balanced_train = self.load_jsonlines(data= enrich_audioset, subset="balanced_train")
        unbalanced_train = self.load_jsonlines(data= enrich_audioset, subset="unbalanced_train")
        self.dataset = balanced_train + unbalanced_train

    def load_jsonlines(self, data, subset):
        datas = []
        for line in data[subset]:
            if line['path']:
                datas.append(line)
        return datas

    def load_tag(self, item):
        tag_list = [tag for tag in item['tag_list'] if tag.lower() not in self.blacklist_tag]
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
        audio_path = os.path.join(self.data_dir,'npy', item['path'] + ".npy")
        audio_tensor = self.load_audio(audio_path)
        return fname, text, audio_tensor

    def __len__(self):
        return len(self.dataset)