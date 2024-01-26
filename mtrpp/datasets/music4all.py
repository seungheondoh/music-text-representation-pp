import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets
import jsonlines
from mtrpp.utils.audio_utils import int16_to_float32

class Music4all(Dataset):
    def __init__(self, data_dir, split, caption_type, sr=22050, duration=10, audio_enc="npy"):
        self.data_dir = os.path.join(data_dir, "music4all")
        self.split = split
        self.audio_enc = audio_enc
        self.n_samples = int(sr * duration)
        self.caption_type = caption_type
        self.blacklist_tag = {"music", "instrument"}
        self._load_dataset()
        self.prob = 0.25

    def _load_dataset(self):
        enrich_m4a = load_dataset("seungheondoh/enrich-music4all")
        self.dataset = enrich_m4a["train"]

    def load_tag(self, item):
        tag_list = item['tag_list']
        k = random.choice(range(1, len(tag_list)+1)) 
        sampled_tag_list = random.sample(tag_list, k)
        text = ", ".join(sampled_tag_list)
        return text.lower()

    def load_meta(self, item):
        track = item['title']
        album = item["release"]
        artist = item["artist_name"]
        text = f"music track {track} by {artist} from {album}"
        return text
    
    def load_caption(self, item):
        text = item['pseudo_caption']
        return text
    
    def load_text(self, item):
        text_pool = [] 
        if "meta" in self.caption_type:
            text_pool.append(self.load_meta(item))
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
        audio_path = os.path.join(self.data_dir,'npy', fname + ".npy")
        audio_tensor = self.load_audio(audio_path)
        return fname, text, audio_tensor

    def __len__(self):
        return len(self.dataset)

# dataset = Music4all(data_dir="../../../dataset", split="train", caption_type="meta_tag_caption")
# for idx, i in enumerate(dataset):
#     fname, text, audio_tensor = i
#     print(fname, text)
#     if idx == 30:
#         break