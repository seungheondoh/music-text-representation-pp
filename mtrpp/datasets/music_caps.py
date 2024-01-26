import os
import random
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from mtrpp.utils.audio_utils import float32_to_int16, int16_to_float32, load_audio, STR_CH_FIRST

class MusicCaps(Dataset):
    def __init__(self, data_dir, split, caption_type, audio_loader="ffmpeg", sr=22050, duration=10, audio_enc=".npy"):
        self.dataset_name = "music_caps"
        self.data_dir = os.path.join(data_dir, "music_caps")
        self.split = split
        self.audio_loader = audio_loader
        self.audio_enc = audio_enc
        self.caption_type =caption_type
        self.sr = sr
        self.n_samples = int(sr * duration)
        self.dataset = load_dataset("seungheondoh/LP-MusicCaps-MC")
        self.get_split()
        self.get_columns()
        self.prob = 0.01
        
    def get_columns(self):
        self.id_col = "ytid"
        self.tag_col = "aspect_list"
        self.caption_col = "caption_ground_truth"
        # metadata
        self.title_col = None
        self.artist_col = None
        self.album_col = None
        self.year_col = None
        # mid-level data
        self.key_col = None
        self.tempo_col = None
        self.chord_col = None

    def get_split(self):
        self.fl = self.dataset[self.split]
        self.annotations = pd.DataFrame(self.fl)
        self.ytid_to_idx = {ytid:idx for idx, ytid in enumerate(self.annotations['ytid'])}

    def _load_audio(self, fname):
        if self.audio_enc == ".npy": # for fast audio loading
            audio_path = os.path.join(self.data_dir, "npy", fname + self.audio_enc)
            audio = np.load(audio_path, mmap_mode='r')
            audio = int16_to_float32(audio)
        else:
            audio_path = os.path.join(self.data_dir, "audio", fname + self.audio_enc)
            audio, _ = load_audio(
                path=audio_path,
                ch_format= STR_CH_FIRST,
                resample_by = self.audio_loader,
                sample_rate= self.sr,
                downmix_to_mono= True
            )
            audio = int16_to_float32(float32_to_int16(audio))
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

    def load_tag(self, item):
        tag_list = item['aspect_list']
        k = random.choice(range(1, len(tag_list)+1)) 
        sampled_tag_list = random.sample(tag_list, k)
        text = ", ".join(sampled_tag_list)
        return text
    
    def load_caption(self, item):
        return item['caption_ground_truth']
    
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

    def get_audio(self, ytid):
        idx = self.ytid_to_idx[ytid]
        item = self.annotations.iloc[idx]
        return  self._load_audio(item["fname"])

    def __getitem__(self, index):
        item = self.fl[index]
        fname = item['fname']
        text = self.load_text(item)
        audio_tensor = self._load_audio(fname)
        return fname, text, audio_tensor

    def __len__(self):
        return len(self.fl)


# dataset = MusicCaps(data_dir="../../../dataset", split="test", caption_type="tag")
# for idx, i in enumerate(dataset):
#     fname, text, audio_tensor = i
#     print(fname, text)
#     if idx == 30:
#         break