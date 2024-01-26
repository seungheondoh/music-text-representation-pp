import os
import random
import numpy as np
import pandas as pd
import torch
import json
import jsonlines
import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset
from mtrpp.utils.audio_utils import float32_to_int16, int16_to_float32, load_audio, STR_CH_FIRST

class OLGA_Inference(Dataset):
    def __init__(self, data_dir, split, caption_type, audio_loader="ffmpeg", sr=22050, duration=10, audio_enc=".npy"):
        self.data_dir = os.path.join(data_dir, "olga-msd")
        self.audio_dir = os.path.join(data_dir, "msd")
        self.split = split
        self.audio_loader = audio_loader
        self.audio_enc = audio_enc
        self.caption_type =caption_type
        self.sr = sr
        self.n_samples = int(sr * duration)
        self.track_meta = json.load(open(os.path.join(self.data_dir, "track_meta.json"), 'r'))
        self.fnames = list(self.track_meta.keys())
    
    def load_audio(self, fname):
        audio_path = self.track_meta[fname]["path"]
        if self.audio_enc == ".npy": # for fast audio loading
            audio_path = os.path.join(self.audio_dir,'npy', audio_path.replace(".mp3", ".npy"))
            audio = np.load(audio_path, mmap_mode='r')
            audio = int16_to_float32(audio)
        else:
            audio_path = os.path.join(self.audio_dir,'songs', audio_path)
            audio, _ = load_audio(
                path=audio_path,
                ch_format= STR_CH_FIRST,
                resample_by = self.audio_loader,
                sample_rate= self.sr,
                downmix_to_mono= True
            )
            audio = int16_to_float32(float32_to_int16(audio.astype('float32')))
        if len(audio.shape) == 2:
            audio = audio.squeeze(0)
        input_size = int(30 * self.sr) # max 30 sec
        if audio.shape[-1] < input_size:
            pad = np.zeros(input_size)
            pad[:audio.shape[-1]] = audio
            audio = pad
        else:
            audio = audio[:input_size]
        # chunkfy audio
        ceil = int(audio.shape[-1] // self.n_samples)
        audio_tensor = torch.from_numpy(np.stack(np.split(audio[:ceil * self.n_samples], ceil)).astype('float32'))
        return audio_tensor

    def __getitem__(self, index):
        fname = self.fnames[index]
        audio_tensor = self.load_audio(fname)
        return fname, audio_tensor

    def __len__(self):
        return len(self.fnames)