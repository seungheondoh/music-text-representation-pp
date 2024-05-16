import os
import json
import random
import pickle
import numpy as np
import pandas as pd
import torch
from typing import Callable, List, Dict, Any
from torch.utils.data import Dataset
from mtrpp.utils.audio_utils import int16_to_float32, float32_to_int16, load_audio, STR_CH_FIRST

class GTZAN_Dataset(Dataset):
    def __init__(self, data_path, split, sr, duration, num_chunks):
        self.data_path = data_path
        self.split = split
        self.sr = sr
        self.input_length = int(sr * duration)
        self.num_chunks = num_chunks
        self.get_split()
        self.get_file_list()
    
    def get_split(self):
        track_split = json.load(open(os.path.join(self.data_path, "gtzan", "track_split.json"), "r"))
        self.train_track = [_id.split("/")[1] for _id in track_split['train_track']] #_id: blues/blues.00029.wav
        self.valid_track = [_id.split("/")[1] for _id in track_split['valid_track']]
        self.test_track = [_id.split("/")[1] for _id in track_split['test_track']]
    
    def get_file_list(self):
        annotation = json.load(open(os.path.join(self.data_path, "gtzan", "annotation.json"), 'r'))
        self.list_of_label = json.load(open(os.path.join(self.data_path, "gtzan", "gtzan_tags.json"), 'r'))
        self.tag_to_idx = {i:idx for idx, i in enumerate(self.list_of_label)}
        if self.split == "TRAIN":
            self.fl = [annotation[str(i)] for i in self.train_track]
        elif self.split == "VALID":
            self.fl = [annotation[str(i)] for i in self.valid_track]
        elif self.split == "TEST":
            self.fl = [annotation[str(i)] for i in self.test_track]
        elif self.split == "ALL":
            self.fl = list(annotation.values())
        else:
            raise ValueError(f"Unexpected split name: {self.split}")
        del annotation

    def audio_load(self, path):
        audio_path = os.path.join(self.data_path, "gtzan", "audio", path)
        audio, _ = load_audio(
            path=audio_path,
            ch_format= STR_CH_FIRST,
            sample_rate= self.sr,
            downmix_to_mono= True)
        if len(audio.shape) == 2:
            audio = audio.squeeze(0)
        audio = int16_to_float32(float32_to_int16(audio.astype('float32'))) # for float32 loader
        hop = (audio.shape[-1] - self.input_length) // self.num_chunks
        audio = np.stack([np.array(audio[i * hop : i * hop + self.input_length]) for i in range(self.num_chunks)]).astype('float32')
        return audio

    def tag_to_binary(self, text):
        bainry = np.zeros([len(self.list_of_label),], dtype=np.float32)
        if isinstance(text, str):
            bainry[self.tag_to_idx[text]] = 1.0
        elif isinstance(text, list):
            for tag in text:
                bainry[self.tag_to_idx[tag]] = 1.0
        return bainry

    def __getitem__(self, index):
        item = self.fl[index]
        tag_list = item['tag']
        binary = self.tag_to_binary(tag_list)
        text = ", ".join(tag_list)
        tags = self.list_of_label
        track_id = item['track_id']
        path = f"{tag_list}/{track_id}"
        audio = self.audio_load(path)
        return {
            "audio":audio, 
            "track_id":track_id, 
            "tags":tags, 
            "binary":binary, 
            "text":text
            }

    def __len__(self):
        return len(self.fl)
