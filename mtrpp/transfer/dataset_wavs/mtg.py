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

class MTG_Dataset(Dataset):
    def __init__(self, data_path, split, sr, duration, num_chunks, eval_dataset="autotagging_top50tags"):
        self.data_path = data_path
        self.split = split
        self.sr = sr
        self.input_length = int(sr * duration)
        self.num_chunks = num_chunks
        if "top50tags" in eval_dataset:
            self.mtg_split = "autotagging_top50tags"
        elif "genre" in eval_dataset:
            self.mtg_split = "autotagging_genre"
        elif "moodtheme" in eval_dataset:
            self.mtg_split = "autotagging_moodtheme"
        elif "instrument" in eval_dataset:
            self.mtg_split = "autotagging_instrument"
        
        if self.split != "ALL":
            self.get_split()
        self.get_file_list()
    
    def get_split(self):
        track_split = json.load(open(os.path.join(self.data_path, "mtg", f"{self.mtg_split}_track_split.json"), "r"))
        self.train_track = track_split['train_track']
        self.valid_track = track_split['valid_track']
        self.test_track = track_split['test_track']
    
    def get_file_list(self):
        if self.split == "ALL":
            annotation = json.load(open(os.path.join(self.data_path, "mtg", "annotation.json"), 'r'))
            self.list_of_label = json.load(open(os.path.join(self.data_path, "mtg", f"mtg_tags.json"), 'r'))
        else:
            annotation = json.load(open(os.path.join(self.data_path, "mtg", f"{self.mtg_split}_annotation.json"), 'r'))
            self.list_of_label = json.load(open(os.path.join(self.data_path, "mtg", f"mtg_{self.mtg_split}_tags.json"), 'r'))
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

    def audio_load(self, track_id):
        audio_path = os.path.join(self.data_path, "mtg", "audio", track_id)
        audio, _ = load_audio(
            path=audio_path,
            ch_format= STR_CH_FIRST,
            sample_rate= self.sr,
            downmix_to_mono= True)
        if len(audio.shape) == 2:
            audio = audio.squeeze(0)
        audio = int16_to_float32(float32_to_int16(audio.astype('float32'))) # for float32 loader
        hop = (audio.shape[-1] - self.input_length) // self.num_chunks
        audio = np.stack([np.array(audio[(i * hop) : (i * hop) + self.input_length]) for i in range(self.num_chunks)]).astype('float32')
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
        path = item['path']
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
