import os
import json
import random
import pickle
import pandas as pd
import numpy as np
import torch
from re import sub
from typing import Callable, List, Dict, Any
from datasets import load_dataset
from torch.utils.data import Dataset
from mtrpp.preprocessing.audio_utils import float32_to_int16, int16_to_float32, load_audio, STR_CH_FIRST

class SongDescriber(Dataset):
    def __init__(self, data_dir, split, caption_type, audio_loader="ffmpeg", sr=22050, duration=10, audio_enc=".mp3"):
        self.dataset_name = "song_describer"
        self.data_dir = os.path.join(data_dir, "song_describer")
        self.split = split
        self.audio_loader = audio_loader
        self.audio_enc = audio_enc
        self.caption_type = caption_type
        self.sr = sr
        self.n_samples = int(sr * duration)
        self.dataset = load_dataset("music-temp/song-describer-dataset")
        self.get_split()
        self.get_columns()

    def get_split(self):
        dataset = self.dataset["train"]
        df_anno = pd.DataFrame(dataset)
        self.annotations = df_anno[df_anno["is_valid_subset"]]
        self.tid_to_path = {track_id:path for track_id, path in zip(self.annotations["track_id"], self.annotations["path"])}
        self.tid_to_idx = {tid:idx for idx, tid in enumerate(self.annotations['track_id'])}

    def get_columns(self):
        self.id_col = "track_id"
        self.tag_col = "aspect_list"
        self.caption_col = "caption"
        # metadata
        self.title_col = None
        self.artist_col = "artist_id"
        self.album_col = "album_id"
        self.year_col = None
        # mid-level data
        self.key_col = None
        self.tempo_col = None
        self.chord_col = None

    def _load_audio(self, fname):
        audio_path = self.tid_to_path[fname]
        audio_path = audio_path.removesuffix(".mp3") + ".2min.mp3"
        if self.audio_enc == ".npy": # for fast audio loading
            audio_path = os.path.join(self.data_dir, "npy", audio_path.replace(".mp3", self.audio_enc))
            audio = np.load(audio_path, mmap_mode='r')
            audio = int16_to_float32(audio.astype('float32'))
        else:
            audio_path = os.path.join(self.data_dir, "audio", audio_path)
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
        input_size = int(self.n_samples)
        if audio.shape[-1] < input_size:
            pad = np.zeros(input_size)
            pad[:audio.shape[-1]] = audio
            audio = pad
        ceil = int(audio.shape[-1] // self.n_samples)
        audio_tensor = torch.from_numpy(np.stack(np.split(audio[:ceil * self.n_samples], ceil)).astype('float32'))
        return audio_tensor

    def get_audio(self, tid):
        idx = self.tid_to_idx[tid]
        item = self.annotations.iloc[idx]
        return self._load_audio(item["track_id"])

    def __getitem__(self, index):
        item = self.annotations.iloc[index]
        fname = item['track_id']
        text = item['caption']
        audio_tensor = self._load_audio(fname)
        return fname, text, audio_tensor

    def __len__(self):
        return len(self.annotations)