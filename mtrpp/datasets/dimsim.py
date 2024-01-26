import os
import random
import numpy as np
import pandas as pd
import torch
import json
from datasets import load_dataset
from torch.utils.data import Dataset
from mtrpp.utils.audio_utils import int16_to_float32, float32_to_int16, load_audio, STR_CH_FIRST

class Dimsim(Dataset):
    def __init__(self, data_dir, split, caption_type, audio_loader="ffmpeg", sr=22050, duration=3, audio_enc=".mp3"):
        self.data_dir = os.path.join(data_dir, "dim-sim")
        self.split = split
        self.audio_loader = audio_loader
        self.audio_enc = audio_enc
        self.caption_type =caption_type
        self.sr = sr
        self.n_samples = int(sr * duration)
        self.annotations = json.load(open(os.path.join(self.data_dir, "track_meta.json"), 'r'))
        self.dimsim_gt = pd.read_csv(os.path.join(self.data_dir,"clean-dim-sim.csv"))

    def load_text(self, item):
        track = item['title']
        album = item["release"]
        artist = item["artist_name"]
        text = f"similar music track with {track} by {artist} from {album}"
        return text
    
    def load_audio(self, track_id, start_sec):
        audio_path = os.path.join(self.data_dir, "audio", track_id + self.audio_enc)
        audio, _ = load_audio(
            path=audio_path,
            ch_format= STR_CH_FIRST,
            resample_by = self.audio_loader,
            sample_rate= self.sr,
            downmix_to_mono= True
        )
        if len(audio.shape) == 2:
            audio = audio.squeeze(0)
        input_size = int(self.n_samples)
        start_sample = int(float(start_sec) * self.sr)
        end_sample = start_sample + int(3 * self.sr) # dim-sim always use 3sec
        audio = int16_to_float32(float32_to_int16(audio[start_sample : end_sample])) 
        audio_tensor = torch.from_numpy(audio.astype('float32'))
        return audio_tensor

    def load_triplet(self, triplet_item, position):
        track_id = triplet_item[f"{position}_id"]
        start_sec = triplet_item[f"{position}_start_seconds"]
        anno_item = self.annotations[f"{track_id}_{position}_{start_sec}"]
        text = self.load_text(anno_item)
        return track_id, text, start_sec

    def __getitem__(self, index):
        triplet_item = self.dimsim_gt.iloc[index]
        track_id_0, text_0, start_sec_0 = self.load_triplet(triplet_item, position="anchor")
        track_id_1, text_1, start_sec_1 = self.load_triplet(triplet_item, position="song1")
        track_id_2, text_2, start_sec_2 = self.load_triplet(triplet_item, position="song2")
        audio_0 = self.load_audio(track_id_0, start_sec_0)
        audio_1 = self.load_audio(track_id_1, start_sec_1)
        audio_2 = self.load_audio(track_id_2, start_sec_2)
        return {
            "tid_anc": track_id_0, 
            "tid_song1": track_id_1, 
            "tid_song2": track_id_2, 
            "audio_anc": audio_0, 
            "audio_song1": audio_1,
            "audio_song2": audio_2, 
            "text_anc": text_0,
            "text_song1": text_1,
            "text_song2": text_2,
            }

    def __len__(self):
        return len(self.fl)