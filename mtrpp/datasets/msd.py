import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from mtrpp.utils.audio_utils import int16_to_float32
import jsonlines

class MSD(Dataset):
    def __init__(self, data_dir, split, caption_type, sr=22050, duration=10, audio_enc="npy"):
        self.data_dir = os.path.join(data_dir, "msd")
        self.split = split
        self.audio_enc = audio_enc
        self.n_samples = int(sr * duration)
        self.caption_type = caption_type
        self.prob = 0.4
        self.max_artist = 5
        msd_pretrain = load_dataset("seungheondoh/enrich-msd", split="train")
        if caption_type in ["tag", "tag_caption"]:
            self.dataset = [i for i in msd_pretrain if len(i["tag_list"]) > 0]
        elif caption_type in ["meta_tag_caption"]:
            self.dataset = msd_pretrain
        elif caption_type in ["meta_tag_caption_sim"]:
            self.dataset = []
            artist_sim = load_dataset("seungheondoh/olga-track-to-artist", split="train")
            artist_dict = {i["track_id"]:i for i in artist_sim}
            for item in msd_pretrain:
                if item['track_id'] in artist_dict:
                    item['sim_artist_text'] = artist_dict[item['track_id']]['sim_artist_text']
                else:
                    item['sim_artist_text'] = ""
                self.dataset.append(item)

    def load_meta(self, item):
        track = item['title']
        album = item["release"]
        artist = item["artist_name"]
        text = f"music track {track} by {artist} from {album}"
        return text

    def load_caption(self, item):
        text = ""
        if item['pseudo_caption']:
            text += item['pseudo_caption']
        return text

    def load_tag(self, item):
        text = ""
        tag_list = item['tag_list']
        if len(tag_list) > 0:
            k = random.choice(range(1, len(tag_list)+1)) 
            sampled_tag_list = random.sample(tag_list, k)
            text = ", ".join(sampled_tag_list)
        return text
    
    def load_sim(self, item):
        text = ""
        sim_artist = item['sim_artist_text']
        if len(sim_artist) > 0:
            artist_list = sim_artist.split("[SEP]")
            if len(artist_list) > self.max_artist:
                k_pool = self.max_artist+1
            else:
                k_pool = len(artist_list)+1
            k = random.choice(range(1, k_pool)) 
            sampled_sim_artist = random.sample(artist_list, k)
            artist_text = ", ".join(sampled_sim_artist)
            text += f"similar with artist {artist_text}"
        return text
    
    def load_text(self, item):
        text_pool = []
        if "meta" in self.caption_type:
            text_pool.append(self.load_meta(item))
        if "sim" in self.caption_type:
            sim_text = self.load_sim(item)
            if sim_text:
                text_pool.append(sim_text)
        if "caption" in self.caption_type:
            cap_text = self.load_caption(item)
            if cap_text:
                text_pool.append(cap_text)
        if "tag" in self.caption_type:
            tag_text = self.load_tag(item)
            if tag_text:
                text_pool.append(tag_text)
        k = random.choice(range(1, len(text_pool)+1)) 
        sampled_text_pool = random.sample(text_pool, k)
        random.shuffle(sampled_text_pool)
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
        track_id = item['track_id']
        text = self.load_text(item)
        audio_path = os.path.join(self.data_dir,'npy', item['path'].replace(".mp3", ".npy"))
        audio_tensor = self.load_audio(audio_path)
        return track_id, text, audio_tensor

    def __len__(self):
        return len(self.dataset)