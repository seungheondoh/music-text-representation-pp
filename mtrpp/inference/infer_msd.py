import os
import random
import torch
import pickle
import numpy as np
from tqdm import tqdm
from glob import glob
from torch.utils.data import Dataset
from mtrpp.utils.audio_utils import int16_to_float32, float32_to_int16, load_audio
from mtrpp.utils.eval_utils import load_ttmr_pp
from datasets import load_dataset

class AUDIO_DATASET(Dataset):
    def __init__(self, data_dir, _src_ext_audio=".npy", sr=22050, duration=10):
        MSD_id_to_7D_id = pickle.load(open(os.path.join(data_dir, "MSD_id_to_7D_id.pkl"), 'rb'))
        id_to_path = pickle.load(open(os.path.join(data_dir, "7D_id_to_path.pkl"), 'rb'))
        self.path_to_id = {id_to_path[v]:k for k,v in MSD_id_to_7D_id.items() if v in id_to_path}
        self.data_dir = os.path.join(data_dir, "npy")
        self._src_ext_audio = _src_ext_audio
        self.sr = sr        
        self.n_samples = int(sr * duration)
        self.fl = glob(
            os.path.join(self.data_dir, "**", "*{}".format(self._src_ext_audio)),
            recursive=True,
        )
        
    def load_npy(self, audio_path):
        audio = np.load(audio_path, mmap_mode='r')
        audio = int16_to_float32(audio) # for float32 loader
        if len(audio.shape) == 2:
            audio = audio.squeeze(0)
        input_size = int(self.n_samples * 3) + 1 # 30sec
        if audio.shape[-1] < input_size:
            pad = np.zeros(input_size)
            pad[:audio.shape[-1]] = audio
            audio = pad
        if input_size < audio.shape[-1]:
            audio = audio[:input_size]
        ceil = int(audio.shape[-1] // self.n_samples)
        audio_tensor = torch.from_numpy(np.stack(np.split(audio[:ceil * self.n_samples], ceil)).astype('float32'))
        return audio_tensor

    def __getitem__(self, index):
        audio_path = self.fl[index]
        track_id = self.path_to_id[audio_path.replace(self.data_dir + "/", "").replace(".npy", ".mp3")]
        audio_tensor = self.load_npy(audio_path)
        return track_id, audio_tensor

    def __len__(self):
        return len(self.fl)

def main(): 
    save_dir = f"exp/ttmrpp/meta_tag_caption_sim"
    model, sr, duration = load_ttmr_pp(save_dir)
    model = model.to("cuda:0")
    model.eval()
    
    dataset = AUDIO_DATASET(data_dir="/data/seungheon/dataset/msd")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=256, shuffle=False,
        num_workers=8, pin_memory=True, drop_last=False
        )
    track_embs = {}
    for i in tqdm(dataloader):
        fname, audio_tensor = i
        B, C, T= audio_tensor.size()
        batch_audio = audio_tensor.view(-1, T)
        with torch.no_grad():
            audio_embs = model.audio_forward(batch_audio.to("cuda:0"))
        unbatch_audio = audio_embs.view(B,C,-1)
        audio_embs = unbatch_audio.mean(1, False).detach().cpu()
        for name, embs in zip(fname, audio_embs):
            track_embs[name] = embs.numpy()
    os.makedirs(os.path.join(save_dir, "embs", "msd"), exist_ok=True)
    torch.save(track_embs, os.path.join(save_dir, "embs", "msd", "track_embs.pt"))
    
if __name__ == "__main__":
    main()  