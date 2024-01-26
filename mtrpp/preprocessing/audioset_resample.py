import os
import random
from datasets import load_dataset
from contextlib import contextmanager
import multiprocessing

import json
import numpy as np
from audio_utils import load_audio, STR_CH_FIRST, float32_to_int16
from tqdm import tqdm

# hard coding hpamras
DATASET_PATH = "../../../dataset/audioset/"
MUSIC_SAMPLE_RATE = 22050
DURATION = 10
DATA_LENGTH = int(MUSIC_SAMPLE_RATE * DURATION)

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()
    
def as_resampler(sample):
    path = sample['path']
    fname = sample['fname']
    split = sample['split']
    audio_path = os.path.join(DATASET_PATH, path)
    save_path = os.path.join(DATASET_PATH, "npy", split, fname + ".npy")
    try:
        src, _ = load_audio(
            path=audio_path,
            ch_format= STR_CH_FIRST,
            sample_rate= MUSIC_SAMPLE_RATE,
            downmix_to_mono= True)
        if src.shape[-1] < DATA_LENGTH: # short case
            pad = np.zeros(DATA_LENGTH)
            pad[:src.shape[-1]] = src
            src = pad
        elif src.shape[-1] > DATA_LENGTH: # too long case
            src = src[:DATA_LENGTH]
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        src_int16 = float32_to_int16(src.astype(np.float32))
        np.save(save_path, src_int16)

        fname_ext = fname + ".wav"
        tar = os.path.join(DATASET_PATH, "music", split, fname_ext)
        if not os.path.exists(os.path.dirname(tar)):
            os.makedirs(os.path.dirname(tar), exist_ok=True)
        os.system(f"mv {audio_path} {tar}")
    except:
        error_path = os.path.join(DATASET_PATH, f"blacklist/{fname}.npy")
        if not os.path.exists(os.path.dirname(error_path)):
            os.makedirs(os.path.dirname(error_path), exist_ok=True)
        np.save(error_path, fname)
    
def main():
    as_dataset = load_dataset("seungheondoh/audioset-music")
    audioset_path = json.load(open(os.path.join(DATASET_PATH, "metadata", "audioset_path.json"), 'r'))
    blacklist = [i.replace(".npy", "") for i in os.listdir(os.path.join(DATASET_PATH, "blacklist"))]
    annotation = []
    for split in ["unbalanced_train", "balanced_train", "eval"]:
        if split == "unbalanced_train":
            path = audioset_path["unbalance"]
        elif split == "balanced_train":
            path = audioset_path["balance"]
        elif split == "eval":
            path = audioset_path["eval"]
        for i in as_dataset[split]: 
            if i["is_crawl"] and (i["ytid"] not in blacklist):
                path_info = os.path.join(f"{split}_segments", path[i["ytid"]])
                if os.path.isfile(os.path.join(DATASET_PATH, path_info)):
                    annotation.append({"fname": i["ytid"], "path": path_info, "split": split})
    print("start mp")
    print(len(annotation))
    with poolcontext(processes=multiprocessing.cpu_count()) as pool:
        pool.map(as_resampler, annotation)

if __name__ == '__main__':
    main()