import os
import random
import torch
from datasets import load_dataset
from contextlib import contextmanager
import multiprocessing
import argparse
import json
import numpy as np
from audio_utils import load_audio, STR_CH_FIRST, float32_to_int16
from tqdm import tqdm

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()
    
def msd_resampler(sample):
    path = sample['path']
    fname = sample['track_id']
    data_dir = sample["data_dir"]
    sampling_rate = sample["sampling_rate"]
    target_legnth = sample['target_legnth']
    save_name = os.path.join(data_dir,'npy', path.replace(".mp3",".npy"))
    try:
        src, _ = load_audio(
            path=os.path.join(data_dir,'songs',path),
            ch_format= STR_CH_FIRST,
            sample_rate= sampling_rate,
            downmix_to_mono= True)
        if len(src.shape) == 2:
            src = src.squeeze(0)
        if src.shape[-1] < target_legnth: # short case
            print(src.shape)
            raise ValueError(f"{fname} is shorter than 10sec")
        max_amp = src.max()
        if (max_amp == 0.) and (len(src) == len(src[src == max_amp])): # max value is zero
            raise ValueError(f"{fname} max value is {max_amp}")
        if not os.path.exists(os.path.dirname(save_name)):
            os.makedirs(os.path.dirname(save_name))
        src_int16 = float32_to_int16(src.astype(np.float32))
        np.save(save_name, src_int16)
    except:
        os.makedirs("./error/msd", exist_ok=True)
        os.system(f"touch error/msd/{fname}")
    
def main(args):
    args.target_legnth = int(args.sampling_rate * args.duraiton)
    msd_pretrain = load_dataset("seungheondoh/enrich-msd", split="train")
    annotation = []
    for item_dict in msd_pretrain["train"]:
        item_dict["data_dir"] = os.path.join(args.data_dir, args.dataset)
        item_dict["target_legnth"] = args.target_legnth
        item_dict["sampling_rate"] = args.sampling_rate # duraiton
        annotation.append(item_dict)
    annotation = list(msd_pretrain["train"])
    print("start preprocessing")
    with poolcontext(processes=multiprocessing.cpu_count()) as pool:
        pool.map(msd_resampler, annotation)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='audio preprocessing')
    parser.add_argument('--data_dir', type=str, default="../../../dataset")
    parser.add_argument('--dataset', type=str, default="msd")
    parser.add_argument('--sampling_rate', type=int, default=22050)
    parser.add_argument('--duraiton', type=int, default=10)
    args = parser.parse_args()
    main(args)