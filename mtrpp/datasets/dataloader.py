import os
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import numpy as np
from mtrpp.datasets.msd import MSD
from mtrpp.datasets.audioset import Audioset
from mtrpp.datasets.music4all import Music4all
from mtrpp.datasets.fma import FMA
from mtrpp.datasets.music_caps import MusicCaps
from mtrpp.datasets.sampler import Sampler

def load_train_dataset(args):    
    if args.train_data == "all":
        dataset_builder = [MSD, Audioset, Music4all, FMA, MusicCaps]
        dataset = Sampler([d(
                data_dir=args.data_dir,
                split="train",
                caption_type= args.caption_type,
                sr=args.sr,
                duration=args.duration) 
            for d in dataset_builder])
    elif args.train_data == "msd":
        dataset_builder = MSD
        dataset = dataset_builder(
            data_dir=args.data_dir,
            split="train",
            caption_type= args.caption_type,
            sr=args.sr,
            duration=args.duration,
        )
    return dataset

    