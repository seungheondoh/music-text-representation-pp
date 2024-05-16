import json
import os
import argparse
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import torch.backends.cudnn as cudnn
# backbones
from mtrpp.transfer.dataset_embs.data_manger import get_dataloader
from mtrpp.utils.transfer_utils import single_query_evaluation, get_evaluation
from sklearn import metrics

parser = argparse.ArgumentParser(description='PyTorch MSD Training')
parser.add_argument('--msu_dir', type=str, default="/hdd2/seungheon/msu-benchmark/dataset")
parser.add_argument('--framework', type=str, default="contrastive") # or transcription
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers')
# downstream options
parser.add_argument("--probe_type", default="zs", type=str)
parser.add_argument("--eval_dataset", default="fma", type=str)
args = parser.parse_args()

def main(args) -> None:
    save_dir = f"/data/seungheon/music-text-representation-pp/mtrpp/exp/ttmrpp/meta_tag_caption_sim/"
    embs_dir = f"{save_dir}/embs/{args.eval_dataset}"
    if args.eval_dataset in ["mtg_top50tags", "mtg_genre", "mtg_instrument", "mtg_moodtheme"]:
        embs_dir = f"{save_dir}/embs/mtg"
    folder_name = f"zeroshot"
    save_dir = os.path.join(save_dir, args.eval_dataset, folder_name) # update save_dir
    
    audio_embs = torch.load(os.path.join(embs_dir, 'audio_embs.pt'))
    tag_embs = torch.load(os.path.join(embs_dir, 'tag_embs.pt'))
    test_loader = get_dataloader(args=args, audio_embs=audio_embs, split="TEST")
    t_embs = [tag_embs[tag] for tag in test_loader.dataset.list_of_label]
    a_embs, groudturths = [], []
    for batch in tqdm(test_loader):
        a_embs.append(batch['audio'])
        groudturths.append(batch['binary'])
    a_embs = torch.stack(a_embs, dim=0)
    t_embs = torch.stack(t_embs, dim=0)
    targets = torch.stack(groudturths, dim=0)
    if len(a_embs.size()) == 3:
        a_embs = a_embs.squeeze(1)
    if len(t_embs.size()) == 3:
        t_embs = t_embs.squeeze(1)
    if len(targets.size()) == 3:
        targets = targets.squeeze(1)
    a_embs = nn.functional.normalize(a_embs, dim=-1)
    t_embs = nn.functional.normalize(t_embs, dim=-1)
    logits = a_embs @ t_embs.T
    if args.eval_dataset in ['fma', 'gtzan', 'emotify']:
        results = get_evaluation(targets.numpy(),logits.numpy(), test_loader.dataset.list_of_label, 'multiclass')
    else:
        results = get_evaluation(targets.numpy(),logits.numpy(),test_loader.dataset.list_of_label, 'multilabel')
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"zeroshot.json"), mode="w") as io:
        json.dump(results, io, indent=4)
    
if __name__ == "__main__":
    main(args)