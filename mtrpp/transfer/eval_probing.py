import os
import json
import math
import argparse
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

import torch
import torch.backends.cudnn as cudnn
# backbones
from mtrpp.utils.eval_utils import load_ttmr_pp
from mtrpp.transfer.model_probing import ProbingLayer
from mtrpp.transfer.dataset_embs.data_manger import get_dataloader
from mtrpp.utils.transfer_utils import get_cls_config, single_query_evaluation, get_evaluation
from sklearn import metrics

parser = argparse.ArgumentParser(description='PyTorch MSD Training')
parser.add_argument('--msu_dir', type=str, default="/hdd2/seungheon/msu-benchmark/dataset")
parser.add_argument("--model_type", default="meta_tag_caption_sim", type=str)
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers')
# downstream options
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--warmup_epochs', default=0, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--min_lr', default=1e-9, type=float)
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=1, type=int,
                    help='GPU id to use.')
parser.add_argument('--print_freq', default=100, type=int)
# downstream options
parser.add_argument("--probe_type", default="mlp", type=str)
parser.add_argument("--mlp_dim", default=512, type=int)
parser.add_argument("--eval_dataset", default="gtzan", type=str)
parser.add_argument("--dropout", default=0, type=float)
parser.add_argument("--is_norm", default=1, type=int)
parser.add_argument("--l2_weight_decay", default=0, type=float)

args = parser.parse_args()

def main():
    save_dir = f"../exp/ttmrpp/{args.model_type}"
    model, sr, duration = load_ttmr_pp(save_dir)
    embs_dir = os.path.join(save_dir, "embs", args.eval_dataset)
    args.sr = sr
    args.duration = duration

    if args.eval_dataset in ["mtg_top50tags", "mtg_genre", "mtg_instrument", "mtg_moodtheme"]:
        embs_dir = os.path.join(save_dir, "embs", "mtg")
    audio_embs = torch.load(os.path.join(embs_dir, 'audio_embs.pt'))
    folder_name = f"{args.batch_size}_{args.lr}_{args.mlp_dim}_{args.dropout}_{args.is_norm}_{args.l2_weight_decay}"
    save_dir = os.path.join(save_dir, args.eval_dataset, folder_name) # update save_dir
    os.makedirs(save_dir, exist_ok=True)
    audio_dim = 128
    task_type, output_dim, loss_fn = get_cls_config(args)
    model = ProbingLayer(
        audio_dim = audio_dim,
        mlp_dim = args.mlp_dim,
        output_dim = output_dim,
        task_type = task_type,
        probe_type = args.probe_type,
        dropout = args.dropout,
        is_norm = args.is_norm,
        loss_fn = loss_fn
    )
    pretrained_object = torch.load(f'{save_dir}/best.pth', map_location='cpu')
    state_dict = pretrained_object['state_dict']
    model.load_state_dict(state_dict)

    test_loader = get_dataloader(args=args, audio_embs=audio_embs, split="TEST")
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    cudnn.benchmark = True
    model.eval()
    predictions, groudturths = [], []
    for batch in tqdm(test_loader):
        x = batch['audio']
        y = batch['binary']
        if args.gpu is not None:
            x = x.cuda(args.gpu, non_blocking=True)
            y = y.cuda(args.gpu, non_blocking=True)
        with torch.no_grad():
            predict = model.test_forward(x) # flatten batch
        predictions.append(predict.mean(0, True).detach().cpu())
        groudturths.append(y.detach().cpu())
    
    logits = torch.cat(predictions, dim=0)
    targets = torch.cat(groudturths, dim=0)
    if args.eval_dataset in ['fma', 'gtzan', 'emotify']:
        results = get_evaluation(targets.numpy(), logits.numpy(), test_loader.dataset.list_of_label, 'multiclass')
    else:
        results = get_evaluation(targets.numpy(), logits.numpy(),test_loader.dataset.list_of_label, 'multilabel')
    with open(os.path.join(save_dir, f"probing.json"), mode="w") as io:
        json.dump(results, io, indent=4)

if __name__ == '__main__':
    main()

    