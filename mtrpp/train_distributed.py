import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import math
import random
import shutil
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
# ddp setup
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group

import numpy as np
from mtrpp.datasets.dataloader import load_train_dataset, load_valid_dataset
from mtrpp.model.dual_encoder import DualEncoderModel
from mtrpp.utils.train_utils import Logger, AverageMeter, ProgressMeter, EarlyStopping, save_hparams
from mtrpp.utils.eval_utils import print_model_params

parser = argparse.ArgumentParser(description='PyTorch MSD Training')
parser.add_argument('--framework', type=str, default="pretrain")
parser.add_argument('--data_dir', type=str, default="../../dataset")
parser.add_argument('--train_data', type=str, default="all")
parser.add_argument("--caption_type", default="meta_tag_caption_sim", type=str) # gpu caption_type
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--total_steps', default=2**15, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_steps', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=96, type=int, metavar='N')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--min_lr', default=1e-9, type=float)
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--print_freq', default=10, type=int)
parser.add_argument("--cos", default=True, type=bool)
# loader config
parser.add_argument("--n_fft", default=1024, type=int)
parser.add_argument("--hop_size", default=0.01, type=float)
parser.add_argument("--sr", default=22050, type=int)
parser.add_argument("--duration", default=10, type=int)
parser.add_argument("--max_length", default=128, type=int)
parser.add_argument("--audio_loader", default="ffmpeg", type=str)
# train confing 
parser.add_argument("--audio_arch", default="resnet", type=str)
parser.add_argument("--text_arch", default="roberta-base", type=str)
parser.add_argument("--n_heads", default=8, type=int)
parser.add_argument("--width", default=64, type=int)
parser.add_argument("--n_mels", default=128, type=int)
parser.add_argument("--audio_dim", default=768, type=int)
parser.add_argument("--text_dim", default=768, type=int)
parser.add_argument("--mlp_dim", default=128, type=int)
parser.add_argument("--temperature", default=0.1, type=float)
parser.add_argument("--tid", default="base", type=str)

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    # single node with multiple GPUs
    num_of_node = 1
    ngpus_per_node = torch.cuda.device_count()
    world_size = ngpus_per_node * num_of_node
    mp.spawn(
        main_worker, nprocs=world_size, args=(ngpus_per_node, args)
    )
    destroy_process_group()

def main_worker(rank, ngpus_per_node, args):
    args.gpu = rank
    ddp_setup(rank, ngpus_per_node)
    train_dataset = load_train_dataset(args)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True,
        sampler=DistributedSampler(train_dataset)
    )
    model = DualEncoderModel(
        text_arch=args.text_arch,
        n_mels=args.n_mels, 
        n_fft=args.n_fft,
        hop_size=args.hop_size,
        width=args.width,
        head =args.n_heads,
        sr=args.sr, 
        duration=args.duration, 
        max_length=args.max_length, 
        audio_dim=args.audio_dim, 
        text_dim=args.text_dim, 
        mlp_dim=args.mlp_dim, 
        temperature=args.temperature
    )
    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DistributedDataParallel(model, device_ids=[args.gpu])

    optimizer = torch.optim.AdamW(model.parameters(), args.lr)
    save_dir = f"exp/{args.train_data}/{args.caption_type}_{args.tid}/"
    args.epochs = args.total_steps // len(train_loader)
    args.start_epoch = args.start_steps // len(train_loader)
    args.warmup_steps = 5000 # 10% of training step is warmup
    logger = None
    if rank == 0:
        logger = Logger(save_dir)
        print_model_params(model.module.audio_encoder)
        print_model_params(model.module.text_encoder)
        args.batch_size = int(args.batch_size * ngpus_per_node)
        save_hparams(args, save_dir)
        print(args.start_epoch, args.epochs)
    
    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader, model, optimizer, epoch, logger, args)
        
    torch.save({'epoch': epoch, 'state_dict': model.module.state_dict(), 'optimizer' : optimizer.state_dict()}, f'{save_dir}/last.pth')
    print("We are at epoch:", epoch)

def train(train_loader, model, optimizer, epoch, logger, args):
    train_losses = AverageMeter('Train Loss', ':.4e')
    progress = ProgressMeter(len(train_loader),[train_losses],prefix="Epoch: [{}]".format(epoch))
    iters_per_epoch = len(train_loader)
    model.train()
    train_loader.sampler.set_epoch(epoch)
    for data_iter_step, batch in enumerate(train_loader):
        current_step = int(epoch * iters_per_epoch) + data_iter_step
        lr = adjust_learning_rate(optimizer, current_step, args)
        fname, text, audio_tensor = batch
        if args.gpu is not None:
            audio_tensor = audio_tensor.cuda(args.gpu, non_blocking=True)
        optimizer.zero_grad()
        # compute output
        loss = model(audio=audio_tensor, text=text)
        train_losses.step(loss.item(), audio_tensor.size(0))
        if args.gpu == 0:
            logger.log_train_loss(loss, epoch * iters_per_epoch + data_iter_step)
            logger.log_learning_rate(lr, epoch * iters_per_epoch + data_iter_step)
        loss.backward()
        optimizer.step()
        if data_iter_step % args.print_freq == 0:
            progress.display(data_iter_step)

def adjust_learning_rate(optimizer, current_step, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    warmup_steps = args.warmup_steps
    total_steps = args.total_steps
    if current_step < warmup_steps:
        lr = args.lr * current_step / warmup_steps
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (current_step - warmup_steps) / (total_steps - warmup_steps)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

if __name__ == '__main__':
    main()
    