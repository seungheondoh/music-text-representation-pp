import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn import metrics
from omegaconf import DictConfig, OmegaConf
from mtrpp.model.dual_encoder import DualEncoderModel
        
def get_query2target_idx(query2target, target2idx):
    query2target_idx = {}
    for query, target_list in query2target.items():
        query2target_idx[query] = [target2idx[i] for i in target_list]
    return query2target_idx

def get_task_predictions(query_features, target_features):
    """Get similarity matrix from model output."""
    query_features = torch.nn.functional.normalize(query_features, dim=-1)
    target_features = torch.nn.functional.normalize(target_features, dim=-1)
    sim_matrix = query_features @ target_features.T
    sim_matrix = sim_matrix.numpy()
    return sim_matrix

def print_model_params(model):
    n_parameters = sum(p.numel() for p in model.parameters())
    train_n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("============")
    print('number of params (M): %.2f' % (n_parameters / 1.e6))
    print('number train of params (M): %.2f' % (train_n_parameters / 1.e6))
    print("============")


def load_pretrain_model(args):
    if args.model_type == "random":
        model, sr, duration  = torch.nn.Identity(), 48000, 10
    elif args.model_type == "ttmr_pp":
        model, sr, duration = load_unttim_pp(args)
    else:
        model, sr, duration = load_baselines(args)
    return model, sr, duration

def load_ttmr_pp(save_dir, model_types="last"):
    config = OmegaConf.load(f'{save_dir}/hparams.yaml')
    pretrained_object = torch.load(f'{save_dir}/{model_types}.pth', map_location='cpu')
    state_dict = pretrained_object['state_dict']
    model = DualEncoderModel(
        text_arch=config.text_arch,
        n_mels=config.n_mels, 
        n_fft=config.n_fft,
        hop_size=config.hop_size,
        width=config.width,
        head =config.n_heads,
        sr=config.sr, 
        duration=config.duration, 
        max_length=config.max_length, 
        audio_dim=config.audio_dim, 
        text_dim=config.text_dim, 
        mlp_dim=config.mlp_dim, 
        temperature=config.temperature
    )
    model.load_state_dict(state_dict)
    return model, config.sr, config.duration