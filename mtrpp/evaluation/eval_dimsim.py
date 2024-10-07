import os
import json
import torch
import torch.nn as nn
import argparse
import transformers
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from mtrpp.datasets.dimsim import Dimsim
from mtrpp.utils.eval_utils import load_ttmr_pp, print_model_params
from sklearn import metrics

parser = argparse.ArgumentParser(description="")
parser.add_argument("--data_dir", type=str, default="../../dataset")
parser.add_argument("--data_type", type=str, default="dimsim")
parser.add_argument("--audio_loader", type=str, default="ffmpeg")
parser.add_argument("--eval_query", type=str, default="caption")
parser.add_argument("--device", type=str, default="cuda:0")
# ttmr_pp config
parser.add_argument('--model_type', type=str, default="ttmrpp")
parser.add_argument("--caption_type", type=str, default="meta_tag_caption_sim")
# train confing 
parser.add_argument("--tid", default="base", type=str)
parser.add_argument("--ckpt_type", default="last", type=str)
args = parser.parse_args()


def _sim_matrix(query_embs, target_embs):
    query_embs  = nn.functional.normalize(query_embs,  dim=1)
    target_embs = nn.functional.normalize(target_embs, dim=1)
    sim_matrix = query_embs  @ target_embs.T
    return sim_matrix.cpu().numpy()[0] # query index

def load_unique_id(track_id, position, start_sec):
    return f"{track_id}_{position}_{start_sec}"

def prediction(df_dimsim, query_embs, audio_embs, query_type="text"):
    prediction_song1_vote = []
    for idx in range(len(df_dimsim)):
        triplet = df_dimsim.iloc[idx]
        anc_id = load_unique_id(triplet.anchor_id, "anchor", triplet.anchor_start_seconds)
        song1_id = load_unique_id(triplet.song1_id, "song1", triplet.song1_start_seconds)
        song2_id = load_unique_id(triplet.song2_id, "song2", triplet.song2_start_seconds)

        if query_type == "text":
            anc_feature = query_embs[anc_id]
        elif query_type == "audio":
            anc_feature = audio_embs[anc_id]
        song1_feature = audio_embs[song1_id]
        song2_feature = audio_embs[song2_id]

        song1_sim = _sim_matrix(anc_feature, song1_feature)
        song2_sim = _sim_matrix(anc_feature, song2_feature)
        if song1_sim > song2_sim:
            prediction_song1_vote.append(1)
        else:
            prediction_song1_vote.append(0)
    return prediction_song1_vote

def main(args):
    save_dir = f"exp/{args.model_type}/{args.caption_type}"
    model, sr, duration = load_ttmr_pp(save_dir, model_types=args.ckpt_type)
    print_model_params(model)
    model = model.to(args.device)
    model.eval()
    
    dataset = Dimsim(
        data_dir=args.data_dir,
        split=None,
        audio_loader=args.audio_loader,
        caption_type = "caption",
        sr=sr,
        duration=duration,
        audio_enc = ".mp3"
    )

    pred_a2a, pred_t2t = [], []
    pred_t2a, pred_t2m = [], []
    for item in tqdm(dataset):
        audio_anc = item["audio_anc"]
        audio_song1 = item["audio_song1"]
        audio_song2 = item["audio_song2"]
        
        text_anc = item["text_anc"]
        text_song1 = item["text_song1"]
        text_song2 = item["text_song2"]
        
        with torch.no_grad():
            z_audio_anc = model.audio_forward(audio_anc.unsqueeze(0).to(args.device))
            z_audio_song1 = model.audio_forward(audio_song1.unsqueeze(0).to(args.device))
            z_audio_song2 = model.audio_forward(audio_song2.unsqueeze(0).to(args.device))
        
            z_text_anc = model.text_forward([text_anc])            
            z_text_song1 = model.text_forward([text_song1])            
            z_text_song2 = model.text_forward([text_song2])            
            
        audio_embs = torch.cat([z_audio_anc, z_audio_song1, z_audio_song2], dim=0)
        query_embs = torch.cat([z_text_anc, z_text_song1, z_text_song2], dim=0)
        mixed_embs = torch.stack([query_embs, audio_embs]).mean(dim=0) 
        
        text2text = _sim_matrix(query_embs, query_embs)
        text2audio = _sim_matrix(query_embs, audio_embs)
        text2mix = _sim_matrix(audio_embs, mixed_embs)
        audio2audio = _sim_matrix(audio_embs, audio_embs)

        if text2audio[1] > text2audio[2]:
            pred_t2a.append(1)
        else:
            pred_t2a.append(0)

    df_dimsim = dataset.dimsim_gt
    ground_truth = list(df_dimsim['song1_vote'])
    scores = {
        "t2a_acc": metrics.accuracy_score(ground_truth, pred_t2a),
        "num_of_track": len(ground_truth)
    }
    os.makedirs(os.path.join(save_dir, args.data_type), exist_ok=True)
    with open(os.path.join(save_dir, args.data_type, f"track.json"), "w") as json_file:
        json.dump(scores, json_file, indent=4)

if __name__ == "__main__":
    main(args)
