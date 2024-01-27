import os
import json
import torch
import torch.nn as nn
import argparse
import scipy
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from mtrpp.datasets.olga import OLGA_Inference
from mtrpp.utils.eval_utils import load_ttmr_pp, print_model_params
from sklearn import metrics

parser = argparse.ArgumentParser(description="")
parser.add_argument("--data_dir", type=str, default="../../dataset")
parser.add_argument("--data_type", type=str, default="olga")
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
parser.add_argument("--audio_enc", type=str, default=".npy")
args = parser.parse_args()


def load_olga_annotation(args):
    artist_meta = json.load(open(os.path.join(args.data_dir, "olga-msd", "artist_meta.json"), 'r'))
    df_olga = pd.read_csv(os.path.join(args.data_dir,"olga-msd","olga.csv"), index_col = 0)
    artist_id = list(df_olga['musicbrainz_id'])
    artist_connections = scipy.sparse.load_npz(os.path.join(args.data_dir, "olga-msd","filip2021","artist_connections.npz"))
    matrix = artist_connections + artist_connections.T
    dense = matrix.todense()
    query_artist = list(df_olga[df_olga['partition'] == "test"]['musicbrainz_id'])
    df_gt = pd.DataFrame(dense, index=artist_id, columns=artist_id, dtype='int') # query by artist
    return artist_meta, df_gt, query_artist

def _sim_matrix(query_embs, target_embs):
    query_embs  = nn.functional.normalize(query_embs,  dim=1)
    target_embs = nn.functional.normalize(target_embs, dim=1)
    sim_matrix = query_embs  @ target_embs.T
    return sim_matrix.numpy()

def main(args):
    artist_meta, df_gt, query_artist = load_olga_annotation(args)
    save_dir = f"exp/{args.model_type}/{args.caption_type}"
    model, sr, duration = load_ttmr_pp(save_dir, model_types=args.ckpt_type)
    print_model_params(model)
    model = model.to(args.device)
    model.eval()
    
    dataset = OLGA_Inference(
        data_dir=args.data_dir,
        split=None,
        audio_loader=args.audio_loader,
        caption_type = "caption",
        sr=sr,
        duration=duration,
        audio_enc = args.audio_enc
    )
    
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
            audio_embs = model.audio_forward(batch_audio.to(args.device))
        unbatch_audio = audio_embs.view(B,C,-1)
        audio_embs = unbatch_audio.mean(1, False).detach().cpu()
        for name, embs in zip(fname, audio_embs):
            track_embs[name] = embs
    
    artist_embs, q_art, t_art = [], [], []
    for artist_mbid in df_gt.columns:
        instance = artist_meta[artist_mbid]
        if len(instance["pos_tracks"]) > 0:
            artist_emb = torch.stack([track_embs[track_id] for track_id in instance["pos_tracks"]])
            artist_embs.append(artist_emb.mean(0, False))
            t_art.append(artist_mbid)
            if artist_mbid in query_artist:
                q_art.append(artist_mbid)

    query_embs = []
    for artist_mbid in query_artist:
        item = artist_meta[artist_mbid]
        artist_text = ", ".join(item["artist_name"])
        text = f"similar with artist {artist_text}"
        with torch.no_grad():
            query_emb = model.text_forward([text]) 
        query_embs.append(query_emb.detach().cpu().squeeze(0))

    query_embs = torch.stack(query_embs)
    artist_embs = torch.stack(artist_embs)
    text_prediction = _sim_matrix(query_embs, artist_embs)
    audio_prediction = _sim_matrix(artist_embs, artist_embs)

    audio_gt = df_gt.loc[q_art][t_art]
    audio_pred = pd.DataFrame(audio_prediction, index=t_art, columns=t_art).loc[q_art]

    text_gt = df_gt.loc[query_artist][t_art]
    text_pred = pd.DataFrame(text_prediction, index=query_artist, columns=t_art)

    scores = {
        "audio_ndcg@200": metrics.ndcg_score(audio_gt, audio_pred, k=200),
        "text_ndcg@200": metrics.ndcg_score(text_gt, text_pred, k=200),
        "audio_data": audio_gt.shape,
        "text_data": text_gt.shape
    }
    os.makedirs(os.path.join(save_dir, args.data_type), exist_ok=True)
    with open(os.path.join(save_dir, args.data_type, f"artist.json"), "w") as json_file:
        json.dump(scores, json_file, indent=4)

if __name__ == "__main__":
    main(args)
