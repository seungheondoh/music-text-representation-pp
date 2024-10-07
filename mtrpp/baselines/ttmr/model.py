import os
import torch
from torch import nn
from .utils import get_model


class TTMR(nn.Module):
    """warpping class for univerisal text-to-music retrieval, ICASSP, Doh et al."""

    def __init__(self, pretrain_dir, device):
        super(UniTTM, self).__init__()
        self.pretrain_dir = pretrain_dir
        self.device = device
        self.build_model(pretrain_dir)

    def build_model(self, pretrain_dir):
        model, tokenizer, config = get_model(save_dir=pretrain_dir)
        self.model = model
        self.tokenizer = tokenizer

    def audio_forward(self, audio):
        audio_embed = self.model.encode_audio(audio)
        return audio_embed

    def text_forward(self, text):
        tokens = self.tokenizer(
            text, padding="longest", truncation=True, return_tensors="pt"
        )
        text_input = tokens["input_ids"].to(self.device)
        attn_mask = tokens["attention_mask"].to(self.device)
        text_embed = self.model.encode_bert_text(
            text_input, attn_mask
        )
        return text_embed

    def audio_backbone_features(self, audio):
        audio_emb = self.model.audio_encoder(audio.to(self.device))
        h_audio = self.model.a_latent(audio_emb[:, 0, :])
        return h_audio

    def text_backbone_features(self, text):
        tokens = self.tokenizer(
            text, padding="longest", truncation=True, return_tensors="pt"
        )
        text_input = tokens["input_ids"].to(self.device)
        attn_mask = tokens["attention_mask"].to(self.device)
        text_emb = self.model.text_encoder(input_ids=text_input, attention_mask=attn_mask)
        h_text = self.model.t_latent(text_emb["last_hidden_state"][:, 0, :])
        return h_text