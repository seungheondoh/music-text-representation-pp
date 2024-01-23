import os
import torch
import torchaudio
import numpy as np
import torch.nn.functional as F
from torch import Tensor, nn
from typing import Dict, Iterable, Optional
from dataclasses import dataclass
from mtrpp.modules.mel import MelEncoder
from mtrpp.modules.frontend import SpecPatchEmbed, CNN2D, CNN1D
from mtrpp.modules.pos_embed import get_2d_sincos_pos_embed_flexible
from mtrpp.modules.attention import Block, LayerNorm

class AST(nn.Module):
    def __init__(
        self, layers, frontend, pooling_type, output_dim=768, heads=8, n_mels=128, n_fft=1024, win_length=1024, duration=10, sr=22050, mask_ratio=0.75):
        super().__init__()
        self.n_layer = layers
        self.frontend = frontend
        self.pooling_type = pooling_type
        self.n_heads = heads
        self.output_dim = output_dim
        self.n_mels = n_mels
        self.n_sample = sr * duration
        self.hop_length = int(0.01 * sr) # hard coding hop_size
        self.n_frames = int(self.n_sample // self.hop_length) + 1
        self.f_out_dim = n_mels // 16
        self.t_out_dim = int(self.n_frames // 16) # patch size
        self.n_ctx = int(self.f_out_dim * self.t_out_dim)
        self.max_length = 512 # max length (504, 502, 496)
        self.mask_ratio = mask_ratio
        self.len_keep = int(self.n_ctx * (1 - self.mask_ratio))
        # mel_transform
        self.mel_transform = MelEncoder(
            sample_rate=sr,
            f_min=0,
            f_max=int(sr//2),
            n_fft=n_fft,
            win_length=win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
        if self.frontend == "patchify":
            self.patch_proj = SpecPatchEmbed(
                f_size = self.n_mels,
                t_size = self.n_frames,
                p_w=16, 
                p_h=16, 
                in_chans=1, 
                embed_dim=output_dim, 
                flatten=True
            )
        elif self.frontend == "cnn2d":
            self.patch_proj = CNN2D(
                in_chans=1, 
                embed_dim=output_dim
            )
        elif self.frontend == "cnn1d":
            self.patch_proj = CNN2D(
                in_chans=1, 
                embed_dim=output_dim
            )
        self.num_patches = self.patch_proj.num_patches
        self.positional_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, output_dim), requires_grad=False) 
        if self.pooling_type == "cls":
            self.class_embedding = nn.Parameter(torch.randn(output_dim))    
        if self.pooling_type == "attn":
            self.attn_pool = AttentionPool1d(embed_dim=output_dim, num_heads=4)
        self.transformer = nn.ModuleDict(dict(
            h = nn.ModuleList([
                Block(
                        n_layer=self.n_layer, n_head=self.n_heads, n_embd=self.output_dim, bias=False) 
                        for _ in range(self.n_layer
                    )
                ]),
            ln_f = LayerNorm(self.output_dim, bias=False),
        ))
        self.initialize_weights()

    
    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed_flexible(self.positional_embedding.shape[-1], self.patch_proj.grid_size, cls_token=True)    
        self.positional_embedding.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        w = self.patch_proj.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        if self.pooling_type == "cls":
            torch.nn.init.normal_(self.class_embedding, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - self.mask_ratio))
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward_cls_pooling(self, x, is_train):
        x = self.mel_transform(x) # (batch_size, n_mels, n_ctx)
        x = self.patch_proj(x)
        class_embedding = self.class_embedding.repeat(x.shape[0], 1, 1)
        x = torch.cat((class_embedding, x), dim=1)
        x = (x + self.positional_embedding[:x.shape[1]]).to(x.dtype)
        if is_train:
            x, _, _ = self.random_masking(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        x = x[:,0,:] # cls embedding
        return x

    def forward_avg_pooling(self, x, is_train):
        x = self.mel_transform(x) # (batch_size, n_mels, n_ctx)
        x = self.patch_proj(x)
        x = (x + self.positional_embedding[:, :x.shape[1], :]).to(x.dtype)
        if is_train:
            x, _, _ = self.random_masking(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        x = x.mean(dim=1, keepdim=False)
        return x

    def forward_attn_pooling(self, x, is_train):
        x = self.mel_transform(x) # (batch_size, n_mels, n_ctx)
        x = self.patch_proj(x)
        x = (x + self.positional_embedding[:x.shape[1]]).to(x.dtype)
        if is_train:
            x, _, _ = self.random_masking(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        x = self.attn_pool(x)
        return x

    def forward(self, x: Tensor, is_train: bool = True):
        """
        x : torch.Tensor, shape = (batch_size, waveform)
            single channel wavform
        """
        if self.pooling_type == "cls":
            x = self.forward_avg_pooling(x, is_train)
        elif self.pooling_type == "avg":
            x = self.forward_avg_pooling(x, is_train)
        elif self.pooling_type == "attn":
            x = self.forward_attn_pooling(x, is_train)
        return x


class AttentionPool1d(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.permute(1, 0, 2)  # BLD -> LBD
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (L+1)NC
        # x = x + self.positional_embedding[:, None, :].to(x.dtype)  # Already we add positional embedding
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)
