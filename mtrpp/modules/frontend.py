import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class SpecPatchEmbed(nn.Module):
    """ 2D spectrogram to Patch Embedding
    """
    def __init__(self, f_size=128, t_size=1003, p_w=16, p_h=16, in_chans=1, embed_dim=768, flatten=True):
        super().__init__()
        self.spec_size = (f_size, t_size)
        self.patch_size = (p_h, p_w)
        self.grid_size = (self.spec_size[0] // self.patch_size[0], self.spec_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = nn.LayerNorm(embed_dim) 

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class CNN2D(nn.Module):
    """ 2D spectrogram to more small patch filter
    """
    def __init__(self, in_chans=1, embed_dim=768, flatten=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(embed_dim // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(embed_dim // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(embed_dim)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)
        self.norm = nn.LayerNorm(embed_dim) 
        self.flatten = flatten

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class CNN1D(nn.Module):
    """ 2D spectrogram to Patch Embedding
    """
    def __init__(self, in_chans=128, embed_dim=768, flatten=True):
        super().__init__()
        self.conv1 = nn.Conv1d(in_chans, embed_dim // 2, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(embed_dim // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(embed_dim)
        self.relu2 = nn.ReLU(inplace=True)
        self.norm = nn.LayerNorm(embed_dim) 

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x