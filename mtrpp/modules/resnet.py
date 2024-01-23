# reference: https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/model.py#L94
import torch
import torch.nn as nn
from collections import OrderedDict
from mtrpp.modules.mel import MelEncoder
from mtrpp.modules.pooling_layer import AttentionPool2d

class ModifiedResNet(nn.Module):
    def __init__(self, layers, output_dim=768, heads=8, n_mels=128, width=64, n_fft=1024, hop_size=0.01, duration=10, sr=22050):
        super().__init__()
        self.output_dim = output_dim
        self.n_mels = n_mels
        self.n_sample = sr * duration
        self.hop_length = int(hop_size * sr) # hard coding hop_size
        self.n_frames = int(self.n_sample // self.hop_length)
        self.num_of_pool = 2 ** 5
        self.n_ctx = int(self.n_frames // self.num_of_pool)
        # mel_transform
        self.mel_transform = MelEncoder(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
        # the 3-layer stem
        self.conv1 = nn.Conv2d(1, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)
        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)
        embed_dim = width * 32  # the ResNet feature dimension
        f_out_dim = n_mels // 32
        t_out_dim = int(self.n_frames // 2 ** self.num_of_pool)
        fc_dim = int(f_out_dim * t_out_dim)
        self.attn_pool = AttentionPool2d(fc_dim+1, embed_dim, heads, output_dim)
        
    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]
        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.mel_transform(x) # B x F x T
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x
        x = x.unsqueeze(1).type(self.conv1.weight.dtype) # B x 1 x 128, 1003
        x = stem(x) # B, 64, F//4, T//4
        x = self.layer1(x) # B, 256, F//4, T//4
        x = self.layer2(x) # B, 512, 16, 125
        x = self.layer3(x) # B, 1024, 8, 62
        x = self.layer4(x) # B, 2048, 4, 31
        x = self.attn_pool(x) # B x 2048 x 124 -> B x 768
        return x

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out