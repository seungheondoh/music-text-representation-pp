# refernece: https://github.com/openai/whisper/blob/main/whisper/audio.py
import os
from functools import lru_cache
import librosa
import torch
import torchaudio
import torch.nn as nn
import numpy as np

@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:
    np.savez_compressed(
        "mel_filters.npz",
        mel_128=librosa.filters.mel(sr=22050, n_fft=1024, n_mels=128),
    )
    """
    assert n_mels in {128}, f"Unsupported n_mels: {n_mels}"
    filters_path = os.path.join(os.path.dirname(__file__), "assets", "mel_filters.npz")
    with np.load(filters_path, allow_pickle=False) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)

class MelEncoder(nn.Module):
    """
    time-frequency represntation
    """
    def __init__(self, 
                sample_rate=22050,
                n_fft=1024,
                hop_length=int(22050*0.01),
                n_mels=128,
                padding=0,
                ):
        super(MelEncoder, self).__init__()
        self.sample_rate = sample_rate 
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.padding = padding

    def forward(self, audio):
        device = audio.device
        if not torch.is_tensor(audio):
            audio = torch.from_numpy(audio)
        if self.padding > 0:
            audio = F.pad(audio, (0, padding))
        window = torch.hann_window(self.n_fft).to(device)
        stft = torch.stft(audio, self.n_fft, self.hop_length, window=window, return_complex=True)
        magnitudes = stft[..., :-1].abs() ** 2
        filters = mel_filters(device, self.n_mels)
        mel_spec = filters @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec