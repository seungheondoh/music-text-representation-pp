STR_CLIP_ID = 'clip_id'
STR_AUDIO_SIGNAL = 'audio_signal'
STR_TARGET_VECTOR = 'target_vector'


STR_CH_FIRST = 'channels_first'
STR_CH_LAST = 'channels_last'

AUDIOSET_MUSIC = ['/m/0z9c','/m/0mkg','/m/042v_gx','/m/0fd3y','/t/dd00036','/m/025td0t','/m/0192l','/m/018j2','/m/0bm02','/m/018vs','/m/02cz_7','/m/0395lw','/m/0gg8l','/m/0155w','/m/0l14_3','/m/01kcd','/m/015vgc','/m/01xqw','/m/02bk07','/m/0l14jd','/m/02mscn','/m/0140xf','/m/01wy6','/m/0ggq0m','/m/01lyv','/m/0239kh','/m/01qbl','/m/0ggx5q','/m/02bxd','/m/026z9','/m/02fsn','/m/0283d','/m/02hnl','/m/02k_mr','/m/026t6','/m/07s72n','/m/02sgy','/m/08cyft','/m/02lkt','/m/03xq_f','/m/0m0jc','/t/dd00035','/m/0326g','/m/0l14j_','/m/02w4v','/m/0319l','/m/02x8m','/t/dd00032','/m/0dwtp','/m/0mbct','/m/0dls3','/m/0342h','/m/03gvt','/t/dd00031','/m/03qjg','/m/03m5k','/m/03q5t','/m/03lty','/m/0glt670','/m/03mb9','/m/05rwpb','/m/03_d0','/m/03r5q_','/m/05148p4','/m/07pkxdp','/m/0j45pbj','/m/04rzd','/m/0dwsp','/m/06j64v','/m/05fw6t','/m/0164x2','/m/028sqc','/m/0dq0md','/m/0g293','/m/02v2lh','/m/05pd6','/m/013y1f','/m/0l14md','/m/05r5c','/m/0fx80y','/m/064t9','/m/0dl5d','/m/05w3f','/m/05r6t','/m/05r5wn','/m/06cqb','/m/06j6l','/m/03t3fj','/m/07sbbz2','/m/06by7','/t/dd00033','/m/0ln16','/m/06ncr','/t/dd00037','/m/01hgjl','/m/0l14l2','/m/0l14t7','/m/0jtg0','/m/06rqw','/m/06rvn','/m/0gywn','/m/0l14gg','/m/06w87','/m/0l156b','/m/02qmj0d','/m/07s0s5r','/m/015y_n','/m/0l14qv','/m/01p970','/m/07brj','/m/01glhc','/m/07gxw','/t/dd00034','/m/02cjck','/m/07kc_','/m/011k_j','/m/02p0sh1','/m/07lnk','/m/07c6l','/m/07gql','/m/016622','/m/07xzm','/m/0dwt5','/m/01z7dr','/m/07y_7','/m/0y4f8','/m/04wptg','/m/085jw','/m/01sm1g','/m/01bns_']

import io
import os
import torch
import logging
import subprocess
from typing import Tuple
from pathlib import Path

# import librosa
import numpy as np
import soundfile as sf
import librosa
import itertools
from numpy.fft import irfft

def _resample_load_ffmpeg(path: str, sample_rate: int, downmix_to_mono: bool) -> Tuple[np.ndarray, int]:
    """
    Decoding, downmixing, and downsampling by librosa.
    Returns a channel-first audio signal.

    Args:
        path:
        sample_rate:
        downmix_to_mono:

    Returns:
        (audio signal, sample rate)
    """
    def _decode_resample_by_ffmpeg(filename, sr):
        """decode, downmix, and resample audio file"""
        channel_cmd = '-ac 1 ' if downmix_to_mono else ''  # downmixing option
        resampling_cmd = f'-ar {str(sr)}' if sr else ''  # downsampling option
        cmd = f"ffmpeg -i \"{filename}\" {channel_cmd} {resampling_cmd} -f wav -"
        p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        return out

    src, sr = sf.read(io.BytesIO(_decode_resample_by_ffmpeg(path, sr=sample_rate)))
    return src.T, sr


def _resample_load_librosa(path: str, sample_rate: int, downmix_to_mono: bool, **kwargs) -> Tuple[np.ndarray, int]:
    """
    Decoding, downmixing, and downsampling by librosa.
    Returns a channel-first audio signal.
    """
    src, sr = librosa.load(path, sr=sample_rate, mono=downmix_to_mono, **kwargs)
    return src, sr


def load_audio(
    path: str or Path,
    ch_format: str,
    sample_rate: int = None,
    downmix_to_mono: bool = False,
    resample_by: str = 'librosa',
    **kwargs,
) -> Tuple[np.ndarray, int]:
    """A wrapper of librosa.load that:
        - forces the returned audio to be 2-dim,
        - defaults to sr=None, and
        - defaults to downmix_to_mono=False.

    The audio decoding is done by `audioread` or `soundfile` package and ultimately, often by ffmpeg.
    The resampling is done by `librosa`'s child package `resampy`.

    Args:
        path: audio file path
        ch_format: one of 'channels_first' or 'channels_last'
        sample_rate: target sampling rate. if None, use the rate of the audio file
        downmix_to_mono:
        resample_by (str): 'librosa' or 'ffmpeg'. it decides backend for audio decoding and resampling.
        **kwargs: keyword args for librosa.load - offset, duration, dtype, res_type.

    Returns:
        (audio, sr) tuple
    """
    if ch_format not in (STR_CH_FIRST, STR_CH_LAST):
        raise ValueError(f'ch_format is wrong here -> {ch_format}')

    if os.stat(path).st_size > 8000:
        if resample_by == 'librosa':
            src, sr = _resample_load_librosa(path, sample_rate, downmix_to_mono, **kwargs)
        elif resample_by == 'ffmpeg':
            src, sr = _resample_load_ffmpeg(path, sample_rate, downmix_to_mono)
        else:
            raise NotImplementedError(f'resample_by: "{resample_by}" is not supposred yet')
    else:
        raise ValueError('Given audio is too short!')
    return src, sr


def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x):
    x = ((x - x.min()) / (x.max() - x.min())) * (2) - 1.  # peak -1, 1 normalize
    return (x * 32767.).astype(np.int16)

# def int16_to_float32_torch(x):
#     return (x / 32767.0).type(torch.float32)

# def float32_to_int16_torch(x):
#     x = torch.clamp(x, min=-1., max=1.)
#     return (x * 32767.).type(torch.int16)
