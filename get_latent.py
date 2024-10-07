import os
import argparse
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from mtrpp.utils.eval_utils import load_ttmr_pp
from mtrpp.utils.audio_utils import int16_to_float32, float32_to_int16, load_audio, STR_CH_FIRST

parser = argparse.ArgumentParser(description='Get latent embeddings for text or audio inputs')
parser.add_argument("--gpu", default=0, type=int, help="GPU device index")
parser.add_argument("--text_input", default="", type=str, help="Input text for embedding")
parser.add_argument("--audio_path", default="", type=str, help="Path to audio file for embedding")
args = parser.parse_args()

if not os.path.isfile("best.pth"):
    torch.hub.download_url_to_file('https://huggingface.co/seungheondoh/ttmr-pp/resolve/main/ttmrpp_resnet_roberta.pth', 'best.pth')
    torch.hub.download_url_to_file('https://huggingface.co/seungheondoh/ttmr-pp/resolve/main/ttmrpp_resnet_roberta.yaml', 'hparams.yaml')
    
SR = 22050
N_SAMPLES = int(SR * 10)

def load_wav(audio_path):
    """
    Load and preprocess audio file.
    
    Args:
        audio_path (str): Path to the audio file.
        
    Returns:
        torch.Tensor: Preprocessed audio tensor.
    """
    audio, _ = load_audio(
        path=audio_path,
        ch_format=STR_CH_FIRST,
        sample_rate=22050,
        downmix_to_mono=True)
    if len(audio.shape) == 2:
        audio = audio.squeeze(0)
    audio = int16_to_float32(float32_to_int16(audio))
    ceil = int(audio.shape[-1] // N_SAMPLES)
    audio_tensor = torch.from_numpy(np.stack(np.split(audio[:ceil * N_SAMPLES], ceil)).astype('float32'))
    return audio_tensor

def get_audio_embedding(model, audio_path):
    """
    Get embedding for audio input.
    
    Args:
        model (torch.nn.Module): The TTMR++ model.
        audio_path (str): Path to the audio file.
        
    Returns:
        torch.Tensor: Audio embedding.
    """
    audio = load_wav(audio_path)
    with torch.no_grad():
        z_audio = model.audio_forward(audio.cuda(args.gpu))
    z_audio = z_audio.mean(0).detach().cpu()
    return z_audio.float()

def get_text_embedding(model, text):
    """
    Get embedding for text input.
    
    Args:
        model (torch.nn.Module): The TTMR++ model.
        text (str): Input text.
        
    Returns:
        torch.Tensor: Text embedding.
    """
    with torch.no_grad():
        z_tag = model.text_forward([text])    
    z_tag = z_tag.squeeze(0).detach().cpu()
    return z_tag.float()

if __name__ == "__main__":
    save_dir = "./"
    model, sr, duration = load_ttmr_pp(save_dir, model_types="best")
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    cudnn.benchmark = True
    model.eval()
    
    if args.text_input:
        z_tag = get_text_embedding(model, args.text_input)
        print(z_tag.shape)
        print(z_tag)
    elif args.audio_path:
        z_audio = get_audio_embedding(model, args.audio_path)
        print(z_audio.shape)
        print(z_audio)