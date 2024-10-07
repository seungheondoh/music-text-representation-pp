import os
from torch import nn
from .hook import CLAP_Module

class LAION_CLAP(nn.Module):
    """warpping class for LAION CLAP"""

    def __init__(self, pretrain_dir, ckpt, device):
        super(LAION_CLAP, self).__init__()
        self.pretrain_dir = pretrain_dir
        self.device = device
        self.build_model(ckpt)

    def build_model(self, ckpt):
        if ckpt == "630k-best.pt":
            self.model = CLAP_Module(
                enable_fusion=False,
                amodel="HTSAT-tiny",
                tmodel="roberta",
                device=self.device,
            )
        elif ckpt == "music_audioset_epoch_15_esc_90.14.pt":
            self.model = CLAP_Module(
                enable_fusion=False,
                amodel="HTSAT-base",
                tmodel="roberta",
                device=self.device,
            )
        elif ckpt == "630k-audioset-best.pt":
            self.model = CLAP_Module(
                enable_fusion=False,
                amodel="HTSAT-tiny",
                tmodel="roberta",
                device=self.device,
            )
        elif ckpt == "630k-fusion-best.pt":
            self.model = CLAP_Module(
                enable_fusion=True,
                amodel="HTSAT-tiny",
                tmodel="roberta",
                device=self.device,
            )
        elif ckpt == "630k-audioset-fusion-best.pt":
            self.model = CLAP_Module(
                enable_fusion=True,
                amodel="HTSAT-tiny",
                tmodel="roberta",
                device=self.device,
            )
        self.model.load_ckpt(
            ckpt=os.path.join(self.pretrain_dir, ckpt),
        )  # download the default pretrained checkpoint.

    def audio_forward(self, audio):
        audio_embed = self.model.get_audio_embedding_from_data(audio, use_tensor=True)
        return audio_embed

    def text_forward(self, text):
        if len(text) == 1: # for single prompt inference, return error
            text_embed = self.model.get_text_embedding(text * 2, use_tensor=True)
            text_embed = text_embed[:1]
        else:    
            text_embed = self.model.get_text_embedding(text, use_tensor=True)
        return text_embed
