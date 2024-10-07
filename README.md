# Text-to-Music Retrieval++ 
[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2410.03264-<COLOR>.svg)](https://arxiv.org/abs/2410.03264) 
[![Model_Wieghts](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-yellow)](https://huggingface.co/seungheondoh/ttmr-pp)
[![Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-yellow)](https://huggingface.co/collections/seungheondoh/enriching-music-descriptions-661e9342edcea210d61e981d)

This repository implements TTMR++ (Text-to-Music Retrieval++), an joint embedding model for finding music based on natural language queries. TTMR++ enhances music retrieval by utilizing rich text descriptions generated from a finetuned large language model and metadata. It addresses the challenge of matching descriptive queries about musical attributes and contextual elements, as well as finding similar tracks to user favorites. Our approach leverages various seed text sources, including music tag and caption datasets, and a knowledge graph of artists and tracks. 

<p align = "center">
<img src = "https://i.imgur.com/DkX6x8H.png">
</p>


> [**Enriching Music Descriptions with a Finetuned-LLM and Metadata for Text-to-Music Retrieval**](#)

> SeungHeon Doh, Minhee Lee, Dasaem Jeong and Juhan Nam 
> IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) 2024 


## Open Source Material
- [Demo-Example](https://seungheondoh.github.io/music-text-representation-pp-demo/)
- [Pre-trained model weights](https://huggingface.co/seungheondoh/ttmr-pp/tree/main) 
- [Dataset for training](https://huggingface.co/collections/seungheondoh/enriching-music-descriptions-661e9342edcea210d61e981d) 

## Installation
To run this project locally, follow the steps below:

default docker env setup
```
nvidia/cuda:12.4.0-cudnn-devel-ubuntu22.04
```

Install python and PyTorch:

- python==3.10
- torch==2.4.0 (Please install it according to your CUDA version.) 

```
conda create -n ttmrpp python=3.10
pip install torch==2.4.0 torchvision torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install -e .
```

## Quick Start
 
Our joint embedding model maps given text or audio inputs into a 128-dimensional space where dot product operations can be performed. This enables vector-based similarity search within this shared embedding space.

Key features:
- Converts text and audio inputs to 128-dimensional vectors
- Allows for efficient similarity comparisons using dot product
- Supports vector-based similarity search for both text-to-audio and audio-to-audio queries

This approach facilitates various music information retrieval tasks, including finding similar tracks or matching textual descriptions to audio content.

**Text Embedding Inference**
```
python get_latent.py --text_input "Solo acoustic performance with a mix of easy listening, suspenseful, and quirky tones, using percussion, piano, drums, guitar, and acoustic guitar in the key of D"
```
```
# output
torch.Size([128])
tensor([ 0.6764,  1.4445,  1.6331, -0.3785, -1.6008, -1.8907, -0.5335, -2.4279,
         0.5862, -1.4265,  1.5700,  0.2941, -0.1644,  0.4214, -0.3602, -0.4427,
        -0.5152,  0.4549, -0.4845,  0.8588,  0.5241, -0.1089, -0.3433,  1.1568,
        -0.3235,  0.1440,  1.3501, -1.0882,  0.5225,  0.3031,  0.0888, -1.8808,
        -1.3627, -0.9572,  0.5141,  0.2269, -1.3208,  0.5641,  1.3694,  0.5827,
        -1.8541, -0.9248,  0.5938, -0.9932,  1.3392, -0.1249, -0.2030,  0.6059,
         1.9466,  0.5013,  0.9022,  1.3424,  0.2544, -1.4827,  0.0810,  0.4067,
         0.2013, -0.9685, -0.0357, -1.4470,  0.1612,  1.0236, -0.5595,  1.2407,
        -1.0852, -1.5297,  0.0269, -1.7990, -1.7735,  0.8635, -1.2336, -1.0182,
        -0.3646,  0.1823,  0.4835,  0.2734,  1.1936, -0.1702, -0.2861,  0.5545,
         0.2683,  0.8781, -0.6287,  0.7145, -0.0621,  0.0727,  1.6062,  1.5663,
         0.8099,  0.8607, -0.5761, -0.4112,  0.0249, -0.4700, -0.8659,  2.5185,
         0.6934,  0.0688, -1.2782,  0.4591,  0.6703, -0.1287, -1.0413, -1.7163,
         0.4604, -0.7972, -1.1084, -0.2047, -2.7326,  0.5690,  0.9788, -0.4369,
         0.2757,  0.5833,  0.2993, -1.0671,  0.3357,  0.8352,  1.3189, -0.9685,
         0.6253,  0.3622, -0.0561,  0.3926,  0.0390, -0.2245,  1.8282, -1.1265])
```

**Audio Embedding Inference**
```
python get_latent.py --audio_path /data/songdescriber/audio/00/1051200.2min.mp3
```
```
# output
torch.Size([128])
tensor([-0.8627, -0.3114, -0.7411,  0.2821,  0.0451, -0.5751, -0.6763,  0.8624,
        -1.7115,  0.2977, -1.7292, -0.6813,  0.5862, -0.0491, -0.5726,  0.3168,
         2.0153,  1.6547,  1.3747,  1.3564,  0.6332,  3.0040,  0.9165,  0.0232,
        -0.8500, -1.8098,  0.1055,  0.1852,  1.1632,  0.6390,  1.1104, -0.7877,
         2.1423,  2.3567,  1.7464, -0.3014,  0.0438,  0.5841, -0.6928, -2.5418,
        -0.0198,  0.3745, -0.0359, -0.5544, -0.2944,  0.0439, -1.8000,  1.5492,
         1.2135,  1.0327, -2.5759,  1.2982,  1.1887,  0.3731,  0.1798,  0.2140,
        -1.4810, -0.0711,  1.3866,  2.1831, -1.7652,  0.1700, -0.3678, -0.3683,
        -2.2318, -0.5275, -0.1133, -0.8906, -1.6242,  0.1020, -0.7544,  0.1009,
        -1.3666,  0.4578,  0.2509,  0.4003,  1.1552,  1.0413,  0.7591,  0.0789,
        -0.1002, -0.4736, -0.4919, -0.8823,  0.0847, -1.0610,  0.9784, -0.6810,
         1.0142, -1.0927, -1.4362,  1.0715, -0.3416,  0.1231,  1.3490, -0.0231,
        -0.8571, -1.2626,  1.0419, -0.2499,  0.0827,  0.5733, -0.8997,  0.5425,
        -1.1702, -0.6742, -0.0705, -1.4876, -0.7527, -0.5106,  0.3274,  0.7580,
        -1.7251, -0.1431, -3.0829,  0.1516,  1.3367,  0.5953,  0.5610, -1.2726,
         2.6081, -1.1901, -0.3952, -1.2774,  0.6816, -1.7909,  2.0710, -1.1446])
```

## License
This project is under the CC-BY-NC 4.0 license.

## Finetuned LLaMA2 for Tag-to-Caption Augmentation

For Tag-to-Caption Augmentation using Large Language Model, please refer to our dedicated repository: [Tag-to-Caption Augmentation using Large Language Model](https://github.com/seungheondoh/llm-tag-to-caption). This repository contains the implementation details and code for the LLaMA2 finetuning process used in our project. Additionally, all preprocessed datasets used in this study, including the augmented captions, are available in our [Enriching Music Descriptions Dataset Collection](https://huggingface.co/collections/seungheondoh/enriching-music-descriptions-661e9342edcea210d61e981d) on Hugging Face. These resources provide comprehensive access to both the methodology and data used in our research.

## Acknowledgement
Part of the code is borrowed from the following repos. We would like to thank the authors of these repos for their contribution.

- Modified ResNet: [OpenAI CLIP](https://github.com/openai/CLIP/tree/main)
- Audio Frontend: [OpenAI Whisper](https://github.com/openai/whisper/blob/main/whisper/audio.py)
- Distributed Data Parallel Training: [Pytorch DDP](https://pytorch.org/tutorials/beginner/ddp_series_theory.html)

## Citation
Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follow.

```
@inproceedings{doh2024enriching,
  title={Enriching Music Descriptions with a Finetuned-LLM and Metadata for Text-to-Music Retrieval},
  author={Doh, SeungHeon and Lee, Minhee and Jeong, Dasaem and Nam, Juhan},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2024}
}
```