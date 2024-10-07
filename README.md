# Text-to-Music Retrieval++ 
[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-0000.0000-<COLOR>.svg)](https://arxiv.org/abs/2410.03264) 
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

## Re-Implementation

## License
This project is under the CC-BY-NC 4.0 license.

## Finetuned LLaMA2 for Tag-to-Caption Augmentation
- see this repo: [Tag-to-Caption Augmentation using Large Language Model](https://github.com/seungheondoh/llm-tag-to-caption)

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