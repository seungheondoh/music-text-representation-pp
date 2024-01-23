# Text-to-Music Retrieval++ 

This is a implementation of [TTMR++](#)(Enriching Music Descriptions with a Finetuned-LLM and Metadata for Text-to-Music Retrieval). This project aims to search music with text query. 
<!-- 
> [**Enriching Music Descriptions with a Finetuned-LLM and Metadata for Text-to-Music Retrieval**](#)

> SeungHeon Doh, Minhee Lee, Dasaem Jeong and Juhan Nam 
> IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) 2024 
 -->

## TL;DR

Update Soon

## Open Source Material
- [Pre-trained model weights](https://huggingface.co/seungheondoh/ttmr-pp/tree/main) 

## Installation

## Quick Start

## Re-Implementation

### License
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