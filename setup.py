from setuptools import setup

setup(
    name="mtrpp",
    packages=["mtrpp"],
    install_requires=[
        'wget',
        'librosa >= 0.8',
        'numpy',
        'pandas',
        'einops',
        'wandb',
        'jupyter',
        'matplotlib',
        'omegaconf',
        'datasets',
        'transformers',
        'tokenizers',
        "evaluate",
        "tensorboard",
        "torchmetrics==1.2.1",
        "jsonlines==4.0.0"
    ]
)
