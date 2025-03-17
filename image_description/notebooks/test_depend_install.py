try:
    import torch
    import torchvision
    import transformers
    import accelerate
    import peft
    import tokenizers
    import gradio
    import pandas
    import scipy
    import einops
    import sentencepiece
    import tiktoken
    import google.protobuf
    import uvicorn
    import pydantic
    import fastapi
    import sse_starlette
    import matplotlib
    import fire
    import packaging
    import yaml
    import numpy
    import av
    import librosa
    import tyro
    print("All dependencies imported successfully!")
except Exception as e:
    print("Error importing dependencies:", e)
