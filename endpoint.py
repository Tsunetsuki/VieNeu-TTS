import sys
import os

sys.path.append(os.path.dirname(__file__))
from io import BytesIO

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from numpy.typing import ArrayLike

from clean_speak_function import get_synth_speech

import torch


app = FastAPI()


def arr2stream(arr: ArrayLike) -> StreamingResponse:
    buffer = BytesIO()
    np.save(buffer, arr)
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="application/octet-stream")


# torch.cuda.empty_cache()
# speak = get_synth_speech("Nam 1 (id_0001)", "cuda")

is_processing: bool


def _load_model():
    global speak
    global is_processing
    speak = None
    is_processing = False
    torch.cuda.empty_cache()
    print("Cleared CUDA cache.")
    speak = get_synth_speech("Vĩnh (nam miền Nam)", "cuda")
    print("Model loaded once!")


@app.on_event("startup")
def startup():
    _load_model()


@app.get("/reload_model")
def reload_model():
    _load_model()


@app.get("/tts")
def tts(text: str):
    global is_processing
    if is_processing or speak is None:
        print(f"Busy -> denying inference for '{text}'.")
        raise HTTPException(503)

    print(f"Starting inference for '{text}'...")
    is_processing = True
    arr = speak(text)
    is_processing = False

    print(f"Finished inference for '{text}'.")
    return arr2stream(arr)
