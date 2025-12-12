from contextlib import asynccontextmanager

import sys
import os

sys.path.append(os.path.dirname(__file__))
from io import BytesIO

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from numpy.typing import ArrayLike

from clean_speak_function import get_synth_speech
import asyncio
import torch

# os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"


def arr2stream(arr: ArrayLike) -> StreamingResponse:
    buffer = BytesIO()
    np.save(buffer, arr)
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="application/octet-stream")


# deload if model has not been used for 10 minutes
_INACTIVITY_TIME_UNTIL_MODEL_DELOADS_IN_SEC = 600
is_busy: bool = False
deload_model_task: asyncio.Task | None = None
speak = None

main_loop: asyncio.AbstractEventLoop | None = None


def _deload_model():
    global speak
    speak = None
    torch.cuda.empty_cache()
    print("Deloaded model from GPU memory.")


async def _set_deload_model_timer():
    await asyncio.sleep(_INACTIVITY_TIME_UNTIL_MODEL_DELOADS_IN_SEC)
    _deload_model()


# FastAPI functions -------------------------------------------------


@asynccontextmanager
async def lifespan(_: FastAPI):
    global main_loop
    # global speak
    # global deload_model_task
    # _load_model()

    # if deload_model_task is not None:
    #     deload_model_task.cancel()
    # deload_model_task = asyncio.create_task(_set_deload_model_timer())
    main_loop = asyncio.get_running_loop()

    yield
    # Clean up the ML models and release the resources
    # _deload_model()


app = FastAPI(lifespan=lifespan)


def _load_model():
    global speak
    global deload_model_task

    _deload_model()
    speak = get_synth_speech("Vĩnh (nam miền Nam)", "cuda")

    print("Model loaded!")


@app.get("/test")
def t():
    print("test")
    return "test"


@app.get("/tts")
def tts(text: str):
    global is_busy
    global deload_model_task
    global speak

    if is_busy:
        print(f"Busy -> denying inference for '{text}'.")
        raise HTTPException(503)

    is_busy = True

    if deload_model_task is not None:
        deload_model_task.cancel()

    if speak is None:
        _load_model()

    if speak is None:
        raise Exception("Speak is still none")

    print(f"Starting inference for '{text}'...")
    arr = speak(text)
    is_busy = False
    # deload_model_task = asyncio.create_task(_set_deload_model_timer())
    if main_loop is not None:
        deload_model_task = main_loop.create_task(_set_deload_model_timer())

    print(f"Finished inference for '{text}'.")
    return arr2stream(arr)
