from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from transformers import AutoModel, AutoProcessor
import torch
import numpy as np
import soundfile as sf
import io
import time
import asyncio

app = FastAPI()

# Load processor and custom Bark model
model_id = "canopylabs/3b-hi-ft-research_release"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def generate_audio(text):
    inputs = processor(text, return_tensors="pt").to(device)
    start = time.time()
    audio = model.generate(**inputs)
    latency = time.time() - start
    print(f"🕒 TTS Latency: {latency:.2f}s")
    audio = audio.cpu().numpy().squeeze()
    return audio, 24000

async def stream_audio(text):
    audio, sr = generate_audio(text)
    with io.BytesIO() as wav_io:
        sf.write(wav_io, audio, sr, format="WAV")
        wav_io.seek(0)
        while chunk := wav_io.read(2048):
            yield chunk
            await asyncio.sleep(0)

@app.post("/api/tts")
async def tts_endpoint(request: Request):
    data = await request.json()
    text = data.get("text", "")
    return StreamingResponse(stream_audio(text), media_type="audio/wav")

