from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from TTS.api import TTS
import torch
import io
import soundfile as sf
import asyncio

app = FastAPI()

# Initialize TTS (VITS model is fast + supports streaming)
tts = TTS(model_name="tts_models/en/ljspeech/vits",
          gpu=torch.cuda.is_available())

_ = tts.tts(text="Warm up Text",split_sentences=True)

async def stream_audio(text: str):
    # Generate waveform using TTS
    waveform = tts.tts(text=text,split_sentences=True)
    # Save to bytes buffer as WAV
    with io.BytesIO() as wav_io:
        sf.write(wav_io, waveform, 22050, format='WAV')
        wav_io.seek(0)
        while chunk := wav_io.read(2048):
            yield chunk
            await asyncio.sleep(0)  # Yield control to event loop

@app.post("/api/tts")
async def tts_endpoint(request: Request):
    data = await request.json()
    text = data.get("text", "")
    return StreamingResponse(stream_audio(text), media_type="audio/wav")