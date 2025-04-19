from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from kokoro import KPipeline
import soundfile as sf
import io
import asyncio
import time
import os

app = FastAPI()

LANG_CODE = os.getenv("LANG_CODE", "a")
VOICE = os.getenv("VOICE", "af_heart")

# Initialize Kokoro TTS pipeline
pipeline = KPipeline(lang_code=LANG_CODE)

# Warm-up call
_ = list(pipeline("Hello, just warming up!", voice="af_heart"))

async def stream_audio(text: str):
    print(text)
    start = time.perf_counter()
    generator = pipeline(text, voice=VOICE)
    for i, (_, _, audio) in enumerate(generator):
        with io.BytesIO() as wav_io:
            sf.write(wav_io, audio, 24000, format='WAV')
            wav_io.seek(0)
            while chunk := wav_io.read(2048):
                yield chunk
                await asyncio.sleep(0)
    generation_end = time.perf_counter()
    print(f"[Latency] Total generation time: {generation_end - start:.3f} sec")


@app.post("/api/tts")
async def tts_endpoint(request: Request):
    data = await request.json()
    text = data.get("text", "")
    return StreamingResponse(stream_audio(text), media_type="audio/wav")

