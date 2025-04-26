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
pipeline_eng = KPipeline(lang_code="a")
pipeline_hin = KPipeline(lang_code="h")

# Warm-up call
_ = list(pipeline_eng("Hello, just warming up!", voice="af_heart"))
_ = list(pipeline_hin("Hello, just warming up!", voice="hm_psi"))

async def stream_audio(text: str, lang_code:str = LANG_CODE, voice: str = VOICE):
    print(text)
    start = time.perf_counter()
    if lang_code == "a":
        pipeline = pipeline_eng
    elif lang_code == "h":
        pipeline = pipeline_hin
    else:
        raise ValueError(f"Unsupported language code: {lang_code}")
    
    generator = pipeline(text, voice=voice)
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
    lang = data.get("language", "a")
    voice = data.get("voice", "am_michael")
    return StreamingResponse(stream_audio(text, lang, voice), media_type="audio/wav")

