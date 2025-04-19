from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import torch
import io
import soundfile as sf
import asyncio
import numpy as np
from transformers import AutoModel
import time

app = FastAPI()

# Load IndicF5 model and move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True).to(device)

# Warm-up (optional but recommended)
_ = model(
    "सिस्टम प्रारंभ किया जा रहा है।", 
    ref_audio_path="IndicF5/prompts/PAN_F_HAPPY_00001.wav", 
    ref_text="ਭਹੰਪੀ ਵਿੱਚ ਸਮਾਰਕਾਂ ਦੇ ਭਵਨ ਨਿਰਮਾਣ ਕਲਾ ਦੇ ਵੇਰਵੇ ਗੁੰਝਲਦਾਰ ਅਤੇ ਹੈਰਾਨ ਕਰਨ ਵਾਲੇ ਹਨ, ਜੋ ਮੈਨੂੰ ਖੁਸ਼ ਕਰਦੇ  ਹਨ।"
)

async def stream_audio(text: str):
    # Generate waveform with reference audio & text (required for IndicF5)
    start = time.time()
    print(text)
    audio = model(
        text,
        ref_audio_path="IndicF5/prompts/PAN_F_HAPPY_00001.wav",
        ref_text="ਭਹੰਪੀ ਵਿੱਚ ਸਮਾਰਕਾਂ ਦੇ ਭਵਨ ਨਿਰਮਾਣ ਕਲਾ ਦੇ ਵੇਰਵੇ ਗੁੰਝਲਦਾਰ ਅਤੇ ਹੈਰਾਨ ਕਰਨ ਵਾਲੇ ਹਨ, ਜੋ ਮੈਨੂੰ ਖੁਸ਼ ਕਰਦੇ  ਹਨ।"
    )

    # Normalize if needed
    if isinstance(audio, np.ndarray) and audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0

    # Stream WAV bytes
    with io.BytesIO() as wav_io:
        sf.write(wav_io, np.array(audio, dtype=np.float32), 24000, format='WAV')
        wav_io.seek(0)
        while chunk := wav_io.read(2048):
            yield chunk
            await asyncio.sleep(0)
    latency = time.time() - start
    print(f"⏱️ TTS Latency: {latency:.2f}s")


@app.post("/api/tts")
async def tts_endpoint(request: Request):
    data = await request.json()
    text = data.get("text", "")
    return StreamingResponse(stream_audio(text), media_type="audio/wav")

