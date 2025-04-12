from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import torch
import io
import soundfile as sf
import asyncio
import time
from torch.serialization import add_safe_globals

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Allow unpickling of XttsConfig
add_safe_globals([XttsConfig])

# Load XTTS model
print("🔄 Loading XTTS model...")
config = XttsConfig()
config.load_json("/path/to/xtts/config.json")  # 🔁 Update this path

model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="/path/to/xtts/", use_deepspeed=True)  # 🔁 Update this path
model.cuda()

# Optionally compute speaker latents once if using a fixed reference voice
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=["reference.wav"])  # 🔁 Replace with your reference file

app = FastAPI()

async def stream_audio(text: str):
    start = time.time()
    # Generate audio using XTTS model
    out = model.inference(
        text,
        # "hi",  # Set language code explicitly: 'hi' = Hindi, 'mr' = Marathi, 'en' = English
        gpt_cond_latent,
        speaker_embedding,
        temperature=0.75
    )
    # Stream the audio
    with io.BytesIO() as wav_io:
        sf.write(wav_io, torch.tensor(out["wav"]).unsqueeze(0), 24000, format="WAV")
        wav_io.seek(0)
        while chunk := wav_io.read(4096):
            yield chunk
            await asyncio.sleep(0)
    print(f"⏱️ Latency: {time.time() - start:.2f}s")

@app.post("/api/tts")
async def tts_endpoint(request: Request):
    data = await request.json()
    text = data.get("text", "")
    return StreamingResponse(stream_audio(text), media_type="audio/wav")
