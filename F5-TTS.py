from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import torch
import io
import soundfile as sf
import asyncio

from f5_tts.api import F5TTS

app = FastAPI()

# Path to your downloaded F5TTS model checkpoint
CHECKPOINT_PATH = "f5_hifi_multilingual.pt"

# Instantiate F5TTS with vocoder 'vocos' (from huggingface hub charactr/vocos-mel-24khz)
f5 = F5TTS(
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Warmup
_ = f5.infer(
        ref_file = "amit.wav", 
        ref_text = "मनुष्य का सबसे अन्मोल रत्न होता है प्रयत्न। जिस इंसान में प्रयत्न करने की क्षमता है उसके आगे मुकदर भी झुकता है।",
        gen_text = "Warm-up line.",
             )

async def stream_audio(text: str, lang="hi", speaker="default"):
    waveform = f5.infer(
            ref_file = "amit.wav",
            ref_text= "मनुष्य का सबसे अन्मोल रत्न होता है प्रयत्न। जिस इंसान में प्रयत्न करने की क्षमता है उसके आगे मुकदर भी झुकता है।",
            gen_text=text,
            )
    # Save to a WAV stream
    with io.BytesIO() as wav_io:
        sf.write(wav_io, waveform, samplerate=24000, format='WAV')
        wav_io.seek(0)
        while chunk := wav_io.read(2048):
            yield chunk
            await asyncio.sleep(0)

@app.post("/api/tts")
async def tts_endpoint(request: Request):
    data = await request.json()
    text = data.get("text", "")
    lang = data.get("lang", "hi")
    speaker = data.get("speaker", "default")

    return StreamingResponse(stream_audio(text, lang, speaker), media_type="audio/wav")

