# src/realtime/ws_server.py
import asyncio, base64, io, wave, tempfile
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from .vad_stream import VADSegmenter
from ..asr_whisper import transcribe_file
from ..rag_query import ask_with_rag
from ..tts_piper import synth_bytes

router = APIRouter()

SAMPLE_RATE = 16000


def _pcm_to_wav_path(pcm: bytes, sr: int = SAMPLE_RATE) -> str:
    """Tulis bytes PCM16 mono ke file WAV sementara; kembalikan path."""
    tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm)
    return tf.name


@router.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    await ws.send_json({"type": "ready", "sample_rate": SAMPLE_RATE})
    vad = VADSegmenter(
        sample_rate=SAMPLE_RATE, frame_ms=30, vad_level=2, silence_ms=600
    )

    loop = asyncio.get_running_loop()

    async def process_segment(seg_bytes: bytes):
        # 1) ASR
        wav_path = _pcm_to_wav_path(seg_bytes, SAMPLE_RATE)
        text = await loop.run_in_executor(None, transcribe_file, wav_path, "id")
        await ws.send_json({"type": "asr", "text": text})

        # 2) LLM+RAG
        answer = await loop.run_in_executor(None, ask_with_rag, text)

        # 3) TTS → kirim balik audio
        audio_wav = await loop.run_in_executor(None, synth_bytes, answer)
        audio_b64 = base64.b64encode(audio_wav).decode("ascii") if audio_wav else ""
        await ws.send_json({"type": "answer", "text": answer, "audio": audio_b64})

    try:
        while True:
            msg = await ws.receive_json()
            mtype = msg.get("type")
            if mtype == "start":
                await ws.send_json({"type": "ack", "msg": "start"})
            elif mtype == "audio":
                pcm = base64.b64decode(msg["data"])
                segments = vad.add_audio(pcm)
                for seg in segments:
                    # proses tiap segmen tanpa blokir
                    asyncio.create_task(process_segment(seg))
            elif mtype == "stop":
                for seg in vad.flush():
                    asyncio.create_task(process_segment(seg))
                await ws.send_json({"type": "ack", "msg": "stopped"})
                break
    except WebSocketDisconnect:
        pass
