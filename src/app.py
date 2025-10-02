from fastapi import FastAPI
from pydantic import BaseModel
from .asr_whisper import transcribe_file
from .rag_query import ask_with_rag
from .tts_piper import speak
from .realtime.ws_server import router as ws_router

app = FastAPI(title="Desa Voice Agent (Whisper + Llama + RAG)")
app.include_router(ws_router)


class AskText(BaseModel):
    text: str


class VoiceChat(BaseModel):
    audio_path: str
    language: str = "id"


@app.post("/chat")
def chat(q: AskText):
    answer = ask_with_rag(q.text)
    speak(answer)
    return {"answer": answer}


@app.post("/voice_chat")
def voice_chat(v: VoiceChat):
    user_text = transcribe_file(v.audio_path, language=v.language)
    answer = ask_with_rag(user_text)
    speak(answer)
    return {"user_text": user_text, "answer": answer}
