import whisper

# Load sekali saat modul diimport
# Pilihan: tiny, base, small, medium, large-v3
_model = whisper.load_model("small")


def transcribe_file(path: str, language: str = "id") -> str:
    """
    Transkrip satu berkas audio (mp3/wav) menjadi teks.
    """
    result = _model.transcribe(path, language=language, fp16=False)
    return result.get("text", "").strip()
