import subprocess, tempfile, os, sys, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPER_BIN = ROOT / ("piper/piper.exe" if os.name == "nt" else "piper/piper")
VOICE_DIR = PIPER_BIN.parent
VOICE = VOICE_DIR / "id_ID-news_tts-medium.onnx"
CONFIG = VOICE_DIR / "id_ID-news_tts-medium.onnx.json"


def speak(text: str):
    if not (PIPER_BIN.exists() and VOICE.exists() and CONFIG.exists()):
        print(
            "[TTS] Piper/voice files not found. Skipping audio output.", file=sys.stderr
        )
        return
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
        wav_path = wf.name
    cmd = [str(PIPER_BIN), "-m", str(VOICE), "-c", str(CONFIG), "-f", wav_path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    p.communicate(input=text.encode("utf-8"))

    player = shutil.which("ffplay")
    if player:
        os.system(
            f'ffplay -nodisp -autoexit "{wav_path}" >nul 2>&1'
            if os.name == "nt"
            else f'ffplay -nodisp -autoexit "{wav_path}" >/dev/null 2>&1'
        )
    else:
        print(f"[TTS] Audio saved at: {wav_path}")


def synth_bytes(text: str) -> bytes:
    """Synthesize ke WAV bytes (tanpa memutar)."""
    if not (PIPER_BIN.exists() and VOICE.exists() and CONFIG.exists()):
        print("[TTS] Piper/voice files not found. Skipping audio output.")
        return b""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
        wav_path = wf.name
    cmd = [str(PIPER_BIN), "-m", str(VOICE), "-c", str(CONFIG), "-f", wav_path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    p.communicate(input=text.encode("utf-8"))
    data = Path(wav_path).read_bytes()
    try:
        os.remove(wav_path)
    except:
        pass
    return data
