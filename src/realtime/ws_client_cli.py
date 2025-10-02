# src/realtime/ws_client_cli.py
import base64
import json
import queue
import threading
import tempfile
import os

import numpy as np
import sounddevice as sd
import websocket
from websocket import WebSocketConnectionClosedException

WS_URL = "ws://localhost:8080/ws"
SAMPLE_RATE = 16000
FRAME_MS = 30


def pcm_from_float(indata: np.ndarray) -> bytes:
    return (indata[:, 0] * 32767).astype(np.int16).tobytes()


def play_wav_bytes(wav_bytes: bytes):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        path = f.name
    exit_code = os.system(f'ffplay -nodisp -autoexit "{path}" >nul 2>&1')
    if exit_code != 0:
        print(f"[TTS] Audio saved at: {path}")


def main():
    ws = websocket.WebSocket()
    ws.connect(WS_URL)
    ws.send(json.dumps({"type": "start", "sample_rate": SAMPLE_RATE}))
    print("[client] Connected to server. Speak after 'Recording...' appears.")

    q = queue.Queue()
    stop_event = threading.Event()

    def audio_cb(indata, frames, t, status):
        if stop_event.is_set():
            raise sd.CallbackStop()
        q.put(pcm_from_float(indata))

    def sender():
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, callback=audio_cb
            ):
                print("[client] Recording... (Ctrl+C to stop)")
                while not stop_event.is_set():
                    try:
                        pcm = q.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    b64 = base64.b64encode(pcm).decode("ascii")
                    ws.send(json.dumps({"type": "audio", "data": b64}))
        except (WebSocketConnectionClosedException, ConnectionResetError):
            pass
        except sd.PortAudioError as exc:
            print(f"[client] Audio input error: {exc}")

    def receiver():
        while not stop_event.is_set():
            try:
                raw = ws.recv()
            except (WebSocketConnectionClosedException, ConnectionResetError):
                break
            if raw is None:
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type")
            if msg_type == "ready":
                print("[server] Ready.")
            elif msg_type == "ack":
                print(f"[server] {msg.get('msg')}")
            elif msg_type == "asr":
                print(f"[ASR] Kamu: {msg.get('text', '')}")
            elif msg_type == "answer":
                text = msg.get("text", "")
                print(f"[Agent] {text}")
                audio_b64 = msg.get("audio", "")
                if audio_b64:
                    play_wav_bytes(base64.b64decode(audio_b64))

        stop_event.set()
        ws.close()
        print("[client] Disconnected.")

    recv_thread = threading.Thread(target=receiver, daemon=True)
    recv_thread.start()
    try:
        sender()
    except KeyboardInterrupt:
        print("\n[client] Stop requested.")
    finally:
        stop_event.set()
        try:
            ws.send(json.dumps({"type": "stop"}))
        except Exception:
            pass
        ws.close()
        recv_thread.join(timeout=1)


if __name__ == "__main__":
    main()

