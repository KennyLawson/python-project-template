# src/realtime/vad_stream.py
import webrtcvad


class VADSegmenter:
    """Potong audio berdasarkan suara vs hening.
    Input: PCM16 mono @16kHz dalam potongan 30ms.
    Output: daftar segmen (bytes) ketika bicara berakhir.
    """

    def __init__(self, sample_rate=16000, frame_ms=30, vad_level=3, silence_ms=500):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_bytes = int(sample_rate * (frame_ms / 1000.0) * 2)  # 16-bit mono
        self.vad = webrtcvad.Vad(vad_level)  # 0..3 (3 paling agresif)
        self.silence_frames_needed = max(1, int(silence_ms / frame_ms))
        self._buf = bytearray()
        self._seg = bytearray()
        self._voice = False
        self._silence_frames = 0

    def add_audio(self, pcm16: bytes):
        """Tambahkan bytes PCM16; kembalikan list segmen selesai (jika ada)."""
        results = []
        self._buf.extend(pcm16)
        while len(self._buf) >= self.frame_bytes:
            frame = self._buf[: self.frame_bytes]
            del self._buf[: self.frame_bytes]
            is_speech = self.vad.is_speech(frame, self.sample_rate)
            if is_speech:
                self._voice = True
                self._silence_frames = 0
                self._seg.extend(frame)
            else:
                if self._voice:
                    self._silence_frames += 1
                    self._seg.extend(frame)
                    if self._silence_frames >= self.silence_frames_needed:
                        results.append(bytes(self._seg))
                        self._seg.clear()
                        self._voice = False
                        self._silence_frames = 0
        return results

    def flush(self):
        if self._seg:
            out = bytes(self._seg)
            self._seg.clear()
            self._voice = False
            self._silence_frames = 0
            return [out]
        return []
