import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.1:8b"

SYSTEM_PROMPT = (
    "Kamu adalah Asisten Desa yang menjawab singkat, akurat, ramah. "
    "Gunakan Bahasa Indonesia sehari-hari. Jika tidak yakin, katakan "
    '"Saya belum punya datanya" dan sarankan langkah selanjutnya.'
)


def generate_llm_reply(
    prompt: str, temperature: float = 0.3, num_ctx: int = 8192
) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "options": {"temperature": temperature, "num_ctx": num_ctx},
        "stream": False,
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()

    # Prefer chat response field, fall back to legacy format if present.
    message = data.get("message", {})
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = ""

    if not content:
        content = data.get("response", "")

    return content.strip()
