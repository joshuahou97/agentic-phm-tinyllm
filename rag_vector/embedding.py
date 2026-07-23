from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request


OPENAI_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"


def require_api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        try:
            from config_local import LLM_API_KEY

            api_key = LLM_API_KEY
        except ImportError:
            api_key = None
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Run:\n"
            'export OPENAI_API_KEY="your_api_key"\n'
            "or create rag_vector/config_local.py with LLM_API_KEY."
        )
    return api_key


def _post_json(url: str, payload: dict, api_key: str, timeout: int = 120) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI API request failed: {exc}") from exc


def embed_texts(texts: list[str], model: str, batch_size: int = 64) -> list[list[float]]:
    api_key = require_api_key()
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        payload = {"model": model, "input": batch}
        result = _post_json(OPENAI_EMBEDDINGS_URL, payload, api_key)
        embeddings.extend(item["embedding"] for item in result["data"])
        time.sleep(0.05)
    return embeddings


def chat_completion(prompt: str, model: str) -> str:
    api_key = require_api_key()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    result = _post_json(OPENAI_CHAT_URL, payload, api_key)
    return result["choices"][0]["message"]["content"]
