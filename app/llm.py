"""Thin wrapper over LiteLLM so the provider (Anthropic, OpenAI, local) is a config change."""

from __future__ import annotations

from app.config import get_settings


def embed(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    import litellm

    s = get_settings()
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        resp = litellm.embedding(model=s.embedding_model, input=texts[i : i + batch_size])
        vectors.extend(item["embedding"] for item in resp.data)
    return vectors


def chat(system: str, user: str, temperature: float = 0.0) -> str:
    import litellm

    s = get_settings()
    resp = litellm.completion(
        model=s.chat_model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""
