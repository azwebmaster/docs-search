from __future__ import annotations

from functools import lru_cache

import numpy as np

from docs_search.config import DEFAULT_EMBED_MODEL


@lru_cache(maxsize=2)
def _get_model(model_name: str):
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=model_name)


def embed_texts(
    texts: list[str],
    *,
    model_name: str = DEFAULT_EMBED_MODEL,
    batch_size: int = 32,
) -> np.ndarray:
    """Embed texts with a local ONNX model. Returns L2-normalized float32 matrix."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    model = _get_model(model_name)
    vectors = list(model.embed(texts, batch_size=batch_size))
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return matrix / norms


def embed_query(query: str, *, model_name: str = DEFAULT_EMBED_MODEL) -> np.ndarray:
    return embed_texts([query], model_name=model_name)[0]
