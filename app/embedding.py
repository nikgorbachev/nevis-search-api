"""BGE-small-en-v1.5 passage indexing and CPU inference."""

import logging
from functools import lru_cache
from itertools import batched
from threading import Lock

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from app.models import Document, DocumentChunk

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def tokenizer() -> Tokenizer:
    model_tokenizer = Tokenizer.from_file("/model/tokenizer.json")
    model_tokenizer.no_padding()
    model_tokenizer.no_truncation()
    return model_tokenizer


@lru_cache(maxsize=1)
def encoder() -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    return ort.InferenceSession("/model/model.onnx", options, providers=["CPUExecutionProvider"])


# Bound peak memory when concurrent requests share the small Compose VM.
inference_lock = Lock()


class EmbeddingUnavailable(RuntimeError):
    pass


def embed(texts: list[str]) -> list[np.ndarray]:
    # BGE uses the normalized first token of the last hidden layer.
    try:
        with inference_lock:
            session = encoder()
            encoded = tokenizer().encode_batch(texts)
            rows = [item.ids for item in encoded]
            width = min(512, max(map(len, rows)))
            ids = np.zeros((len(rows), width), dtype=np.int64)
            mask = np.zeros_like(ids)
            for i, row in enumerate(rows):
                if len(row) > width:
                    row = row[:width - 1] + [row[-1]]
                ids[i, :len(row)] = row
                mask[i, :len(row)] = 1
            inputs = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
            hidden = session.run(None, {i.name: inputs[i.name] for i in session.get_inputs()})[0]
            vectors = hidden[:, 0, :]
            return list(vectors / np.linalg.norm(vectors, axis=1, keepdims=True))
    except Exception as error:
        logging.getLogger("uvicorn.error").error("Embedding failure (%s)", type(error).__name__)
        raise EmbeddingUnavailable("Embedding service unavailable") from error


def passage_bounds(content: str):
    offsets = tokenizer().encode(content, add_special_tokens=False).offsets
    start = 0
    # Preserve original character offsets. Prefer a complete paragraph or form row.
    while start + 180 < len(offsets):
        stop = start + 180
        for i in range(stop, start + 95, -1):
            if "\n" in content[offsets[i - 1][1]:offsets[i][0]]:
                stop = i
                break
        yield offsets[start][0], offsets[stop - 1][1]
        start = stop if stop < start + 180 else stop - 20
    yield offsets[start][0] if start else 0, len(content)


def _passages(documents):
    for document in documents:
        title_tokens = tokenizer().encode(document.title, add_special_tokens=False).offsets
        title = document.title[:title_tokens[63][1]] if len(title_tokens) > 64 else document.title
        for ordinal, (start, end) in enumerate(passage_bounds(document.content)):
            chunk = DocumentChunk(document_id=document.id, ordinal=ordinal, start_char=start, end_char=end)
            yield chunk, f"{title}\n{document.content[start:end]}"


def index_documents(documents: list[Document]):
    for batch in batched(_passages(documents), 16):
        chunks, texts = zip(*batch)
        for chunk, vector in zip(chunks, embed(list(texts)), strict=True):
            chunk.embedding = vector
            yield chunk
