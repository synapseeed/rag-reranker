import os
import time
from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = FastAPI(title="RAG Reranker", version="1.0.0")

MODEL_NAME = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
MAX_LENGTH = int(os.getenv("RERANK_MAX_LENGTH", "512"))
BATCH_SIZE = int(os.getenv("RERANK_BATCH_SIZE", "16"))
MAX_DOCUMENTS = int(os.getenv("RERANK_MAX_DOCUMENTS", "100"))
MAX_DOC_CHARS = int(os.getenv("RERANK_MAX_DOC_CHARS", "5000"))
PRIOR_WEIGHT = float(os.getenv("RERANK_PRIOR_WEIGHT", "0.2"))
MODEL_WEIGHT = float(os.getenv("RERANK_MODEL_WEIGHT", "0.8"))

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"🚀 Reranker Service Loading: {MODEL_NAME} on {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
model.eval()


class RerankRequest(BaseModel):
    query: str = Field(..., min_length=1)
    documents: List[str] = Field(default_factory=list)
    prior_scores: Optional[List[float]] = None
    use_score_fusion: bool = True
    top_k: Optional[int] = None


class RerankResponse(BaseModel):
    scores: List[float]
    raw_scores: List[float]
    ranked_indices: List[int]
    count: int
    trimmed: bool
    time_ms: float
    model: str
    device: str


@app.on_event("startup")
def warmup() -> None:
    try:
        _score_pairs([["warmup query", "warmup document"]])
        print("✅ Reranker warmup complete")
    except Exception as exc:
        print(f"⚠️ Warmup failed: {exc}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": str(device),
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "max_documents": MAX_DOCUMENTS,
    }


@app.get("/meta")
def meta():
    return {
     "rerank_model": MODEL_NAME,
        "device": str(device),
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "max_documents": MAX_DOCUMENTS,
        "max_doc_chars": MAX_DOC_CHARS,
        "fusion": {
            "enabled_by_default": True,
            "model_weight": MODEL_WEIGHT,
            "prior_weight": PRIOR_WEIGHT,
        },
    }


@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest):
    if not req.documents:
        return RerankResponse(
            scores=[],
            raw_scores=[],
            ranked_indices=[],
            count=0,
            trimmed=False,
            time_ms=0.0,
            model=MODEL_NAME,
            device=str(device),
        )

    start_time = time.time()
    documents, trimmed = trim_documents(req.documents, MAX_DOCUMENTS)
    prepared_docs = [prepare_document(doc) for doc in documents]
    pairs = [[req.query, doc] for doc in prepared_docs]

    raw_scores = _score_pairs(pairs)

    final_scores = raw_scores.copy()
    if req.use_score_fusion and req.prior_scores:
        priors = align_and_normalize_priors(req.prior_scores, len(final_scores))
        final_scores = fuse_scores(raw_scores, priors)

    ranked_indices = np.argsort(-final_scores).tolist()

    if req.top_k is not None and req.top_k > 0:
        ranked_indices = ranked_indices[: req.top_k]

    duration_ms = (time.time() - start_time) * 1000
    max_score = float(np.max(final_scores)) if len(final_scores) > 0 else 0.0

    print(
        f"✅ Reranked {len(documents)} docs"
        f" | Max Score: {max_score:.4f}"
        f" | Trimmed: {trimmed}"
        f" | {duration_ms:.1f}ms"
    )

    return RerankResponse(
        scores=final_scores.tolist(),
        raw_scores=raw_scores.tolist(),
        ranked_indices=ranked_indices,
        count=len(documents),
        trimmed=trimmed,
        time_ms=duration_ms,
        model=MODEL_NAME,
        device=str(device),
    )


def _score_pairs(pairs: List[List[str]]) -> np.ndarray:
    all_scores: List[np.ndarray] = []

    with torch.inference_mode():
        for i in range(0, len(pairs), BATCH_SIZE):
            batch = pairs[i : i + BATCH_SIZE]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(device)

            logits = model(**inputs).logits.view(-1)
            scores = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
            all_scores.append(scores)

    if not all_scores:
        return np.array([], dtype=np.float32)

    return np.concatenate(all_scores)


def trim_documents(documents: List[str], max_documents: int) -> tuple[List[str], bool]:
    if len(documents) <= max_documents:
        return documents, False
    return documents[:max_documents], True


def prepare_document(document: str) -> str:
    if len(document) <= MAX_DOC_CHARS:
        return document

    head_chars = int(MAX_DOC_CHARS * 0.7)
    tail_chars = MAX_DOC_CHARS - head_chars
    return (
        document[:head_chars]
        + "\n...\n[TRUNCATED]\n...\n"
        + document[-tail_chars:]
    )


def align_and_normalize_priors(prior_scores: List[float], target_len: int) -> np.ndarray:
    trimmed = prior_scores[:target_len]
    if len(trimmed) < target_len:
        trimmed = trimmed + [0.0] * (target_len - len(trimmed))

    arr = np.array(trimmed, dtype=np.float32)
    if arr.size == 0:
        return arr

    min_val = float(arr.min())
    max_val = float(arr.max())

    if max_val - min_val < 1e-8:
        return np.zeros_like(arr)

    return (arr - min_val) / (max_val - min_val)


def fuse_scores(model_scores: np.ndarray, prior_scores: np.ndarray) -> np.ndarray:
    if prior_scores.size == 0 or model_scores.size == 0:
        return model_scores

    return (MODEL_WEIGHT * model_scores) + (PRIOR_WEIGHT * prior_scores)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
