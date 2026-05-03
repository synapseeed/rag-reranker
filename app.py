import torch
import time
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()

# Optimized for local CPU/GPU inference speed
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Device detection (CUDA > MPS > CPU)
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
    query: str
    documents: list[str]

@app.post("/rerank")
def rerank(req: RerankRequest):
    if not req.documents:
        return {"scores": [], "time_ms": 0}

    start_time = time.time()
    pairs = [[req.query, doc] for doc in req.documents]

    inputs = tokenizer(
        pairs,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)

    with torch.inference_mode():
        logits = model(**inputs).logits.view(-1)
        # Sigmoid: preserves absolute relevance signal
        scores = torch.sigmoid(logits).cpu().numpy()

    max_score = float(np.max(scores))
    duration_ms = (time.time() - start_time) * 1000
    print(f"✅ Reranked {len(req.documents)} docs | Max Score: {max_score:.4f} | {duration_ms:.1f}ms")

    return {
        "scores": scores.tolist(),
        "time_ms": duration_ms
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)