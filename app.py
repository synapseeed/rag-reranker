from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

class RerankRequest(BaseModel):
    query: str
    documents: list[str]

@app.post("/rerank")
def rerank(req: RerankRequest):
    pairs = [(req.query, doc) for doc in req.documents]

    inputs = tokenizer(
        pairs,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        scores = model(**inputs).logits.squeeze().tolist()

    return {"scores": scores}