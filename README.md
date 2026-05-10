# RAG Reranker Service

A high-performance reranking service designed for Retrieval-Augmented Generation (RAG) pipelines. This service uses Cross-Encoder models to provide precise relevance scores for a set of documents relative to a query, significantly improving retrieval accuracy over standard vector search or BM25.

## 🚀 Features

- **Cross-Encoder Reranking**: Utilizes state-of-the-art transformer models (defaults to `cross-encoder/ms-marco-MiniLM-L-6-v2`).
- **Hardware Acceleration**: Automatic detection and utilization of **CUDA** (NVIDIA), **MPS** (Apple Silicon), or CPU.
- **Score Fusion**: Integrates prior retrieval scores (e.g., from vector search) with model scores using weighted fusion.
- **Dynamic Truncation**: Handles long documents and large batches efficiently with configurable limits.
- **FastAPI Core**: Low-latency, asynchronous API handling.
- **Batch Processing**: Optimized inference through batching.

## 🛠 Tech Stack

- **Python 3.14+**
- **FastAPI** & **Uvicorn**
- **PyTorch**
- **Hugging Face Transformers**
- **NumPy**
- **Pydantic**

## 📥 Installation

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd rag-reranker
   ```

2. **Create a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Configuration**:
   Create a `.env` file (or use the provided one) to configure the service:
   ```env
   RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
   RERANK_MAX_LENGTH=512
   RERANK_BATCH_SIZE=16
   RERANK_MAX_DOCUMENTS=100
   RERANK_MAX_DOC_CHARS=5000
   RERANK_MODEL_WEIGHT=0.8
   RERANK_PRIOR_WEIGHT=0.2
   RERANK_OUTPUT_ACTIVATION=sigmoid
   RERANK_LOG_LEVEL=INFO
   ```

## 🏃 Usage

### Starting the Service

Run the service using Uvicorn:

```bash
uvicorn app:app --port 8001
```

The service will be available at `http://0.0.0.0:8001`.

### API Endpoints

#### 1. Rerank Documents

**POST** `/rerank`

Refines the ranking of a list of documents based on a query.

**Request Body**:

```json
{
  "query": "What is the capital of France?",
  "documents": [
    "Paris is the capital and most populous city of France.",
    "The Eiffel Tower is located in Paris.",
    "Lyon is a major city in France."
  ],
  "prior_scores": [0.9, 0.8, 0.7],
  "use_score_fusion": true,
  "top_k": 5
}
```

**Response**:

```json
{
  "scores": [0.98, 0.45, 0.12],
  "raw_scores": [0.97, 0.4, 0.1],
  "ranked_indices": [0, 1, 2],
  "count": 3,
  "trimmed": false,
  "time_ms": 45.2,
  "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "device": "cuda"
}
```

#### 2. Health Check

**GET** `/health`

Returns the current status and basic configuration.

#### 3. Metadata

**GET** `/meta`

Returns detailed configuration including model parameters and fusion weights.

## ⚙️ Configuration Parameters

| Variable                   | Description                                         | Default                                |
| -------------------------- | --------------------------------------------------- | -------------------------------------- |
| `RERANK_MODEL`             | Hugging Face model identifier                       | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `RERANK_MAX_LENGTH`        | Max sequence length for the model                   | `512`                                  |
| `RERANK_BATCH_SIZE`        | Batch size for inference                            | `16`                                   |
| `RERANK_MAX_DOCUMENTS`     | Max number of documents per request                 | `100`                                  |
| `RERANK_MAX_DOC_CHARS`     | Max characters per document (truncated if exceeded) | `5000`                                 |
| `RERANK_MODEL_WEIGHT`      | Weight for model score in fusion                    | `0.8`                                  |
| `RERANK_PRIOR_WEIGHT`      | Weight for prior score in fusion                    | `0.2`                                  |
| `RERANK_OUTPUT_ACTIVATION` | `sigmoid` or `logit`                                | `sigmoid`                              |

## ⚖️ Score Fusion Logic

When `use_score_fusion` is enabled, the final score is calculated as:

```text
Final Score = (RERANK_MODEL_WEIGHT * model_score) + (RERANK_PRIOR_WEIGHT * normalized_prior_score)
```

Prior scores are automatically normalized to a 0-1 range before fusion.
