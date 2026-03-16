# WiM RAG PDF System

An end-to-end Retrieval-Augmented Generation (RAG) system for PDF understanding with:

- text extraction and semantic chunking
- embedded image extraction and image embeddings
- local Qdrant vector database
- streaming chat responses over FastAPI
- persistent chat sessions (SQLite)
- two UIs: React frontend and Streamlit control panel

## Architecture Overview

1. Upload PDF
2. Ingestion pipeline extracts text + images
3. Text and image embeddings are generated
4. Vectors are stored in Qdrant with metadata
5. Query retrieval returns relevant text/image context
6. LLM generates grounded response with sources

Core backend entrypoint: `fastapi_server.py`

## Main Features

- PDF text extraction with page metadata
- PDF image extraction using PyMuPDF
- Nomic text and vision embedding support
- Hybrid retrieval pipeline with reranking hooks
- Query streaming endpoint (`/chat/query/stream`, NDJSON)
- Multi-session chat management APIs
- Local LLM or cloud LLM provider mode (configurable)

## Project Layout

```text
rag-pdf/
  fastapi_server.py            # FastAPI server and API routes
  config.yaml                  # Runtime configuration
  app.py                       # Streamlit app
  frontend/                    # React + Vite frontend
  backend/
    ingestion/                 # text/image extraction + chunking
    embeddings/                # text/vision embedders
    vectorstore/               # qdrant + retriever + reranker
    coref_intent/              # coref + intent classifiers
    llm/                       # LLM inference wrapper
    memory/                    # conversation memory
    session_store.py           # SQLite chat persistence
  ui/                          # Streamlit UI components/client
  scripts/start_demo.ps1       # one-click demo launcher
```

## Prerequisites

- Python 3.11.x
- Node.js 18+ (for React frontend)
- Optional NVIDIA GPU with CUDA-compatible torch build
- Optional local llama.cpp server binary/model (if using `llm.provider: local`)

## Setup

### 1) Python environment

```powershell
cd rag-pdf
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements_fixed.txt
```

### 2) Frontend dependencies

```powershell
cd frontend
npm install
cd ..
```

### 3) Environment variables

If using cloud provider mode, set:

```powershell
$env:GROQ_API_KEY="your-api-key-here"
```

Or create `.env`:

```env
GROQ_API_KEY=your-api-key-here
API_BASE_URL=http://localhost:8000
```

### 4) Configure runtime

Edit `config.yaml`:

- `llm.provider`: `local` or `groq`
- `embedding` chunk + vector settings
- `retrieval` top-k and thresholds
- `vectorstore.path` for local persistence

## Run Options

### Option A: FastAPI + React (recommended app UI)

Terminal 1:

```powershell
cd rag-pdf
venv\Scripts\activate
python fastapi_server.py
```

Terminal 2:

```powershell
cd rag-pdf\frontend
npm run dev
```

### Option B: FastAPI + Streamlit (control/debug UI)

Terminal 1:

```powershell
cd rag-pdf
venv\Scripts\activate
python fastapi_server.py
```

Terminal 2:

```powershell
cd rag-pdf
venv\Scripts\activate
streamlit run app.py
```

### Option C: One-click demo script (Windows)

```powershell
cd rag-pdf
powershell -ExecutionPolicy Bypass -File .\scripts\start_demo.ps1
```

## API Summary

### Health

- `GET /health`

### Upload + ingestion

- `POST /upload`
- `POST /document/upload`
- `GET /ingestion/status/{job_id}`

### Query

- `POST /query`
- `POST /query/stream`
- `POST /chat/query`
- `POST /chat/query/stream`

### Session management

- `POST /chat/create`
- `POST /chat/delete`
- `GET /chat/history?chat_id=...`
- `GET /chat/sessions`

### Reset/maintenance

- `POST /index/clear`
- `POST /reset`
- `POST /db/clear`
- `POST /conversation/clear`

### Utility

- `POST /coref`

## Typical Workflow

1. Create/select a chat session
2. Upload PDF
3. Poll ingestion until completed
4. Ask questions
5. Receive streaming answer + source chunks
6. Continue multi-turn conversation with persisted history

## Notes

- Vectors are stored locally in `qdrant_local/`
- Extracted images are stored in `data/images/`
- Chat metadata/history is stored in `data/chat_sessions.db`
- `.gitignore` excludes generated runtime artifacts and frontend dependencies

## Troubleshooting

1. API offline: confirm `python fastapi_server.py` is running.
2. Frontend can't connect: check `VITE_API_BASE_URL` and CORS.
3. No GPU: system falls back to CPU; reduce batch sizes if needed.
4. LLM errors: verify provider settings in `config.yaml` and API key.
5. Upload stuck: inspect `/ingestion/status/{job_id}` and backend logs.

## Validation

```powershell
pip check
python cuda_check.py
```

## License

Provided as-is for academic and development use.



