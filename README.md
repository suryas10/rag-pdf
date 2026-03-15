# RAG PDF System

A complete Retrieval-Augmented Generation (RAG) system for PDF documents with text and image embedding support.

## Features

- 📄 **PDF Text Extraction**: Extract and process text from PDF documents
- 🖼️ **Image Extraction**: Extract images from PDF pages for vision-based retrieval
- 🧠 **Nomic Embeddings**: Text embeddings using `nomic-embed-text-v1.5` and vision embeddings using `nomic-embed-vision-v1.5`
- 🗄️ **Qdrant Vector Store**: Local persistent vector database for storing embeddings
- 🔍 **Hybrid Retrieval**: Context-aware retrieval with coreference resolution and intent classification
- 🤖 **Grok LLM**: Generate responses using Grok Reasoner model
- 💬 **Streamlit UI**: User-friendly interface for uploading PDFs and chatting with documents

## Setup

### 1. Activate Virtual Environment

```bash
cd rag-pdf
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

### 2. Install Dependencies

```bash
python -m pip install --upgrade pip setuptools wheel

pip install -r requirements.txt

```

### 3. Set Environment Variables

Set your Grok API key:

```bash
# Windows PowerShell
$env:GROQ_API_KEY="your-api-key-here"

```

Or create a `.env` file (not included in repo):

```
GROQ_API_KEY=your-api-key-here
```

### 4. Configure Settings

Edit `config.yaml` to adjust:
- Embedding models and batch sizes
- Chunk sizes and overlaps
- Vector store settings
- LLM parameters

## Usage

### Start the FastAPI Backend

```bash
python fastapi_server.py
```

The server will start on `http://localhost:8000` by default.

### Start the Streamlit UI

In a new terminal (with venv activated):

```bash
streamlit run app.py
```

The UI will open in your browser at `http://localhost:8501`.

### Workflow

1. **Upload PDF**: Use the sidebar to upload a PDF file
2. **Wait for Ingestion**: The system will extract text/images, generate embeddings, and store them in Qdrant
3. **Chat**: Ask questions about your document in the chat interface
4. **View Sources**: See which parts of the document were used to answer your question

## API Endpoints

### Upload File
```
POST /upload
Content-Type: multipart/form-data
Body: PDF file
Response: { "job_id": "...", "filename": "...", "status": "processing" }
```

### Check Ingestion Status
```
GET /ingestion/status/{job_id}
Response: { "status": "...", "progress": 0.0-1.0, "message": "..." }
```

### Query Documents
```
POST /query
Content-Type: application/json
Body: {
  "query": "Your question",
  "file_id": "optional-file-id",
  "use_coref": true,
  "use_intent": true,
  "use_history": true,
  "use_multimodal": false,
  "image_query_base64": null,
  "top_k": 5
}
Response: {
  "response": "AI response",
  "sources": [...],
  "intent": "qa",
  "resolved_query": "..."
}
```

### Stream Query
```
POST /query/stream
Content-Type: application/json
Body: same as /query
Response: NDJSON stream with token events and a final summary event
```

### Clear Index
```
POST /index/clear
Response: { "status": "cleared" }
```

### Coreference Resolution
```
POST /coref
Content-Type: application/json
Body: {
  "text": "Text with pronouns",
  "context": "Optional context"
}
Response: { "resolved_text": "..." }
```

### Health Check
```
GET /health
Response: { "status": "healthy", "components": {...} }
```

## Project Structure

```
rag-pdf/
├── app.py                      # Streamlit UI
├── fastapi_server.py           # FastAPI backend
├── config.yaml                 # Configuration
├── requirements.txt            # Dependencies
├── backend/
│   ├── ingestion/
│   │   ├── text_extractor.py      # PDF text extraction
│   │   ├── image_extractor.py     # PDF image extraction
│   │   └── cleaner_chunker.py     # Text cleaning and chunking
│   ├── embeddings/
│   │   ├── nomic_text_embed.py    # Text embeddings
│   │   └── nomic_vision_embed.py  # Image embeddings
│   ├── vectorstore/
│   │   ├── qdrant_client.py       # Qdrant operations
│   │   └── retriever.py           # Hybrid retrieval
│   ├── coref_intent/
│   │   ├── coref_resolver.py      # Coreference resolution
│   │   └── intent_classifier.py   # Intent classification
│   └── llm/
│       └── Grok_inference.py  # Grok LLM integration
└── qdrant_local/              # Qdrant data (created automatically)
```

## Configuration

Key settings in `config.yaml`:

- **embedding**: Model names, batch sizes, chunk parameters
- **vectorstore**: Collection name, path, vector dimensions
- **llm**: Model, temperature, max tokens
- **retrieval**: Top-k results, similarity threshold

## Notes

- The system uses local Qdrant storage (no external database needed)
- Text embeddings use matryoshka dimension reduction (512 dim)
- Supports both text and image-based retrieval
- Conversation history is maintained in session state

## Troubleshooting

1. **API Connection Error**: Ensure FastAPI server is running
2. **Model Download Issues**: Check internet connection for model downloads
3. **Poppler Not Found**: Install poppler-utils for PDF image extraction
4. **Out of Memory**: Reduce batch sizes in config.yaml
5. **Grok API Error**: Verify API key is set correctly


## Validate setup
```
pip check
python - <<'PY'
import torch
print("✅ CUDA available:", torch.cuda.is_available(), "CUDA version:", torch.version.cuda)
PY
```

## License

This project is provided as-is for educational and development purposes.



