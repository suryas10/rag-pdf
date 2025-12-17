"""
FastAPI backend server for RAG PDF system.
Handles file uploads, ingestion, embedding, and query processing.
"""

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import yaml
import os
import uuid
import asyncio
from pathlib import Path
import tempfile

# Backend imports
from backend.ingestion.text_extractor import extract_text_from_pdf
from backend.ingestion.image_extractor import extract_images_from_pdf
from backend.ingestion.cleaner_chunker import Chunker
from backend.embeddings.nomic_text_embed import NomicTextEmbedder
from backend.embeddings.nomic_vision_embed import NomicVisionEmbedder
from backend.vectorstore.qdrant_client import QdrantVectorStore
from backend.vectorstore.retriever import HybridRetriever
from backend.coref_intent.coref_resolver import CorefResolver
from backend.coref_intent.intent_classifier import IntentClassifier
from backend.llm.grok_inference import GroqInference

app = FastAPI(title="RAG PDF API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
config = {}
vector_store = None
text_embedder = None
vision_embedder = None
retriever = None
llm = None
ingestion_jobs = {}


def clear_memory_and_vectors(file_id: Optional[str] = None):
    """Clear retriever memory and optionally delete vectors by file."""
    if retriever:
        retriever.memory = []
    if file_id and vector_store:
        try:
            vector_store.delete_by_file_id(file_id)
        except Exception as e:
            print(f"Failed to delete vectors for file {file_id}: {e}")


def load_config():
    """Load configuration from config.yaml"""
    global config
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default config
        config = {
            "embedding": {
                "text_model": "nomic-ai/nomic-embed-text-v1.5",
                "vision_model": "nomic-ai/nomic-embed-vision-v1.5",
                "batch_size": 64,
                "matryoshka_dim": 512,
                "chunk_size": 500,
                "overlap": 50
            },
            "vectorstore": {
                "collection": "rag_chunks",
                "path": "./qdrant_local",
                "vector_size": 512
            },
            "llm": {
                "provider": "groq",
                "model": "llama-3.1-70b-versatile",
                "base_url": "https://api.groq.com/openai/v1",
                "temperature": 0.3,
                "max_tokens": 800
            },
            "retrieval": {
                "top_k": 3,
                "similarity_threshold": 0.5
            }
        }


def initialize_components():
    """Initialize all backend components"""
    global vector_store, text_embedder, vision_embedder, retriever, llm
    
    print("Initializing components...")
    
    # Load config
    load_config()
    
    # Initialize vector store
    vs_config = config.get("vectorstore", {})
    vector_store = QdrantVectorStore(
        collection_name=vs_config.get("collection", "rag_chunks"),
        path=vs_config.get("path", "./qdrant_local"),
        vector_size=vs_config.get("vector_size", 512)
    )
    
    # Initialize embedders
    embed_config = config.get("embedding", {})
    text_embedder = NomicTextEmbedder(
        model_name=embed_config.get("text_model", "nomic-ai/nomic-embed-text-v1.5"),
        matryoshka_dim=embed_config.get("matryoshka_dim", 512),
        batch_size=embed_config.get("batch_size", 64)
    )
    
    vision_embedder = NomicVisionEmbedder(
        model_name=embed_config.get("vision_model", "nomic-ai/nomic-embed-vision-v1.5"),
        batch_size=8
    )
    
    # Initialize coref and intent
    coref_resolver = CorefResolver()
    intent_classifier = IntentClassifier()
    
    # Initialize retriever
    retr_config = config.get("retrieval", {})
    retriever = HybridRetriever(
        vector_store=vector_store,
        text_embedder=text_embedder,
        coref_resolver=coref_resolver,
        intent_classifier=intent_classifier,
        top_k=retr_config.get("top_k", 3),
        similarity_threshold=retr_config.get("similarity_threshold", 0.7)
    )
    
    # Initialize LLM
    llm_config = config.get("llm", {})
    llm = GroqInference(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url=llm_config.get("base_url", "https://api.groq.com/openai/v1"),
        model=llm_config.get("model", "llama-3.1-70b-versatile"),
        temperature=llm_config.get("temperature", 0.2),
        max_tokens=llm_config.get("max_tokens", 800)
    )
    
    print("Components initialized successfully")


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    initialize_components()


# Request/Response models
class QueryRequest(BaseModel):
    query: str
    file_id: Optional[str] = None
    use_coref: bool = True
    use_intent: bool = True


class QueryResponse(BaseModel):
    response: str
    sources: List[Dict]
    intent: str
    resolved_query: str


class CorefRequest(BaseModel):
    text: str
    context: Optional[str] = None


class CorefResponse(BaseModel):
    resolved_text: str


class IngestionStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    message: str


class ClearConversationRequest(BaseModel):
    file_id: Optional[str] = None


async def process_ingestion(
    job_id: str,
    pdf_bytes: bytes,
    filename: str
):
    """Background task to process PDF ingestion"""
    try:
        ingestion_jobs[job_id] = {
            "status": "processing",
            "progress": 0.0,
            "message": "Starting ingestion..."
        }
        
        file_id = str(uuid.uuid4())

        # Clear any existing vectors so only the latest file remains
        try:
            vector_store.clear_collection()
        except Exception as e:
            print(f"Failed to clear collection before new ingestion: {e}")
        
        # Step 1: Extract text
        ingestion_jobs[job_id]["message"] = "Extracting text from PDF..."
        ingestion_jobs[job_id]["progress"] = 0.1
        text_pages = extract_text_from_pdf(pdf_bytes=pdf_bytes)
        
        # Step 2: Extract images
        ingestion_jobs[job_id]["message"] = "Extracting images from PDF..."
        ingestion_jobs[job_id]["progress"] = 0.2
        image_pages = extract_images_from_pdf(pdf_bytes=pdf_bytes)
        
        # Step 3: Chunk text
        ingestion_jobs[job_id]["message"] = "Chunking text..."
        ingestion_jobs[job_id]["progress"] = 0.3
        chunker_config = config.get("embedding", {})
        chunker = Chunker(
            chunk_size=chunker_config.get("chunk_size", 500),
            chunk_overlap=chunker_config.get("overlap", 50)
        )
        chunks = chunker.chunk_pages(text_pages)
        
        # Step 4: Generate text embeddings
        ingestion_jobs[job_id]["message"] = "Generating text embeddings..."
        ingestion_jobs[job_id]["progress"] = 0.5
        chunk_texts = [chunk.text for chunk in chunks]
        chunk_embeddings = text_embedder.encode(chunk_texts, show_progress=True)
        
        # Step 5: Generate image embeddings
        ingestion_jobs[job_id]["message"] = "Generating image embeddings..."
        ingestion_jobs[job_id]["progress"] = 0.7
        images = [page["image"] for page in image_pages]
        image_embeddings = vision_embedder.encode(images, show_progress=True)
        
        # Step 6: Upsert to vector store
        ingestion_jobs[job_id]["message"] = "Storing embeddings in vector database..."
        ingestion_jobs[job_id]["progress"] = 0.9
        
        # Prepare text chunk payloads
        text_payloads = []
        text_embeddings_list = []
        for chunk, embedding in zip(chunks, chunk_embeddings):
            text_payloads.append({
                "text": chunk.text,
                "file_id": file_id,
                "filename": filename,
                "page_no": chunk.page_no,
                "chunk_id": chunk.chunk_id,
                "start_offset": chunk.start_offset,
                "end_offset": chunk.end_offset,
                "type": "text"
            })
            text_embeddings_list.append(embedding.tolist())
        
        # Prepare image payloads
        image_payloads = []
        image_embeddings_list = []
        for page_data, embedding in zip(image_pages, image_embeddings):
            image_payloads.append({
                "file_id": file_id,
                "filename": filename,
                "page_no": page_data["page"],
                "type": "image"
            })
            image_embeddings_list.append(embedding.tolist())
        
        # Upsert all embeddings
        if text_embeddings_list:
            vector_store.upsert(
                embeddings=text_embeddings_list,
                payloads=text_payloads
            )
        
        if image_embeddings_list:
            vector_store.upsert(
                embeddings=image_embeddings_list,
                payloads=image_payloads
            )
        
        ingestion_jobs[job_id] = {
            "status": "completed",
            "progress": 1.0,
            "message": f"Ingestion completed. File ID: {file_id}",
            "file_id": file_id,
            "chunks_count": len(chunks),
            "images_count": len(image_pages)
        }
        
    except Exception as e:
        ingestion_jobs[job_id] = {
            "status": "error",
            "progress": 0.0,
            "message": f"Error: {str(e)}"
        }


@app.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload and process PDF file"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Read file content
    pdf_bytes = await file.read()
    
    # Create ingestion job
    job_id = str(uuid.uuid4())
    
    # Start background processing
    background_tasks.add_task(process_ingestion, job_id, pdf_bytes, file.filename)
    
    return {
        "job_id": job_id,
        "filename": file.filename,
        "status": "processing"
    }


@app.get("/ingestion/status/{job_id}")
async def get_ingestion_status(job_id: str):
    """Get status of ingestion job"""
    if job_id not in ingestion_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return ingestion_jobs[job_id]


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the RAG system"""
    if not retriever or not llm:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    try:
        # Retrieve relevant chunks
        chunks, resolved_query, intent = retriever.retrieve(
            query=request.query,
            use_coref=request.use_coref,
            use_intent=request.use_intent,
            file_id=request.file_id
        )
        
        if not chunks:
            return QueryResponse(
                response="No relevant context found for your query.",
                sources=[],
                intent=intent,
                resolved_query=resolved_query
            )
        
        # Format context
        context = retriever.format_context_for_llm(chunks)
        
        # Get conversation history from retriever memory
        conversation_history = retriever.get_recent_context()
        
        # Generate response
        response = llm.generate_with_sources(
            query=request.query,
            context_chunks=chunks,
            intent=intent,
            conversation_history=conversation_history,
            stream=False
        )
        
        # Add to memory
        retriever.add_to_memory(request.query, response["response"], chunks)
        
        return QueryResponse(
            response=response["response"],
            sources=response["sources"],
            intent=intent,
            resolved_query=resolved_query
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/conversation/clear")
async def clear_conversation(request: ClearConversationRequest):
    """Clear conversation memory and optionally delete vectors for a file."""
    clear_memory_and_vectors(request.file_id)
    return {"status": "cleared", "file_id": request.file_id}


@app.post("/coref", response_model=CorefResponse)
async def resolve_coref(request: CorefRequest):
    """Resolve coreferences in text"""
    if not retriever or not retriever.coref_resolver:
        raise HTTPException(status_code=500, detail="Coreference resolver not available")
    
    resolved = retriever.coref_resolver.resolve(request.text, request.context or "")
    return CorefResponse(resolved_text=resolved)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "components": {
            "vector_store": vector_store is not None,
            "text_embedder": text_embedder is not None,
            "vision_embedder": vision_embedder is not None,
            "retriever": retriever is not None,
            "llm": llm is not None
        }
    }


if __name__ == "__main__":
    import uvicorn
    server_config = config.get("server", {})
    uvicorn.run(
        "fastapi_server:app",
        host=server_config.get("host", "0.0.0.0"),
        port=server_config.get("port", 8000),
        reload=server_config.get("reload", True)
    )

