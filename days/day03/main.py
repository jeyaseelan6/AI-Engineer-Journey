# days/day03/main.py

# ─────────────────────────────────────────────────────
# DAY 03 - FastAPI Server
# ─────────────────────────────────────────────────────

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import sys
sys.path.append(".")

from days.day02.practice import (
    setup_logger,
    validate_document,
    DocumentValidationError,
    PipelineResult
    )

from days.day01.practice import chunk_text


# ─────────────────────────────────────────────────────
# App Instance — the entire server lives here
# ─────────────────────────────────────────────────────

app = FastAPI(
    title="AI Engineer API",
    description="RAG pipeline and document processing API",
    version="0.1.0"
)

logger = setup_logger("day03")

# ─────────────────────────────────────────────────────
# SECTION 1 — Basic Endpoints
# ─────────────────────────────────────────────────────

@app.get("/")
def root():
    """
    Root endpoint.
    Returns a welcome message.
    First endpoint every API has.
    """
    logger.info("Root endpoint called")
    return {
        "message": "AI Engineer API is running",
        "version": "0.1.0",
        "status": "healthy"
    }

@app.get("/health")
def health_check():
    """
    Health check endpoint.
    Used by deployment platforms to verify server is alive.
    Railway, Render, and AWS all ping this automatically.
    Returns 200 OK when server is running correctly.
    """
    logger.info("Health check called")
    return{
        "status": "healthy",
        "message": "All systems operational"
    }

# ─────────────────────────────────────────────────────
# SECTION 2 — Pydantic Request and Response Models
# ─────────────────────────────────────────────────────

# ── Request Models — what comes IN ───────────────────

class ChunkRequest(BaseModel):
    """
    Request model for document chunking endpoint.
    Every field is validated automatically before your function runs.
    """
    text: str = Field(
        ..., # ... means required, no default
        min_length=10,
        description="Document text to chunk"
    )
    chunk_size: int =Field(
        default=200,
        ge=50, # ge = greater than or equal to 50
        le=2000, # le = less than or equal to 2000
        description="Characters per chunk"
    )
    overlap: int = Field(
        default=40,
        ge=0,
        le=200,
        description="Shared characters between consecutive chunks"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Artificial intelligence is transforming software engineering. RAG pipeline combines retrieval with generation to produce grounded answers.",
                "chunk_size": 200,
                "overlap": 40
            }
        }

class ValidateRequest(BaseModel):
    """
    Request model for document validation endpoint.
    """
    content: str = Field(
        ...,
        description="Document text to validate"
    )
    min_length: int = Field(
        default=50,
        ge=10,
        description="Minimum acceptable character count"
    )

    class Config:
        json_scheme_extra = {
            "example": {
                "content": "Artificial intelligence is transforming how we build software systems today.",
                "min_length": 50
            }
        }
    
# ── Response Models — what goes OUT ──────────────────

class ChunkResponse(BaseModel):
    """
    Response model for document chunking endpoint.
    Structured, predicted, always the same shape.
    """
    total_chunks:int
    chunk_size_used: int
    overlap_used: int
    chunks: list[str]
    message: str

class ValidationResponse(BaseModel):
    """
    Response model for document validation endpoint.
    """
    is_valid: bool
    message: str
    character_count: Optional[int] = None
    cleaned_text: Optional[str] = None

# ── Endpoints ─────────────────────────────────────────

@app.post("/chunk-document",response_model=ChunkResponse)
def chunk_document(request: ChunkRequest):
    """
    Splits a document into overlapping chunks.

    Takes a raw text and returns a list of chunks ready
    for embedding in a RAG pipeline.
    """
    logger.info(
        f"Chunk request received | "
        f"text length: {len(request.text)} chars | "
        f"chunk_size: {request.chunk_size}"
    )

    # Use your Day 1 chunk_text function directly
    # FastAPI + Pydantic already validated the input
    # You receive guaranteed clean data here
    chunks = chunk_text(
        text=request.text,
        chunk_size=request.chunk_size,
        overlap=request.overlap
    )

    logger.info(
        f"Chunking complete | "
        f"produced {len(chunks)} chunks"
    )

    return ChunkResponse(
        total_chunks=len(chunks),
        chunk_size_used=request.chunk_size,
        overlap_used=request.overlap,
        chunks=chunks,
        message=f"Successfully Created {len(chunks)} chunks"
    )

@app.post("/validate-document",response_model=ValidationResponse)
def validate_document_endpoint(request: ValidateRequest):
    """
    Validates a document before pipeline processing.

    Checks if document is non-empty, meets minimum length,
    and contains readable text.

    This is the gateway check before any expensive processing.
    """
    logger.info(
        f"Validation request received | "
        f"content length:{len(request.content)} chars"
    )

    # Use your Day 2 validate_document function
    try:
        cleaned = validate_document(
            content=request.content,
            min_length=request.min_length
        )
        logger.info("Document validation passed")

        return ValidationResponse(
            is_valid=True,
            message="Document is valid and ready for processing",
            character_count=len(cleaned),
            cleaned_text=cleaned[:200] + "..." if len(cleaned) > 200 else cleaned
        )
    except DocumentValidationError as e:
        # Document failed validation
        # Return structured response — do not crash
        logger.warning(f"Document validation failed: {e}")

        return ValidationResponse(
            is_valid=False,
            message=str(e),
            character_count=len(request.content)
        )