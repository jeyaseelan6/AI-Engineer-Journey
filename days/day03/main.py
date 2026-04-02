# days/day03/main.py

# ─────────────────────────────────────────────────────
# DAY 03 - FastAPI Server
# ─────────────────────────────────────────────────────

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
import sys
sys.path.append(".")

from days.day02.practice import (
    setup_logger,
    validate_document,
    DocumentValidationError,
    PipelineResult,
    run_rag_pipeline,
    RateLimitError,
    AuthenticationError,
    EmbeddingError
)
from days.day01.practice import chunk_text


# ─────────────────────────────────────────────────────
# App Instance
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
    Returns 200 OK when server is running correctly.
    """
    logger.info("Health check called")
    return {
        "status": "healthy",
        "message": "All systems operational"
    }


# ─────────────────────────────────────────────────────
# SECTION 2 — Pydantic Request and Response Models
# ─────────────────────────────────────────────────────

class ChunkRequest(BaseModel):
    """
    Request model for document chunking endpoint.
    Every field validated automatically before your function runs.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "Artificial intelligence is transforming software engineering. RAG pipelines combine retrieval with generation to produce grounded answers.",
                "chunk_size": 200,
                "overlap": 40
            }
        }
    )

    text: str = Field(
        ...,
        min_length=10,
        description="Document text to chunk"
    )
    chunk_size: int = Field(
        default=200,
        ge=50,
        le=2000,
        description="Characters per chunk"
    )
    overlap: int = Field(
        default=40,
        ge=0,
        le=200,
        description="Shared characters between consecutive chunks"
    )


class ValidateRequest(BaseModel):
    """
    Request model for document validation endpoint.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "content": "Artificial intelligence is transforming how we build software systems today.",
                "min_length": 50
            }
        }
    )

    content: str = Field(
        ...,
        description="Document text to validate"
    )
    min_length: int = Field(
        default=50,
        ge=10,
        description="Minimum acceptable character count"
    )


class ChunkResponse(BaseModel):
    """
    Response model for document chunking endpoint.
    Structured, predictable, always the same shape.
    """
    total_chunks: int
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


@app.post("/chunk-document", response_model=ChunkResponse)
def chunk_document(request: ChunkRequest):
    """
    Splits a document into overlapping chunks.
    Takes raw text and returns chunks ready for RAG pipeline embedding.
    This is Step 1 of your RAG pipeline.
    """
    logger.info(
        f"Chunk request received | "
        f"text length: {len(request.text)} chars | "
        f"chunk_size: {request.chunk_size}"
    )

    chunks = chunk_text(
        text=request.text,
        chunk_size=request.chunk_size,
        overlap=request.overlap
    )

    logger.info(f"Chunking complete | produced {len(chunks)} chunks")

    return ChunkResponse(
        total_chunks=len(chunks),
        chunk_size_used=request.chunk_size,
        overlap_used=request.overlap,
        chunks=chunks,
        message=f"Successfully created {len(chunks)} chunks"
    )


@app.post("/validate-document", response_model=ValidationResponse)
def validate_document_endpoint(request: ValidateRequest):
    """
    Validates a document before pipeline processing.
    Checks if document is non-empty, meets minimum length,
    and contains readable text.
    Gateway check before any expensive processing.
    """
    logger.info(
        f"Validation request received | "
        f"content length: {len(request.content)} chars"
    )

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
        logger.warning(f"Document validation failed: {e}")

        return ValidationResponse(
            is_valid=False,
            message=str(e),
            character_count=len(request.content)
        )


# ─────────────────────────────────────────────────────
# SECTION 3 — HTTP Status Codes and Error Handling
# ─────────────────────────────────────────────────────

class PipelineRequest(BaseModel):
    """
    Request model for full RAG pipeline endpoint.
    Accepts document text and simulation scenario.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "document": "Artificial intelligence is transforming how we build software systems. RAG pipelines combine retrieval with generation to produce accurate answers.",
                "scenario": "success"
            }
        }
    )

    document: str = Field(
        ...,
        min_length=10,
        description="Document text to process through RAG pipeline"
    )
    scenario: str = Field(
        default="success",
        description="success | auth_error | rate_limit | malformed"
    )


class PipelineResponse(BaseModel):
    """
    Response model for RAG pipeline endpoint.
    Returned only on success — errors use HTTPException.
    """
    success: bool
    chunks_processed: int = 0
    chunks_failed: int = 0
    result: Optional[str] = None


@app.post(
    "/run-pipeline",
    response_model=PipelineResponse,
    status_code=200,
    responses={
        400: {"description": "Invalid document — bad request"},
        401: {"description": "Authentication failed — invalid API key"},
        429: {"description": "Rate limit exceeded — slow down"},
        500: {"description": "Unexpected server error"}
    }
)
def run_pipeline(request: PipelineRequest):
    """
    Runs the complete RAG pipeline on a document.

    Returns different HTTP status codes based on failure type:
    200 → success, pipeline completed
    400 → invalid document, client must fix input
    401 → authentication failure, check API key
    429 → rate limit exceeded, slow down requests
    500 → unexpected server error, investigate immediately
    """
    logger.info(
        f"Pipeline request received | "
        f"document length: {len(request.document)} chars | "
        f"scenario: {request.scenario}"
    )

    result = run_rag_pipeline(
        document=request.document,
        scenario=request.scenario
    )

    # ── Translate errors to HTTP status codes ─────────
    if not result.success:
        error_type = result.error_type

        if error_type == "DocumentValidationError":
            logger.warning(
                f"Returning 400 Bad Request | {result.error_message}"
            )
            raise HTTPException(
                status_code=400,
                detail={
                    "error": error_type,
                    "message": result.error_message,
                    "action": "Fix your document and try again"
                }
            )

        elif error_type == "AuthenticationError":
            logger.error(
                f"Returning 401 Unauthorized | {result.error_message}"
            )
            raise HTTPException(
                status_code=401,
                detail={
                    "error": error_type,
                    "message": result.error_message,
                    "action": "Check your OPENAI_API_KEY in .env file"
                }
            )

        elif error_type == "RateLimitError":
            logger.warning(
                f"Returning 429 Too Many Requests | {result.error_message}"
            )
            raise HTTPException(
                status_code=429,
                detail={
                    "error": error_type,
                    "message": result.error_message,
                    "action": "Wait 60 seconds before retrying"
                }
            )

        else:
            logger.error(
                f"Returning 500 Internal Server Error | {result.error_message}"
            )
            raise HTTPException(
                status_code=500,
                detail={
                    "error": error_type,
                    "message": result.error_message,
                    "action": "Server error — contact support"
                }
            )

    # ── Success — return 200 ───────────────────────────
    logger.info(
        f"Pipeline succeeded | "
        f"chunks processed: {result.chunks_processed}"
    )

    return PipelineResponse(
        success=True,
        chunks_processed=result.chunks_processed,
        chunks_failed=result.chunks_failed,
        result=result.result
    )
