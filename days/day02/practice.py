# days/day02/practice.py

# ─────────────────────────────────────────────────────
# DAY 02 - Python Patterns for AI Engineering
# ─────────────────────────────────────────────────────

import logging
import time
import sys
import functools
from typing import Generator, Callable, Optional, Type
from dataclasses import dataclass
from pydantic import BaseModel, ValidationError

sys.path.append(".")
from days.day01.practice import get_api_key, DocumentChunk


# ─────────────────────────────────────────────────────
# SECTION A — Custom Logger
# ─────────────────────────────────────────────────────

def setup_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    Creates a reusable logger with timestamps and severity levels.
    Call this once at the top of every module in production apps.

    Args:
        name  : module name so you know where logs come from
        level : minimum level to display (DEBUG shows everything)

    Returns:
        A configured Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


# Module level logger — used by all sections
logger = setup_logger("day02")


# ─────────────────────────────────────────────────────
# SECTION B — Generators
# ─────────────────────────────────────────────────────

@dataclass
class ProcessedChunk:
    """
    Lightweight internal data structure for a document chunk.
    Dataclass used here — internal data, no API validation needed.
    """
    index: int
    content: str
    char_start: int
    char_end: int
    word_count: int
    is_meaningful: bool


def stream_document_chunks(
    text: str,
    chunk_size: int = 200,
    overlap: int = 40,
    min_words: int = 5
) -> Generator[ProcessedChunk, None, None]:
    """
    Production grade document chunker using a generator.

    Why generator: documents can be arbitrarily large.
    Memory usage stays constant regardless of document size.
    Each chunk is processed and discarded before the next loads.

    Args:
        text       : raw document text
        chunk_size : characters per chunk
        overlap    : characters shared between consecutive chunks
        min_words  : skip chunks shorter than this

    Yields:
        ProcessedChunk — one at a time, never all at once
    """
    index = 0
    start = 0
    total_chars = len(text)

    logger.info(
        f"Starting document stream | size: {total_chars} chars | "
        f"chunk_size: {chunk_size} | overlap: {overlap}"
    )

    while start < total_chars:
        end = min(start + chunk_size, total_chars)
        content = text[start:end].strip()
        word_count = len(content.split())
        is_meaningful = word_count >= min_words

        if content:
            chunk = ProcessedChunk(
                index=index,
                content=content,
                char_start=start,
                char_end=end,
                word_count=word_count,
                is_meaningful=is_meaningful
            )

            if is_meaningful:
                logger.debug(
                    f"Chunk {index} | words: {word_count} | "
                    f"chars: {start}-{end}"
                )
                yield chunk
                index += 1
            else:
                logger.warning(
                    f"Skipping chunk at {start}-{end} | "
                    f"only {word_count} words — below minimum {min_words}"
                )

        start += chunk_size - overlap

    logger.info(f"Document stream complete | {index} meaningful chunks yielded")


def simulate_embed_chunk(chunk: ProcessedChunk) -> dict:
    """
    Simulates embedding a chunk via an API call.
    In Day 9 this becomes a real HuggingFace or OpenAI embedding call.

    Returns:
        Dict representing what a vector DB would store
    """
    time.sleep(0.05)
    fake_vector = [float(ord(c)) for c in chunk.content[:5]]

    return {
        "chunk_id": f"chunk_{chunk.index:04d}",
        "content": chunk.content,
        "vector": fake_vector,
        "metadata": {
            "char_start": chunk.char_start,
            "char_end": chunk.char_end,
            "word_count": chunk.word_count
        }
    }


# ─────────────────────────────────────────────────────
# SECTION C — Retry Decorator
# ─────────────────────────────────────────────────────

def retry(
    attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[Type[Exception], ...] = (Exception,)
) -> Callable:
    """
    Production grade retry decorator with exponential backoff.

    Args:
        attempts   : total tries including first attempt
        delay      : initial wait time in seconds
        backoff    : multiplier applied to delay after each failure
        exceptions : only retry on these specific exception types

    Example:
        @retry(attempts=3, delay=1.0, backoff=2.0)
        def call_openai(prompt):
            return openai.chat(prompt)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay

            for attempt in range(1, attempts + 1):
                try:
                    result = func(*args, **kwargs)

                    if attempt > 1:
                        logger.info(
                            f"'{func.__name__}' recovered successfully "
                            f"on attempt {attempt}/{attempts}"
                        )
                    return result

                except exceptions as e:
                    is_last_attempt = attempt == attempts

                    if is_last_attempt:
                        logger.error(
                            f"'{func.__name__}' failed permanently after "
                            f"{attempts} attempts | "
                            f"final error: {type(e).__name__}: {e}"
                        )
                        raise

                    logger.warning(
                        f"'{func.__name__}' attempt {attempt}/{attempts} "
                        f"failed | {type(e).__name__}: {e} | "
                        f"retrying in {current_delay:.1f}s..."
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff

        return wrapper
    return decorator


_call_counts: dict[str, int] = {}


@retry(attempts=3, delay=0.3, backoff=2.0, exceptions=(ConnectionError,))
def embed_text(text: str) -> list[float]:
    """
    Simulates calling OpenAI embeddings API.
    Fails first 2 attempts, succeeds on 3rd.
    Real version: openai.embeddings.create(input=text, model="text-embedding-3-small")
    """
    _call_counts["embed_text"] = _call_counts.get("embed_text", 0) + 1
    count = _call_counts["embed_text"]

    logger.debug(f"embed_text called | attempt count: {count}")

    if count < 3:
        raise ConnectionError(
            f"OpenAI API temporarily unavailable ({count})"
        )

    fake_vector = [float(ord(c) / 100) for c in text[:8]]
    logger.debug(f"Embedding Generated | vector length: {len(fake_vector)}")
    return fake_vector


@retry(attempts=3, delay=0.3, backoff=2.0, exceptions=(ConnectionError, TimeoutError))
def generate_rag_response(question: str, context: str) -> str:
    """
    Simulates calling OpenAI chat API with RAG context.
    Always succeeds — demonstrates the happy path.
    Real version: openai.chat.completions.create(...)
    """
    _call_counts["generate_response"] = _call_counts.get("generate_response", 0) + 1
    logger.debug(f"generate_rag_response called | question: '{question[:40]}'")
    time.sleep(0.1)

    return (
        f"Based on the provided context, {question[:30]}.... "
        f"[answer generated from {len(context)} chars of context]"
    )


@retry(attempts=2, delay=0.2, backoff=2.0, exceptions=(ConnectionError,))
def store_in_vector_db(chunk_id: str, vector: list[float]) -> bool:
    """
    Simulates storing a vector in ChromaDB or Pinecone.
    Always fails — demonstrates total failure scenario.
    Real version: collection.add(embeddings=[vector], ids=[chunk_id])
    """
    _call_counts["store_vector"] = _call_counts.get("store_vector", 0) + 1
    logger.debug(f"store_in_vector_db called | chunk_id: {chunk_id}")
    raise ConnectionError("Vector DB connection refused (simulated total failure)")


# ─────────────────────────────────────────────────────
# SECTION D — Layered Exception Handling
# ─────────────────────────────────────────────────────

class AIEngineerError(Exception):
    """Base exception for all errors in our AI system."""
    pass


class AuthenticationError(AIEngineerError):
    """Invalid or missing API key. Stop immediately. No retry."""
    pass


class RateLimitError(AIEngineerError):
    """Too many requests. Wait and retry with backoff."""
    pass


class DocumentValidationError(AIEngineerError):
    """Document failed validation. Reject input, tell user why."""
    pass


class EmbeddingError(AIEngineerError):
    """Embedding failed after all retries. Skip chunk, continue."""
    pass


class VectorDBError(AIEngineerError):
    """Vector DB operation failed. Queue for retry."""
    pass


class PipelineResult(BaseModel):
    """
    Structured result from RAG pipeline.
    Every pipeline run returns this — success or failure.
    Never return raw strings from pipelines.
    """
    success: bool
    chunks_processed: int = 0
    chunks_failed: int = 0
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    result: Optional[str] = None


def validate_document(content: str, min_length: int = 50) -> str:
    """
    Validates document before it enters the pipeline.
    First line of defense — bad data never enters the system.

    Args:
        content    : raw document text from user
        min_length : minimum acceptable character count

    Returns:
        Cleaned and validated document text

    Raises:
        DocumentValidationError : if document fails any check
    """
    if not content or not content.strip():
        raise DocumentValidationError(
            "Document is empty. Please provide text content."
        )

    if len(content.strip()) < min_length:
        raise DocumentValidationError(
            f"Document too short: {len(content)} chars. "
            f"Minimum required: {min_length} chars."
        )

    non_text_ratio = sum(
        1 for c in content if not c.isprintable()
    ) / len(content)

    if non_text_ratio > 0.3:
        raise DocumentValidationError(
            f"Document appears to be binary or corrupted. "
            f"Non-printable character ratio: {non_text_ratio:.1%}"
        )

    logger.info(f"Document validated | length: {len(content)} chars")
    return content.strip()


def simulate_api_call(prompt: str, scenario: str = "success") -> dict:
    """
    Simulates different API response scenarios.
    In production this is replaced by real OpenAI API call.

    Args:
        prompt   : input text for the API
        scenario : success | auth_error | rate_limit | malformed

    Returns:
        Raw API response dictionary
    """
    logger.debug(
        f"API call | scenario: {scenario} | prompt: '{prompt[:30]}'"
    )

    if scenario == "auth_error":
        raise AuthenticationError(
            "API key invalid or expired. "
            "Check OPENAI_API_KEY in your .env file."
        )

    if scenario == "rate_limit":
        raise RateLimitError(
            "Rate limit exceeded: 60 requests per minute. "
            "Current usage: 61 requests. Please wait."
        )

    if scenario == "malformed":
        return {"status": "ok"}

    return {
        "answer": f"RAG combines retrieval with generation to answer: {prompt[:40]}",
        "tokens_used": 142,
        "model": "gpt-4"
    }


def parse_api_response(raw: dict) -> str:
    """
    Safely extracts answer from raw API response.

    Args:
        raw : raw dictionary response from API

    Returns:
        Extracted answer string

    Raises:
        EmbeddingError : if expected fields are missing
    """
    if "answer" not in raw:
        raise EmbeddingError(
            f"API response missing 'answer' field. "
            f"Got fields: {list(raw.keys())}"
        )
    return raw["answer"]


def run_rag_pipeline(document: str, scenario: str = "success") -> PipelineResult:
    """
    Runs complete RAG pipeline with layered exception handling.
    Each error type caught and handled differently.
    Pipeline never crashes silently.

    Args:
        document : raw document text from user upload
        scenario : API scenario to simulate

    Returns:
        PipelineResult with success status and details
    """
    logger.info(f"Pipeline Started | scenario: {scenario}")
    chunks_processed = 0
    chunks_failed = 0

    # Layer 1: Validate input first
    try:
        validated_doc = validate_document(document)
    except DocumentValidationError as e:
        logger.warning(f"Document rejected: {e}")
        return PipelineResult(
            success=False,
            error_type="DocumentValidationError",
            error_message=str(e)
        )

    # Layer 2: Call API with specific error handling
    try:
        raw_response = simulate_api_call(validated_doc, scenario)
        answer = parse_api_response(raw_response)
        chunks_processed += 1
        logger.info(
            f"API call successful | tokens: {raw_response.get('tokens_used')}"
        )

    except AuthenticationError as e:
        logger.critical(f"Authentication failed — pipeline stopped: {e}")
        return PipelineResult(
            success=False,
            error_type="AuthenticationError",
            error_message=str(e)
        )

    except RateLimitError as e:
        logger.warning(f"Rate limit hit — request queued for retry: {e}")
        return PipelineResult(
            success=False,
            error_type="RateLimitError",
            error_message=str(e)
        )

    except EmbeddingError as e:
        chunks_failed += 1
        logger.error(f"Response parsing failed — skipping chunk: {e}")
        return PipelineResult(
            success=False,
            chunks_failed=chunks_failed,
            error_type="EmbeddingError",
            error_message=str(e)
        )

    except AIEngineerError as e:
        logger.error(f"Pipeline error: {type(e).__name__}: {e}")
        return PipelineResult(
            success=False,
            error_type=type(e).__name__,
            error_message=str(e)
        )

    except Exception as e:
        logger.critical(f"Unexpected error — investigate immediately: {e}")
        return PipelineResult(
            success=False,
            error_type="UnexpectedError",
            error_message=str(e)
        )

    # Layer 3: Return structured success result
    logger.info(
        f"Pipeline completed successfully | chunks: {chunks_processed}"
    )
    return PipelineResult(
        success=True,
        chunks_processed=chunks_processed,
        chunks_failed=chunks_failed,
        result=answer
    )


# ─────────────────────────────────────────────────────
# MAIN — ONLY runs when file executed directly
# python days/day02/practice.py
# NEVER runs when imported by another file
# ─────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Section A: Logger demo ────────────────────────
    logger.debug("DEBUG — fine detail, use during development only")
    logger.info("INFO — system started, processing document...")
    logger.warning("WARNING — chunk size is very small, may affect quality")
    logger.error("ERROR — failed to connect to vector DB, retrying...")
    logger.critical("CRITICAL — API key missing, system cannot continue")

    print("\n--- Simulated RAG Pipeline Log Flow ---\n")
    logger.info("RAG pipeline started")
    logger.info("Loading document: ai_notes.pdf")
    logger.debug("Document size: 4500 characters")
    logger.debug("Chunk 1/9 created: 500 chars")
    logger.debug("Chunk 2/9 created: 500 chars")
    logger.warning("Chunk 3/9 is smaller than minimum size, skipping")
    logger.info("Embedding 8 chunks...")
    logger.error("Embedding API call failed on chunk 5, retrying...")
    logger.info("Retry successful, continuing...")
    logger.info("All chunks stored in vector DB successfully")
    logger.info("RAG pipeline completed")

    # ── Section B: Generator test ─────────────────────
    logger.info("=" * 55)
    logger.info("SECTION B: Generator — Memory Efficient Chunking")
    logger.info("=" * 55)

    sample_document = """
    Artificial intelligence is transforming software engineering.
    Large language models understand and generate human language.
    Retrieval augmented generation combines search with generation.
    Vector databases store numerical representations of text.
    Engineers who understand these systems are in high demand.
    FastAPI is the standard framework for building AI backends.
    Python has become the dominant language for AI engineering.
    Embeddings capture semantic meaning in high dimensional space.
    LangChain simplifies building applications with language models.
    Production AI systems require careful error handling and logging.
    """ * 5

    total_chunks = 0
    total_words = 0
    start_time = time.time()
    embedded_chunks = []

    for chunk in stream_document_chunks(
        sample_document,
        chunk_size=200,
        overlap=40,
        min_words=5
    ):
        embedded = simulate_embed_chunk(chunk)
        embedded_chunks.append(embedded)
        total_chunks += 1
        total_words += chunk.word_count

    elapsed = time.time() - start_time
    logger.info(f"Processing complete in {elapsed:.2f}s")
    logger.info(f"Total chunks embedded: {total_chunks}")
    logger.info(f"Total words processed: {total_words}")
    logger.info(
        f"Average words per chunk: "
        f"{total_words // total_chunks if total_chunks else 0}"
    )
    logger.info(f"Sample chunk stored: {embedded_chunks[0]['chunk_id']}")
    logger.info(f"Sample vector length: {len(embedded_chunks[0]['vector'])}")

    # ── Section C: Retry decorator test ──────────────
    logger.info("=" * 55)
    logger.info("SECTION C: Retry Decorator with Exponential Backoff")
    logger.info("=" * 55)

    logger.info("\n--- Test 1: Partial failure then recovery ---")
    try:
        vector = embed_text("What is retrieval augmented generation?")
        logger.info(f"embed_text succeeded | vector: {vector}")
    except ConnectionError as e:
        logger.error(f"embed_text failed permanently: {e}")

    logger.info("\n--- Test 2: Happy path, no failures ---")
    try:
        context = "RAG combines retrieval with generation to produce grounded answers."
        response = generate_rag_response(
            question="What is RAG?",
            context=context
        )
        logger.info(f"Response generated: {response}")
    except Exception as e:
        logger.error(f"Response generation failed: {e}")

    logger.info("\n--- Test 3: Total failure, all retries exhausted ---")
    try:
        store_in_vector_db("chunk_0001", [0.1, 0.2, 0.3])
    except ConnectionError:
        logger.error(
            "Vector DB storage failed — chunk will be reprocessed later"
        )

    # ── Section D: Exception handling test ───────────
    logger.info("=" * 55)
    logger.info("SECTION D: Layered Exception Handling")
    logger.info("=" * 55)

    scenarios = [
        {
            "name"    : "Valid document, successful API call",
            "doc"     : "Artificial intelligence is transforming how we build software. RAG pipelines combine retrieval with generation.",
            "scenario": "success"
        },
        {
            "name"    : "Empty document — validation rejection",
            "doc"     : "",
            "scenario": "success"
        },
        {
            "name"    : "Document too short — validation rejection",
            "doc"     : "Too short",
            "scenario": "success"
        },
        {
            "name"    : "Valid document — authentication failure",
            "doc"     : "Artificial intelligence is transforming how we build software systems today.",
            "scenario": "auth_error"
        },
        {
            "name"    : "Valid document — rate limit hit",
            "doc"     : "Artificial intelligence is transforming how we build software systems today.",
            "scenario": "rate_limit"
        },
        {
            "name"    : "Valid document — malformed API response",
            "doc"     : "Artificial intelligence is transforming how we build software systems today.",
            "scenario": "malformed"
        },
    ]

    for test in scenarios:
        logger.info(f"\n--- Test: {test['name']} ---")
        result = run_rag_pipeline(test["doc"], test["scenario"])

        if result.success:
            logger.info(f"SUCCESS | chunks: {result.chunks_processed}")
            logger.info(f"Answer: {result.result[:60]}...")
        else:
            logger.warning(
                f"FAILED | type: {result.error_type} | "
                f"message: {result.error_message}"
            )