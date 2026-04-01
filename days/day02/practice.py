# days/day02/practice.py

# ─────────────────────────────────────────
# DAY 02 - Python Patterns for AI Engineering
# ─────────────────────────────────────────

# === IMPORTS ===
import logging
import time
import sys
import asyncio
import functools
from typing import Generator, Callable, Optional, Type
from dataclasses import dataclass
from pydantic import BaseModel

# Import from Day 1 — reusing your earlier work
sys.path.append(".")
from days.day01.practice import get_api_key, DocumentChunk

# ─────────────────────────────────────────
# SECTION A — Custom Logger
# ─────────────────────────────────────────
# ... logger code here

def setup_logger(name:str, level:int = logging.DEBUG) -> logging.Logger:
    """
    Creates a reusable logger with timestamps and severity levels.
    Call this once at the top of every module in production apps.

    Args:
        name : give it the module name so you know where logs come from.
        level : minimum level to display (DEBUG shows everything)

    returns:
        A configured Logger instance
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Guard: if logger already has handlers, dont add duplicates
    # This matters when the same module gets imported multiple times
    if logger.handlers:
        return logger
    
    # Handler: where logs go - in this case, the terminal
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Formatter: what each log line looks like
    # %(name)s -> logger name you passed in
    # %(levelname)s -> DEBUG, INFO, WARNING etc
    #%(asctime)s -> timestamp
    #%(message)s -> your actual message
    formatter = logging.Formatter(
        fmt = "[%(asctime)s] %(levelname)-8s | %(name)s | %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

# Create module-level logger
# In production, every Python file has its own logger named after the module
logger = setup_logger("day02")
 
# ─────────────────────────────────────────
# SECTION B — Generators
# ─────────────────────────────────────────
# ... generator code here

@dataclass
class ProcessedChunk:
    """
    Dataclass — lightweight internal data structure.
    No validation needed here, this is internal processing data.
    We use Pydantic at the API boundary, dataclass inside the pipeline.
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
        min_words  : skip chunks shorter than this — not meaningful
    
    Yields:
        ProcessedChunk — one at a time, never all at once
    """
    index = 0
    start = 0
    total_chars = len(text)

    logger.info(f"Starting document stream | size : {total_chars} chars | "
                f"chunk_size:{chunk_size} | overlap: {overlap}")
    
    while start < total_chars:
        end = min(start + chunk_size, total_chars)
        content = text[start:end].strip()
        word_count = len(content.split())
        is_meanigful = word_count >= min_words

        if content:
            chunk = ProcessedChunk(
                index = index,
                content = content,
                char_start=start,
                char_end=end,
                word_count=word_count,
                is_meaningful=is_meanigful
            )

            if is_meanigful:
                logger.debug(
                    f"Chunk {index} | words: {word_count} | "
                    f"chars: {start}-{end}"
                )
                yield chunk # pause here,return this chunk
                index +=1
            else:
                logger.warning(
                    f"Skipping chunk at {start}-{end} | "
                    f"Only {word_count} words - below minimum {min_words}"
                )

            start += chunk_size - overlap

        logger.info(f"Document stream complete | {index} meaningful chunks yielded")

def simulate_embed_chunk(chunk: ProcessedChunk) ->  dict:
    """
    Simulates embedding a chunk via an API call.
    In Day 9 this becomes a real HuggingFace or OpenAI embedding call.
    
    Returns a dict representing what a vector DB would store.
    """

    # Simulate API delay  - real embedding APIs take 0.1 to 0.5 seconds
    time.sleep(0.05)

    # Fake embedding vector - in reality this is 768 to 1536 floats
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





# ─────────────────────────────────────────
# SECTION C — Retry Decorator
# ─────────────────────────────────────────
# ... decorator code here
import functools
import time
from typing import Callable, Type

def retry(
        attempts: int = 3,
        delay: float = 1.0,
        backoff: float = 2.0,
        exceptions: tuple[Type[Exception], ... ] = (Exception, )
) -> Callable:
    """
    Production grade retry decorator with exponential backoff.

    How it works:
    → wraps any function
    → if function raises an exception → wait → retry
    → each retry waits longer than the last (exponential backoff)
    → if all attempts exhausted → raise the original error

    Args:
        attempts   : total number of tries including the first attempt
        delay      : initial wait time in seconds between retries
        backoff    : multiplier applied to delay after each failure
        exceptions : only retry on THESE specific exception types
                     everything else fails immediately — no retry

    Example:
        @retry(attempts=3, delay=1.0, backoff=2.0)
        def call_openai(prompt):
            return openai.chat(prompt)
        
        Attempt 1 fails → wait 1.0s
        Attempt 2 fails → wait 2.0s
        Attempt 3 fails → raise error
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay

            for attempt in range(1, attempts + 1):
                try:
                    result = func(*args, **kwargs)

                    #Log recovery if this was not the first attempt
                    if attempt > 1:
                        logger.info(
                            f"'{func.__name__}' recovered successfully "
                            f"on attempt {attempt}/{attempts}"
                        )
                    return result
                except exceptions as e:
                    is_last_attempt = attempt == attempts

                    if is_last_attempt:
                        #All attempts exhausted - give up and raise
                        logger.error(
                            f"'{func.__name__}' failed permanently after "
                            f"{attempts} attempts | "
                            f"final error: {type(e).__name__}: {e}"
                        )
                        raise
                    # Not last attempt - log and wait
                    logger.warning(
                        f"'{func.__name__}' attempt {attempt}/{attempts} "
                        f"failed | {type(e).__name__}: {e} | "
                        f"retrying in {current_delay:.1f}s..."
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff #exponential backoff
        return wrapper
    return decorator

# ── Simulated AI API calls using the decorator ────────

# Tracks how many times each function has been called
# In real code this state lives in the API itself
_call_counts: dict[str ,int] = {}

@retry (attempts=3, delay=0.3, backoff=2.0, exceptions=(ConnectionError,))
def embed_text(text: str) -> list[float]:
    """
    Simulates calling OpenAI embeddings API.
    Fails first 2 attempts, succeeds on 3rd.
    Real version: openai.embeddings.create(input=text, model="text-embedding-3-small)
    """
    _call_counts["embed_text"] = _call_counts.get("embed_text", 0) + 1
    count = _call_counts["embed_text"]

    logger.debug(f"embed_text called | attempt count: {count}")

    if count < 3:
        raise ConnectionError(
            f"OpenAI API temporarily unavailable ({count})"
        )
    
    #success - return fake embedding vector
    fake_vector = [float(ord(c)/100) for c in text[:8]]
    logger.debug(f"Embedding Generated | vector length: {len(fake_vector)}")
    return fake_vector

@retry(attempts=3, delay=0.3, backoff=2.0, exceptions=(ConnectionError, TimeoutError))
def generate_rag_response(question:str, context:str) -> str:    
    """
    Simulates calling OpenAI chat API with RAG context.
    Always succeeds — demonstrates the happy path.
    Real version: openai.chat.completions.create(...)
    """
    _call_counts["generate_response"] = _call_counts.get("generate_response",0) + 1

    logger.debug(f"generate_rag_response called | question: '{question[:40]}")

    #simulate API processing time
    time.sleep(0.1)

    return(
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


# ─────────────────────────────────────────
# SECTION D — Layered Exception Handling
# ─────────────────────────────────────────
# ... exception handling code here
from typing import Optional
from pydantic import BaseModel, ValidationError

# ── Step 1: Define Your Own Exception Hierarchy ───────
#
# Why custom exceptions?
# Built-in exceptions like Exception, ValueError are too generic.
# Custom exceptions let you catch EXACTLY the error you expect
# and handle it appropriately without catching unrelated errors.

class AIEngineerError(Exception):
    """
    Base exception for all errors in our AI system.
    Every custom exception inherits from this.
    Lets you catch ALL our custom errors with one except clause.
    """
    pass

class AuthenticationError(AIEngineerError):
    """
    Invalid or missing API Key.
    Recovery: Stop immediately. No retry. Fix the key.
    Never retry authentication errors - retrying won't help.
    """
    pass

class RateLimitError(AIEngineerError):
    """
    Too Many API requests sent in a short period.
    Recovery: WAIT then retry with exponential backoff.
    This is temporary - API will accept requests again soon.                                                                                                                                                                        
    """
    pass

class DocumentValidationError(AIEngineerError):
    """
    Document or input data failed validation checks.
    Recovery: REJECT the input, return clear error to user.
    Do not process invalid data - garbage in, garbage out.
    """
    pass

class EmbeddingError(AIEngineerError):
    """
    Embedding API call failed after all retries.
    Recovery: LOG the chunk, skip it, continue pipeline.
    One bad chunk should not crash the entire pipeline.
    """
    pass

class VectorDBError(AIEngineerError):
    """
    Vector database operation failed.
    Recovery: Log and add to retry queue.
    Data is not lost - it can be reprocessed later.
    """
    pass

# ── Step 2: Response Model ────────────────────────────

class PipelineResult(BaseModel):
    """
    Structured result from RAG pipeline.
    Every pipeline run returns this - success or failure.
    Never return raw strings or unstructured data from pipelines.
    """
    success: bool
    chunks_processed: int = 0
    chunks_failed: int = 0
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    result: Optional[str] = None

    # ── Step 3: Individual Operation Handlers ────────────
def validate_document (content: str, min_length:int = 50)-> str:
    """
    Validates document before it enters the pipeline.
    First line of defense - bad data never enters the system.

    Args:
        content : raw document text from user
        min_length : minimum acceptable character count

    Returns:
    cleaned and validated document text

    Raises:
        DocumentValidationError : if document fails any check
    """

    # Check 1: document must exist
    if not content or not content.strip():
        raise DocumentValidationError(
            "Document is empty. Please provide text content."
        )
    
    # Check 2: Document must meet minimum length
    if len(content.strip()) < min_length:
        raise DocumentValidationError(
            f"Document too short: {len(content)} chars. "
            f"Minimum required: {min_length} chars."
        )
    
    #check 3: document must be readable text
    non_text_ratio = sum(1 for c in content if not c.isprintable()) / len(content)
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
        prompt : input text for the API
        scenario : which scenario to simulate
                'success' -> normal response
                'auth_error' -> invalid API key
                'rate_limit' -> too many requests
                'malformed' -> corrupted response
    
    Returns:
            Raw API response dictionary

    Raises: 
            AuthenticationError : on auth_error scenario
            RateLimitError : on rate_limit scenario
    """
    logger.debug(f"API call | scenario: {scenario} | prompt: '{prompt[:30]}'")

    if scenario == "auth_error":
        raise AuthenticationError(
            "API key invalid or expired. "
            "Check OPENAI_API_KEY in your .env file."
        )
    
    if scenario == "rate_limit":
        raise RateLimitError(
            "Rate limit exceeded: 60 requests per minute . "
            "Current usage: 61 requests. Please wait."
        )
    
    if scenario == "malformed":
        # Returns response missing expected fields
        return {"status": "ok"} # missing answer field
    
    #success scenario
    return{
        "answer": f"RAG combines retrieval with generation to answer: {prompt[:40]}",
        "tokens_used": 142,
        "model": "gpt-4"
    }

def parse_api_response(raw: dict) -> str:
    """
    Safely extracts answer from raw API response.
    Handles malformed response without crashing pipeline.

    Args:
        raw: raw dictionary response from API

    Returns:
         Extracted answer string

    Raises:
        EmbeddingError: if expected fields are missing
    """

    if 'answer' not in raw:
        raise EmbeddingError(
            f"API response missing 'answer' field. "
            f"Got fields: {list(raw.keys())}"
        )
    return raw["answer"] 

# ── Step 4: The Full Pipeline With Layered Handling ──

def run_rag_pipeline(document: str, scenario: str = "success") -> PipelineResult:
    """
    Runs the complete RAG pipeline with proper exception handling.
    Each error type is caught and handled differently.
    Pipeline never crash silently - every failure is logged and classified.

    Args:
        document: raw document text from user upload.
        scenario: API scenario to simulate.

    Returns:
        PipelineResult with success status and details
    """
    logger.info(f"Pipeline Started | scenario : {scenario}")
    chunks_processed = 0
    chunks_failed = 0

     # ── Layer 1: Validate input first ─────────────────
    # Bad data must never enter the pipeline
    try:
        validated_doc = validate_document(document)
    except DocumentValidationError as e:
        #Reject immediately - clear message to user
        logger.warning(f"Document rejected: {e}")
        return PipelineResult(
            success=False,
            error_type="DocumentValidationError",
            error_message=str(e)
        )

    # ── Layer 2: Call API with specific error handling ─
    try:
        raw_response = simulate_api_call(validated_doc, scenario)
        answer = parse_api_response(raw_response)
        chunks_processed += 1
        logger.info(f"API call successful | takens: {raw_response.get('tokens_used')}")

    except AuthenticationError as e:
        # STOP immediately — retrying will not help
        # Alert the developer — this needs human intervention
        logger.critical(f"Authentication failed - pipeline stopped: {e}")
        return PipelineResult(
            success=False,
            error_type="AuthenticationError",
            error_message=str(e)
        )
    
    except RateLimitError as e:
        # Recoverable — but retry decorator handles this
        # Here we just log and report — retry is at decorator level
        logger.warning(f"Rate limit hit - request queued for retry: {e}")
        return PipelineResult(
            success=False,
            error_type="RateLimitError",
            error_message=str(e)
        )
    
    except EmbeddingError as e:
        #Log and skip - one bad chunk should not crash pipeline
        chunks_failed += 1
        logger.error(f"Response parsing failed - skipped chunk: {e}")
        return PipelineResult(
            success=False,
            chunk_failed=chunks_failed,
            error_type="EmbeddingError",
            error_message=str(e)
        )
    
    except AIEngineerError as e:
        #catch any other custom error we defined
        #Safety net for our own exception hierarchy
        logger.error(f"Pipeline error: {type(e).__name__}: {e}")
        return PipelineResult(
            success=False,
            error_type=type(e).__name__,
            error_message=str(e)
        )
    
    except Exception as e:
        #Catch anything completely unexpected
        # Last resort - should rarely trigger in well written code
        logger.critical(f"Unexpected error - investigate immediately:{e}")
        return PipelineResult(
            success=False,
            error_type="UnexpectedError",
            error_message=str(e)
        )
    
   # ── Layer 3: Return structured success result ──────
    logger.info(f"Pipeline completed successfully | chunks: {chunks_processed}")
    return PipelineResult(
        success=True,
        chunks_processed=chunks_processed,
        chunks_failed=chunks_failed,
        result=answer
    )
# ─────────────────────────────────────────
# MAIN — Run all sections
# ─────────────────────────────────────────
if __name__ == "__main__":
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

# ─────────────────────────────────────────────────
    # SECTION B TESTS — Generators
    # ─────────────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("SECTION B: Generator — Memory Efficient Chunking")
    logger.info("=" * 55)

    # Real document simulation
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
    """ * 5  # multiply to simulate real document size

    # Track stats — real production code always measures performance
    total_chunks = 0
    total_words = 0
    skipped_chunks = 0
    start_time = time.time()

    # Process document — one chunk at a time, never all at once
    embedded_chunks = []
    for chunk in stream_document_chunks(
        sample_document,
        chunk_size=200,
        overlap=40,
        min_words=5
    ):
        # In real RAG pipeline — embed and store each chunk here
        embedded = simulate_embed_chunk(chunk)
        embedded_chunks.append(embedded)
        total_chunks += 1
        total_words += chunk.word_count

    elapsed = time.time() - start_time

    # Real system generated summary log — not hardcoded
    logger.info(f"Processing complete in {elapsed:.2f}s")
    logger.info(f"Total chunks embedded: {total_chunks}")
    logger.info(f"Total words processed: {total_words}")
    logger.info(f"Average words per chunk: {total_words // total_chunks if total_chunks else 0}")
    logger.info(f"Sample chunk stored: {embedded_chunks[0]['chunk_id']}")
    logger.info(f"Sample vector length: {len(embedded_chunks[0]['vector'])}")

# ─────────────────────────────────────────────────
# SECTION C TESTS — Retry Decorator
# ─────────────────────────────────────────────────
logger.info("=" * 55)
logger.info("SECTION C: Retry decorator with Exponential Backoff")
logger.info("=" * 55)



# Test 1 — Partial failure then recovery
logger.info("\n--- Test 1: Partial failure then recovery ---")
try:
    vector = embed_text("What is retrieval augmented generation?")
    logger.info(f"embed_text succeeded | vector: {vector}")
except ConnectionError as e:
    logger.error(f"embed_text failed permanently: {e}")

# Test 2 — Happy path, no failures
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

# Test 3 — Total failure, all retries exhausted
logger.info("\n--- Test 3: Total failure, all retries exhausted ---")
try:
    store_in_vector_db("chunk_0001", [0.1, 0.2, 0.3])
except ConnectionError:
    logger.error("Vector DB storage failed — chunk will be reprocessed later")
    # In production: add to a dead letter queue for reprocessing

    # ─────────────────────────────────────────────────
    # SECTION D TESTS — Exception Handling
    # ─────────────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("SECTION D: Layered Exception Handling")
    logger.info("=" * 55)

    # Define all scenarios to test
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

    # Run every scenario and show structured result
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