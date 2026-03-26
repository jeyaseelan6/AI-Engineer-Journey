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


# ─────────────────────────────────────────
# SECTION D — Exception Handling
# ─────────────────────────────────────────
# ... exception handling code here


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
