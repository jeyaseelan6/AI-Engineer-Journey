# days/day01/practice.py

# ─────────────────────────────────────────────────────
# DAY 01 - Python for AI Engineering
# ─────────────────────────────────────────────────────

import os
import asyncio
import time
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional


# ─────────────────────────────────────────────────────
# FUNCTIONS AND CLASSES — safe to import
# these never run automatically
# they only run when called explicitly
# ─────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Splits a long text into overlapping chunks.
    Used in RAG pipeline on Day 12.

    Args:
        text       : raw document text
        chunk_size : characters per chunk
        overlap    : shared characters between consecutive chunks

    Returns:
        List of text chunk strings
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def get_api_key(key_name: str) -> str:
    """
    Safely retrieves API key from environment variables.
    Never hardcode API keys in source code.

    Args:
        key_name : name of the environment variable

    Returns:
        API key string

    Raises:
        ValueError : if environment variable is missing
    """
    value = os.getenv(key_name)
    if not value:
        raise ValueError(
            f"Missing required environment variable: {key_name}"
        )
    return value


class DocumentChunk(BaseModel):
    """
    Validated data model for a single document chunk.
    Used throughout the RAG pipeline.
    """
    chunk_id: str
    content: str
    source: str
    page_number: Optional[int] = None
    token_estimate: int = Field(default=0, description="Rough token count")

    def estimate_tokens(self) -> int:
        """Rough estimate: 1 token = 4 characters."""
        return len(self.content) // 4


async def fake_llm_call(prompt: str, delay: float = 1.0) -> str:
    """
    Simulates a slow API call to an LLM.
    Used to demonstrate async concurrency pattern.
    """
    await asyncio.sleep(delay)
    return f"Response to: '{prompt[:30]}...'"


async def process_multiple_prompts(prompts: list[str]) -> list[str]:
    """
    Processes multiple prompts concurrently instead of sequentially.
    Sequential takes sum of all delays. Concurrent takes max delay.
    """
    tasks = [fake_llm_call(prompt, delay=1.0) for prompt in prompts]
    results = await asyncio.gather(*tasks)
    return list(results)


# ─────────────────────────────────────────────────────
# MAIN BLOCK
# Runs ONLY when you execute this file directly:
#   python days/day01/practice.py
#
# Does NOT run when another file imports from here:
#   from days.day01.practice import chunk_text
# ─────────────────────────────────────────────────────

if __name__ == "__main__":

    load_dotenv()

    # ── Test chunk_text ───────────────────────────────
    print("\n--- Section A: chunk_text ---")
    sample_text = (
        "AI Engineering is the practice of building systems "
        "that use machine learning. " * 20
    )
    chunks = chunk_text(sample_text, chunk_size=100, overlap=20)
    print(f"Total Chunks created: {len(chunks)}")
    print(f"First chunk: {chunks[0]}")
    print(f"Second chunk starts with overlap: {chunks[1][:30]}")

    # ── Test get_api_key ──────────────────────────────
    print("\n--- Section B: get_api_key ---")
    try:
        api_key = get_api_key("OPENAI_API_KEY")
        print(f"Key loaded successfully. Length: {len(api_key)}")
    except ValueError as e:
        print(f"Error: {e}")

    # ── Test DocumentChunk ────────────────────────────
    print("\n--- Section C: DocumentChunk ---")
    chunk = DocumentChunk(
        chunk_id="chunk_001",
        content="RAG Stands for Retrieval-Augmented Generation",
        source="ai_notes.pdf",
        page_number=3
    )
    chunk.token_estimate = chunk.estimate_tokens()
    print(chunk.model_dump())
    print(f"Token Estimate: {chunk.token_estimate}")

    try:
        bad_chunk = DocumentChunk(
            chunk_id=123,
            content=None,
            source="bad_data.txt"
        )
    except Exception as e:
        print(f"Pydantic Validation Error: {e}")

    # ── Test async ────────────────────────────────────
    print("\n--- Section D: Async ---")
    prompts = [
        "Explain what RAG is in simple terms",
        "What is a vector embedding",
        "How does Langchain Work"
    ]
    start = time.time()
    results = asyncio.run(process_multiple_prompts(prompts))
    elapsed = time.time() - start
    print(f"\nProcessed {len(prompts)} prompts in {elapsed:.2f}s (concurrent)")
    print("Results:")
    for r in results:
        print(f" - {r}")