import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional
import asyncio
import time

load_dotenv() #Load environment variables

# Type hints make your code more readable and help catch errors early. They are not enforced at runtime, but they can be checked with tools like mypy.
def chunk_text(text:str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Splits a long text into overlapping chunks.
    You'll use this exact pattern in your RAG pipeline on Day 12.
    """
    chunks =[]
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

#Test it
sample_text = "AI Engineering is the practice of building systems that use machine learning. " * 20
chunks = chunk_text(sample_text, chunk_size=100,overlap=20)
print(f"Total Chunks created: {len(chunks)}")
print(f"First chunk: {chunks[0]}")
print(f"Second chunk starts with overlap: {chunks[1][:30]}")
      
def get_api_key(key_name: str) -> str:
    """Safely retrieve an API key, raising a clear error if missing"""
    value = os.getenv(key_name)
    if not value:
        raise ValueError(f"Missing required environment variable: {key_name}")
    return value

# add a fake key to .env for testing
try:
    api_key = get_api_key("OPENAI_API_KEY")
    print(f"Key loaded successfully. Length: {len(api_key)}")
except ValueError as e:
    print(f"Error: {e}")

#This is how you will structure data in every Fast API and LLM project.
class DocumentChunk(BaseModel):
    chunk_id: str 
    content: str
    source: str
    page_number: Optional[int] = None
    token_estimate: int = Field(default=0, description="Rough token count")

    def estimate_tokens(self) -> int:
        """Rough estimate: 1 token =  4 characters"""
        return len(self.content) //4
    
#create an Instance and see how pyndatic validates data
chunk = DocumentChunk(
    chunk_id="chunk_001",
    content="RAG Stands for Retrieval-Augmented Generation",
    source="ai_notes.pdf",
    page_number=3
)

chunk.token_estimate = chunk.estimate_tokens()
print(chunk.model_dump()) #converts to dictionary
print(f"Token Estimate: {chunk.token_estimate}")

#Try passing wrong_types  - pyndanti will catch it
try:
    bad_chunk = DocumentChunk(
        chunk_id=123,  # should be str
        content=None, # should be str
        source="bad_data.txt"
    )
except Exception as e:
    print(f"Pydantic Validation Error: {e} ")

#Simulating async LLM Calls - you will use  this pattern with OpenAI's async client
async def fake_llm_call(prompt: str, delay: float = 1.0) -> str:
    """Simulates a slow api call to an LLM """
    await asyncio.sleep(delay) # non blocking wait
    return f"Response to: '{prompt[:30]}...'"

async def process_multiple_prompts(prompts: list[str]) -> list[str]:
    """
    processs multiple prompts concurrently instead of sequentially.
    Sequential would take sum of all delays. Concurrent takes max delay.
    """
    tasks = [fake_llm_call(prompt, delay=1.0) for prompt in prompts]
    results = await asyncio.gather(*tasks)
    return list(results)

#Run and compare timing
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
#Notice it takes around 1 second , not 3 seconds