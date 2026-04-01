# Day 2 — Complete Notes

---

## `days/day02/notes.md`

---

# Day 2: Python Patterns for AI Engineering
## Personal Study Notes

---

## Section A — Custom Logger

### What Is a Logger

A logger is a professional tool for recording what happens inside your application. It replaces `print()` permanently in production code.

```python
# Development beginner way
print("pipeline started")
print("chunk created")

# Production professional way
logger.info("Pipeline started | document: ai_notes.pdf")
logger.debug("Chunk created | index: 0 | words: 21 | chars: 0-200")
```

The difference is not just style. Print statements disappear when your app runs on a server. Log statements are captured, stored, timestamped, and queryable.

---

### The Five Log Levels

Every log message has a severity level. Choose the level based on how serious the event is:

```
DEBUG    → fine detail only useful during development
           "Chunk 3 created: 200 chars, 24 words"
           Turn off in production to reduce noise

INFO     → normal milestones, system is working as expected
           "Pipeline started", "Document validated", "Response generated"
           Always on in production

WARNING  → something unexpected but not breaking
           "Chunk too small, skipping", "Rate limit approaching"
           Always on, investigate when frequent

ERROR    → something failed but system can continue
           "Embedding failed on chunk 5, retrying"
           Always on, investigate immediately

CRITICAL → system cannot continue, needs human intervention
           "API key missing", "Database unreachable"
           Always on, wake someone up at 3am
```

---

### Why Logging Matters in Production

```
Problem without logging:
→ app crashes on server at 3am
→ no terminal, no print output visible
→ no idea what happened
→ spend hours guessing
→ might never find root cause

Solution with logging:
→ every event timestamped and recorded
→ search logs for exact time of failure
→ see the chain of events leading to crash
→ fix in minutes not hours
```

---

### The Logger Setup Function

```python
def setup_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    logger = logging.getLogger(name)      # get or create logger by name
    logger.setLevel(level)                # minimum level to record

    if logger.handlers:                   # prevent duplicate handlers
        return logger                     # on repeated imports

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

logger = setup_logger("day02")            # module level logger
```

---

### What Each Format Part Means

```
[2026-03-26 23:08:00]  →  %(asctime)s    exact timestamp
INFO                   →  %(levelname)s  severity level
day02                  →  %(name)s       which module logged this
Pipeline started       →  %(message)s    your actual message
```

---

### Interview Answer — Logger

> "I use Python's logging module with five severity levels in every module. DEBUG for granular development detail, INFO for pipeline milestones, WARNING for quality issues, ERROR for recoverable failures, and CRITICAL for system stopping problems. In production I route logs to centralized monitoring tools like Datadog so I can query failures across all running instances and set up alerts on error rates."

---

---

## Section B — Generators

### What Is a Generator

A generator is a function that produces values one at a time using `yield` instead of returning everything at once with `return`.

```python
# Normal function — everything computed and loaded into RAM at once
def get_chunks(text: str) -> list[str]:
    result = []
    for i in range(0, len(text), 200):
        result.append(text[i:i+200])
    return result                    # entire list in RAM

# Generator — one value at a time, pauses between each
def stream_chunks(text: str):
    for i in range(0, len(text), 200):
        yield text[i:i+200]          # produces one, pauses here
```

---

### How yield Works

```python
def stream_chunks(text):
    yield text[0:200]      # pause 1 — returns first chunk, freezes
    yield text[200:400]    # pause 2 — returns second chunk, freezes
    yield text[400:600]    # pause 3 — returns third chunk, freezes
                           # StopIteration — no more values

gen = stream_chunks(document)
chunk1 = next(gen)         # runs to pause 1, returns chunk 1
chunk2 = next(gen)         # resumes from pause 1, runs to pause 2
chunk3 = next(gen)         # resumes from pause 2, runs to pause 3
```

The generator remembers exactly where it stopped between calls. A normal function has no memory between calls.

---

### Memory Comparison

```
Normal function with 500 page PDF:
→ 10,000 chunks loaded into RAM simultaneously
→ 1 user = manageable
→ 50 users = 500,000 chunks in RAM
→ server crashes

Generator with 500 page PDF:
→ 1 chunk in RAM at any moment
→ processed and discarded before next loads
→ 1 user = 1 chunk in RAM
→ 50 users = 50 chunks in RAM
→ server stays up
```

---

### Dataclass vs Pydantic

Used `ProcessedChunk` as a dataclass, not a Pydantic model:

```python
@dataclass
class ProcessedChunk:
    index: int
    content: str
    char_start: int
    char_end: int
    word_count: int
    is_meaningful: bool
```

```
Pydantic BaseModel:
→ validates data types on creation
→ converts to JSON easily
→ use at API boundaries — external data coming in
→ slower, more overhead

Dataclass:
→ no validation — you trust the data
→ lightweight and fast
→ use for internal pipeline data
→ data you created yourself
```

Rule of thumb:
```
Data coming FROM outside  → Pydantic    (validate it)
Data created BY your code → Dataclass   (trust it)
```

---

### Overlap — Why It Matters

```
chunk_size = 200, overlap = 40

Chunk 0: chars   0 → 200
Chunk 1: chars 160 → 360   ← starts 40 chars before chunk 0 ends
Chunk 2: chars 320 → 520   ← starts 40 chars before chunk 1 ends

A sentence starting at char 185 and ending at char 215:
→ appears in both Chunk 0 and Chunk 1
→ never split in half
→ context preserved at every boundary
```

Without overlap, sentences at chunk boundaries lose meaning. With overlap, every sentence appears complete in at least one chunk.

---

### Interview Answer — Generators

> "I use generators in my RAG pipeline for document chunking because documents can be arbitrarily large. A generator yields one chunk at a time keeping memory usage constant regardless of file size. Without generators, loading all chunks simultaneously for a 500 page PDF under concurrent load would crash the server. The generator keeps memory flat whether processing one user or one thousand."

---

---

## Section C — Retry Decorator

### What Is a Decorator

A decorator is a function that wraps another function to add behaviour without modifying the original function's code.

```python
@retry(attempts=3, delay=1.0)    # ← decorator applied here
def call_openai(prompt):
    return openai.chat(prompt)   # ← original function unchanged
```

The `@retry` line says: before and after `call_openai` runs, apply the retry logic.

---

### The Problem It Solves

```
Without decorator — copy pasted retry logic everywhere:
def embed_text(text):
    for attempt in range(3):       # copy pasted
        try:
            return openai.embed(text)
        except: time.sleep(1)

def generate_response(prompt):
    for attempt in range(3):       # copy pasted again
        try:
            return openai.chat(prompt)
        except: time.sleep(1)

Change retry logic = edit every single function
```

```
With decorator — written once, applied anywhere:
@retry(attempts=3, delay=1.0)
def embed_text(text):
    return openai.embed(text)      # clean

@retry(attempts=3, delay=1.0)
def generate_response(prompt):
    return openai.chat(prompt)     # clean

Change retry logic = edit one place only
```

---

### Exponential Backoff

```
Fixed delay — naive:
attempt 1 fails → wait 1s
attempt 2 fails → wait 1s
attempt 3 fails → wait 1s
1000 users retry simultaneously every second
→ hammers already overloaded API
→ makes it worse

Exponential backoff — production standard:
attempt 1 fails → wait 1s
attempt 2 fails → wait 2s    (1 × 2.0)
attempt 3 fails → wait 4s    (2 × 2.0)
attempt 4 fails → wait 8s    (4 × 2.0)
retries spread over time
→ overloaded API gets breathing room
→ success rate dramatically higher
```

---

### `functools.wraps` — Why It Matters

```python
@functools.wraps(func)     # preserves original function identity
def wrapper(*args, **kwargs):
    ...

# Without functools.wraps:
print(embed_text.__name__)  # → "wrapper"     ← wrong
print(embed_text.__doc__)   # → None           ← lost

# With functools.wraps:
print(embed_text.__name__)  # → "embed_text"  ← correct
print(embed_text.__doc__)   # → your docstring ← preserved
```

In production tracebacks you want to see `embed_text failed` not `wrapper failed`. `functools.wraps` preserves the original function identity through decoration.

---

### Three Scenarios

```
First attempt succeeds:
→ runs normally, no delay, no retry
→ user never knows decorator exists

Partial failure then recovery:
→ attempt 1 fails → wait 1s
→ attempt 2 fails → wait 2s
→ attempt 3 succeeds
→ user gets response, never knew it failed

Total failure:
→ all attempts exhausted
→ decorator re-raises original error
→ caller catches it and handles gracefully
```

---

### Interview Answer — Retry Decorator

> "I use a retry decorator with exponential backoff on every external API call. LLM APIs fail regularly due to rate limits and transient network issues. The decorator wraps the function, catches specific exception types, waits with exponentially increasing delays, and re-raises only after all attempts are exhausted. Exponential backoff prevents the thundering herd problem where simultaneous retries make an already overloaded API worse. I use functools.wraps to preserve original function identity so tracebacks remain readable in production."

---

---

## Section D — Exception Handling

### What Is an Exception Hierarchy

A custom exception hierarchy is a structured family of error types where each type represents a specific failure with a specific recovery strategy.

```python
Exception                        ← Python base
    └── AIEngineerError          ← our base
            ├── AuthenticationError    → stop immediately
            ├── RateLimitError         → wait and retry
            ├── DocumentValidationError→ reject input
            ├── EmbeddingError         → skip chunk
            └── VectorDBError          → queue for retry
```

---

### Why Not Just Use Exception

```python
# Wrong — catches everything the same way
try:
    call_api()
except Exception as e:
    print("something went wrong")
    # wrong API key?     → retry will never help
    # rate limit?        → should retry with backoff
    # bad document?      → tell user what to fix
    # all treated same   → wrong response every time

# Right — each error handled appropriately
except AuthenticationError:    → stop, alert developer
except RateLimitError:         → wait, retry
except DocumentValidationError:→ reject, tell user
except EmbeddingError:         → skip chunk, continue
```

---

### Order of Except Blocks Always Matters

```python
# Most specific first — always
except AuthenticationError:      # ← specific
except RateLimitError:           # ← specific
except EmbeddingError:           # ← specific
except AIEngineerError:          # ← catches all custom errors
except Exception:                # ← last resort, catches everything
```

If you put `except Exception` first it catches everything and specific handlers never run. Most specific exception always comes first.

---

### Input Validation — First Line of Defense

```python
def validate_document(content: str, min_length: int = 50) -> str:
    # Check 1: document must exist
    if not content or not content.strip():
        raise DocumentValidationError("Document is empty.")

    # Check 2: minimum length
    if len(content.strip()) < min_length:
        raise DocumentValidationError(
            f"Document too short: {len(content)} chars. "
            f"Minimum required: {min_length} chars."
        )

    # Check 3: readable text not binary
    non_text_ratio = sum(
        1 for c in content if not c.isprintable()
    ) / len(content)
    if non_text_ratio > 0.3:
        raise DocumentValidationError("Document appears corrupted.")

    return content.strip()
```

Rule: validate at the boundary before anything expensive runs. Bad data never enters the pipeline.

---

### PipelineResult — Structured Response

```python
class PipelineResult(BaseModel):
    success: bool
    chunks_processed: int = 0
    chunks_failed: int = 0
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    result: Optional[str] = None
```

Every pipeline run returns this structure. Success or failure. Always the same shape.

```python
# Caller is always clean
result = run_rag_pipeline(document)
if result.success:
    return result.result
else:
    return result.error_message
```

Exception handling complexity stays inside the pipeline. Callers stay clean.

---

### Mocking — Simulating Errors for Testing

In Day 2 you used `if` conditions to simulate API errors:

```python
if scenario == "auth_error":
    raise AuthenticationError(...)
```

This is called a **mock** — a deliberate simulation of real behavior for testing purposes. You cannot force a real API to give you a rate limit error on demand. A mock lets you test every error scenario reliably without a real API key.

```
Day 2:  if conditions simulate errors    → learning and testing
Day 5:  real OpenAI SDK raises errors    → production
Always: same exception handlers work     → no changes needed
```

---

### Interview Answer — Exception Handling

> "I build custom exception hierarchies for AI pipelines because different failures need completely different responses. An authentication error means stop immediately. A rate limit means wait and retry with backoff. A validation error means reject the input with a clear message. A malformed response means log and skip that chunk without crashing the pipeline. I never use bare except clauses because catching everything the same way hides bugs and makes debugging impossible in production."

---

---

## The Four Concepts Together — One Production Shield

```
User uploads document
        ↓
validate_document()          Section D — bad data rejected at door
        ↓
stream_document_chunks()     Section B — memory stays flat
        ↓
@retry embed_chunk()         Section C — API failures auto recovered
        ↓
parse_api_response()         Section D — malformed responses caught
        ↓
PipelineResult returned      Section D — structured always
        ↓
logger records everything    Section A — full audit trail
```

Remove any one layer and the system becomes fragile. Together they form a system that handles real production conditions reliably.

---

## Key Terms Quick Reference

```
Logger          → professional print replacement with levels and timestamps
DEBUG           → fine detail, development only
INFO            → normal milestones
WARNING         → unexpected but not breaking
ERROR           → failed but recoverable
CRITICAL        → system cannot continue

Generator       → function using yield, produces one value at a time
yield           → pause point, returns value, resumes on next call
Overlap         → shared characters between consecutive chunks

Decorator       → function that wraps another function
Exponential backoff → delay doubles after each retry
functools.wraps → preserves original function identity

Custom exception → named error type with specific recovery strategy
Exception hierarchy → family of related error types
Mock            → deliberate simulation of real behavior for testing
PipelineResult  → structured response model for pipeline outcomes
Validation      → checking input quality before processing begins
```

---

