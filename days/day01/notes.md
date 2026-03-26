---

# Day 1 — `notes.md`

## Question 1: What does `async/await` actually do differently compared to regular function calls?

**Answer to write in your own words:**
A regular function runs from start to finish and blocks everything else while it runs. `async/await` allows a function to **pause at the `await` keyword** and let other code run while it waits. In AI engineering this matters because LLM API calls can take 2–5 seconds each. Without async, calling OpenAI 3 times takes 15 seconds sequentially. With async and `asyncio.gather`, all 3 calls happen at the same time and finish in 5 seconds total. The program doesn't sit idle — it moves on to the next task while waiting for the slow one to complete.

---

## Question 2: Why would you use a Pydantic `BaseModel` instead of a plain Python dictionary?

**Answer to write in your own words:**
A plain dictionary accepts anything — you can put a number where text is expected and Python won't complain until something breaks much later and is hard to trace. A Pydantic `BaseModel` acts like a strict contract — you define exactly what fields exist, what types they must be, and what defaults they have. If wrong data comes in, Pydantic raises a clear error immediately at the point of entry. In a RAG pipeline processing thousands of document chunks, catching one bad chunk immediately is far better than letting it silently corrupt your vector database.

---

## Question 3: What would happen if you committed your `.env` file to a public GitHub repo?

**Answer to write in your own words:**
Automated bots constantly scan GitHub for exposed API keys. Within minutes of pushing a `.env` file containing your OpenAI API key, bots would find it and start using it. This could result in hundreds or thousands of dollars in API charges billed to your account before you even notice. Some bots use stolen keys to run cryptocurrency mining or spam operations through AI APIs. OpenAI and other providers will suspend your account and you lose all your work. This is why `.env` must always be in `.gitignore` before your very first commit — not added later.

---

## Question 4: Look at the `chunk_text` function — where will this exact pattern show up in the RAG pipeline on Day 12?

**Answer to write in your own words:**
On Day 12, when a user uploads a PDF or text document, the very first step is splitting it into chunks before storing in a vector database. The `chunk_text` function does exactly this. The `chunk_size` controls how much text each embedding represents, and the `overlap` ensures that sentences spanning two chunks don't lose context at the boundary. In the RAG pipeline the flow will be: load document → `chunk_text()` → embed each chunk → store in ChromaDB → retrieve relevant chunks at query time → pass to LLM with the user's question.

---

## Bonus — Key Terms to Remember from Day 1

Write these definitions in your own words as a quick reference:

**Virtual Environment:** An isolated Python installation for a specific project so that library versions don't conflict between projects on the same machine.

**`requirements.txt`:** A file that lists every library your project depends on with their versions, so anyone can recreate your exact environment with `pip install -r requirements.txt`.

**Type Hints:** Labels added to function parameters and return values that tell you and your tools what data type is expected. They don't enforce anything at runtime but make code readable and enable editor autocomplete and static analysis tools like mypy.

**Pydantic BaseModel:** A class that defines a strict data structure with type validation. When you create an instance, Pydantic checks all fields match their declared types before allowing the object to be created.

**Generator / yield:** A function that produces values one at a time using the `yield` keyword instead of returning everything at once. Memory-efficient for large data processing.

**async/await:** Python keywords for writing non-blocking code. `async def` declares a function that can be paused. `await` marks where the pause happens while waiting for a slow operation like an API call.

**`.env` file:** A text file containing secret keys and configuration values that should never be committed to version control.

**`.gitignore`:** A file telling Git which files and folders to never track or commit, keeping secrets and temporary files out of your repository.

---

