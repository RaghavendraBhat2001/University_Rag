# uniRAG — Viva Defense Master Handbook
### Comprehensive Guide to Viva Preparation, Internal Mechanics of RAG & Local LLMs

> **For:** Raghavendra Chandrakantha Bhat (Reg No: P03NK24S126033)  
> **Degree:** Master of Computer Applications (MCA), Bangalore University  
> **Project:** *uniRAG — Offline Study Assistant for University Students*  
> **Guide:** Dr. Muralidhara B.L.

---

## Table of Contents
1. [The 2-Minute Elevator Pitch (How to Introduce Your Project)](#1-the-2-minute-elevator-pitch)
2. [What is RAG? (Explain it to Any Examiner)](#2-what-is-rag)
3. [Deep-Dive: How LLMs Work & Why Inference Takes Time](#3-deep-dive-how-llms-work--why-inference-takes-time)
4. [Vector Embeddings & FAISS Search Explained Simply](#4-vector-embeddings--faiss-search-explained-simply)
5. [The Core Philosophy: "The LLM Writes, Code Counts & Checks"](#5-the-core-philosophy)
6. [Step-by-Step Walkthrough: What Happens When a Student Hits "Ask"](#6-step-by-step-walkthrough)
7. [Mathematical Formulas Used in uniRAG](#7-mathematical-formulas-used-in-unirag)
8. [Tough Viva Questions & Bulletproof Scripted Answers](#8-tough-viva-questions--bulletproof-scripted-answers)
9. [Flawless 7-Minute Live Demo Script](#9-flawless-7-minute-live-demo-script)

---

## 1. The 2-Minute Elevator Pitch

When the external examiner says: *"Raghavendra, tell us about your project and what problem it solves."*

### Say This Clearly:
> "Respected examiners, **uniRAG** is an **offline, retrieval-augmented study assistant** designed specifically for university students—in our case, the MCA programme of Bangalore University.
>
> When students prepare for semester exams, they face two huge problems:
> 1. Their syllabus, textbooks, and past question papers are scattered across PDFs and paper photocopies.
> 2. General chatbots like ChatGPT or Gemini are not reliable for exam preparation: they require active internet, send sensitive student data to external servers, often hallucinate answers outside our specific university syllabus, and cannot accurately tell a student *how many times* a question has been asked in real university exams.
>
> **uniRAG solves this completely offline on a single laptop:**
> - It answers student questions **only from Bangalore University notes, syllabus, and past question papers**.
> - It features a **two-part answer architecture**: a textbook-grounded explanation plus an intuitive real-world analogy.
> - Most importantly, it uses a design principle: **The LLM writes the explanation, but deterministic Python code counts, computes trends, and ranks repeated questions.** The model is never allowed to guess or hallucinate exam numbers.
> - Everything runs locally using **Mistral 7B**, **nomic-embed-text**, **FAISS**, and **Streamlit**—with zero subscription cost and zero data leakage."

---

## 2. What is RAG?

### 2.1 The Problem with Raw Large Language Models (LLMs)
An LLM (like Mistral, LLaMA, or GPT) is essentially a giant statistical neural network trained on public web text up to a fixed training date.
- **Knowledge Cut-off:** It does not know private or localized data (such as Bangalore University's 2020–2026 MCA question papers or professors' notes).
- **Hallucination:** When asked about something it doesn't know, an LLM attempts to generate plausible-sounding text, often producing completely false facts with high confidence.

### 2.2 How RAG Works (The Open-Book Exam Analogy)
Tell the examiner:
> *"Think of a standard LLM as a student taking a closed-book exam purely from memory. They might get confused or make things up.  
> **RAG (Retrieval-Augmented Generation)** turns the test into an **open-book exam**. When a question is asked, our system first walks into the university library, pulls the exact 3 most relevant paragraphs from our verified notes, places them on the desk in front of the LLM, and instructs it: 'Answer the question using ONLY these retrieved paragraphs.' If the paragraphs aren't there, refuse to answer."*

---

## 3. Deep-Dive: How LLMs Work & Why Inference Takes Time

*(This is the exact topic your guide stressed you must master!)*

### 3.1 What is Mistral 7B? What does "7B" mean?
- **7B means 7 Billion Parameters.**
- In a neural network, a "parameter" is a numerical weight (a floating-point number) that connects artificial neurons across layers.
- 7 billion weights represent the learned knowledge and grammatical capabilities of the model.
- At full precision (16-bit float / FP16), each parameter takes 2 bytes $\rightarrow 7 \times 10^9 \times 2 = 14 \text{ GB}$ of RAM just to store the weights.
- **Quantization (4-bit GGUF in Ollama):** Because our laptop has 16 GB RAM and no dedicated GPU VRAM, the model is **quantized** to 4 bits (`Q4_K_M`), shrinking the model in memory to approximately **4.1 GB**, allowing it to run smoothly alongside the OS and Streamlit.

---

### 3.2 Why does the LLM take 1 to 3 minutes to generate on a laptop CPU?
Examiners love asking: *"Why is it slow? Why does it take minutes instead of milliseconds?"*

Give them this precise, technical explanation:

#### A. Autoregressive Generation (Token-by-Token)
LLMs do **not** output entire sentences at once. They generate text **one single token (word/subword) at a time**:
1. It reads the prompt $\rightarrow$ predicts token 1.
2. It feeds (prompt + token 1) back into the network $\rightarrow$ predicts token 2.
3. It repeats this loop sequentially until it hits an end-of-sequence token or token limit.
4. If an answer is 500 tokens long, the neural network must execute **500 sequential forward passes**.

#### B. The CPU Memory Bandwidth Bottleneck (The Real Technical Cause)
- In each forward pass to generate **one single token**, the computer's CPU must read **all 4.1 gigabytes of model weights from system RAM** through the memory bus into CPU cache.
- Modern laptops have dual-channel DDR4/DDR5 RAM with a memory bandwidth of roughly **25 to 40 GB/second**.
- If the model is $\approx 4.1\text{ GB}$, the theoretical maximum generation speed on CPU is:
$$\text{Speed} \approx \frac{30\text{ GB/s RAM Bandwidth}}{4.1\text{ GB Model Weights}} \approx 7 \text{ to } 8 \text{ tokens per second}$$
- In contrast, a modern GPU (like NVIDIA RTX 4090) has high-speed VRAM bandwidth of **1,000 GB/second** (generating 60–100 tokens/sec).
- Therefore, on a CPU-only laptop, generating an in-depth 10-mark answer (600–800 tokens) takes:
$$\text{Time} = \frac{700 \text{ tokens}}{7 \text{ tokens/sec}} \approx 100 \text{ seconds } (\approx 1.5 \text{ to } 2 \text{ minutes})$$

#### C. How uniRAG Optimizes and Mitigates This Latency:
1. **Streaming UI (`st.write_stream`):** Rather than forcing the student to stare at a blank screen for 90 seconds, tokens are streamed to the screen in real-time as they are computed.
2. **Marks-Based Token Capping (`MARKS_MAX_TOKENS` in `config.py`):**
   - 2 Marks $\rightarrow$ capped at 250 tokens ($\approx 25\text{ s}$)
   - 5 Marks $\rightarrow$ capped at 600 tokens ($\approx 60\text{ s}$)
   - 10 Marks $\rightarrow$ capped at 1,100 tokens ($\approx 110\text{ s}$)
   - Runaway generations are strictly prevented.
3. **Model Warmth & Keep-Alive (`LLM_KEEP_ALIVE = "30m"`):** Ollama keeps Mistral in memory for 30 minutes, avoiding the 20-second penalty of loading 4 GB from SSD on every query.
4. **Answer Caching (`ANSWER_CACHE = True`):** Identical questions retrieve previously computed answers instantly in under 0.1 seconds from `faiss_store/_answer_cache.json`.
5. **Zero-Latency Guards:** Out-of-syllabus questions are rejected in **0.00 seconds** by Python rules before any LLM call is even initiated.

---

## 4. Vector Embeddings & FAISS Search Explained Simply

### 4.1 What is an Embedding? (`nomic-embed-text`)
- Computers cannot perform math on raw English words. An embedding model converts a piece of text into a list of numbers (a vector).
- We use `nomic-embed-text`, which maps every text chunk into a **768-dimensional vector** (768 numbers).
- **Key Property:** Sentences with similar semantic meanings end up mathematically close to each other in this 768-dimensional space, even if they use completely different words!
  - *"What is a primary key?"* and *"Explain unique record identifier in relational tables"* have high semantic similarity.

### 4.2 Why Chunk Size 400 and Overlap 50?
- **Chunk Size = 400 characters ($\approx 60\text{--}80$ words):** Keeps each chunk focused on a single logical concept or definition. Large chunks (e.g. 2,000 characters) dilute semantic meaning, causing vector search to become imprecise.
- **Overlap = 50 characters:** Ensures that sentences or definitions occurring across chunk boundaries are not cut in half, preserving context.

### 4.3 Why FAISS? Why not SQL (e.g., MySQL / PostgreSQL)?
- Relational databases (SQL) search via exact keywords (`WHERE text LIKE '%queue%'`). They cannot understand semantic synonyms or conceptual similarity.
- **FAISS (Facebook AI Similarity Search):** An open-source, highly optimized C++ library designed specifically to search billions of high-dimensional vectors in microseconds.
- We use a **FAISS Flat L2 Index** (`IndexFlatL2`):
  - It computes the exact **Squared Euclidean Distance** between the query vector and every chunk vector.
  - Since our university notes collection is concise ($\approx 1,400\text{--}5,000$ chunks), a flat index provides **100% exact retrieval recall** with zero approximation error in under 5 milliseconds on CPU.

### 4.4 Why One FAISS Index Per Subject?
- If all subjects (Operating Systems, DBMS, Mathematics, AI) were dumped into one giant index, large subjects (with many notes) would crowd out smaller subjects during similarity ranking.
- Per-subject partitioning guarantees that an OS query only searches OS notes, eliminating cross-subject contamination.

---

## 5. The Core Philosophy: "The LLM Writes, Code Counts & Checks"

**Examiners frequently ask:** *"How do you calculate trends and most repeated questions? Does the LLM do that?"*

### Your Answer (Emphasize this strongly!):
> **"No! The LLM is never allowed to count, compute trends, or extract exam history.**
>
> LLMs are autoregressive probability models. If you ask an LLM: *'How many times did Quick Sort appear in Bangalore University papers between 2020 and 2026?'*, it will guess a number like '4 times' or '5 times' based on statistical patterns, which is a hallucination.
>
> In **uniRAG**, we enforce a strict separation of concerns:
> 1. **Code (Deterministic):** Extracts questions from papers into a structured JSON database (`_question_bank.json`), searches with fuzzy matching (`difflib`) and vector cut-offs, counts exact occurrences, filters by exam years, and plots the trend charts.
> 2. **LLM (Generative):** Only writes the conceptual explanation and the intuitive analogy based strictly on the retrieved text."

---

## 6. Step-by-Step Walkthrough: What Happens When a Student Hits "Ask"

1. **User Input:** Student submits: *"Explain deadlock and how often is it asked?"*
2. **Query Understanding (`src/query_understanding.py`):**
   - Mistral 7B (at `temperature=0`) parses the messy input into JSON:
     `{ "topic": "Deadlock", "intents": ["definition", "frequency"], "subject": "OS" }`
3. **Negative Syllabus Guard:**
   - Checks if *"Deadlock"* exists in the syllabus or past papers. If not, it halts in 0.00 seconds.
4. **Deterministic Exam Intelligence:**
   - Python code queries `faiss_store/_question_bank.json` for *"Deadlock"*.
   - Returns: *"Appeared 3 times (2021 [6M], 2023 [10M], 2025 [6M])."*
5. **Semantic Notes Retrieval:**
   - Embeds query into 768-D vector $\rightarrow$ searches OS FAISS index.
   - Retrieves top 3 chunks (Distance $0.38 \le 1.15$).
6. **Grounding & LLM Generation:**
   - Injects the 3 chunks into the system prompt.
   - Mistral 7B streams the 2-part explanation (Notes explanation + Real-world analogy).
7. **UI Presentation:**
   - Rendered as three beautiful cards in Streamlit.

---

## 7. Mathematical Formulas Used in uniRAG

### 1. Vector Distance (FAISS L2 Squared Euclidean Distance)
For query vector $q$ and stored chunk vector $v$ in 768-dimensional space:
$$D(q, v) = \sum_{i=1}^{768} (q_i - v_i)^2$$
- Lower distance = higher semantic similarity.
- **Notes Retrieval Cut-off:** $D \le 1.15$
- **Question Bank Matching Cut-off:** $D \le 0.75$

### 2. Fuzzy String Matching (Difflib Ratcliff-Obershelp Formula)
Used in `src/question_matcher.py` to match questions despite student typos or OCR noise:
$$\text{Similarity}(S_1, S_2) = \frac{2 \times M}{T}$$
- Where $M$ is the number of matching characters, and $T = |S_1| + |S_2|$ is total characters.
- Ratio $\ge 0.75$ counts as a near-exact match.

### 3. Classification Metrics (Calibration of Cut-off)
Used in `calibrate_cutoff.py` across 61 test queries to select the $0.75$ threshold:
$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$
$$F_1\text{-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = 0.757 \text{ at cut-off } 0.75$$

### 4. Password Security (PBKDF2-HMAC-SHA256)
Used in student registration and login:
$$\text{Key} = \text{PBKDF2}(\text{HMAC-SHA256}, \text{Password}, \text{Salt}, 100000 \text{ iterations}, 32 \text{ bytes})$$
- Evaluated in constant time using `hmac.compare_digest()` to prevent timing attacks.

---

## 8. Tough Viva Questions & Bulletproof Scripted Answers

### Q1: "Why did you use RAG instead of fine-tuning Mistral 7B?"
**Answer:**  
> "Fine-tuning updates neural network weights. However:
> 1. Fine-tuning a 7B model requires expensive cloud GPUs (A100/H100), making it impossible to run locally on a laptop.
> 2. University notes and papers update every semester. With RAG, when a new syllabus or paper arrives, we embed the PDF into FAISS in seconds—no re-training needed.
> 3. Fine-tuned models still hallucinate and cannot cite page numbers or files. RAG allows 100% auditable citations."

### Q2: "Can uniRAG hallucinate? How do you prevent it?"
**Answer:**  
> "I must be technically honest: **no LLM in the world has zero hallucination**, but in uniRAG we constrain it using **five multi-layer code guards**:
> 1. **Out-of-Syllabus Gate:** Rejects queries not found in syllabus/papers before calling the LLM.
> 2. **Distance Cut-off ($1.15$):** Drops retrieved chunks if semantic distance is too high.
> 3. **Topic-in-Context Check:** Verifies that the keywords of the question actually exist inside the retrieved notes text.
> 4. **Deterministic Counting:** The model is forbidden from answering exam frequency; Python code handles all numbers.
> 5. **Source Citation:** Every answer displays the exact source PDF filename so the student can verify."

### Q3: "What if a student types in Hinglish or has grammatical errors?"
**Answer:**  
> "Our Query Understanding module (`src/query_understanding.py`) uses Mistral 7B at `temperature=0` specifically prompted to normalize colloquial phrases and Hinglish (e.g. *'Binary search kitni baar aaya'* $\rightarrow$ Topic: `Binary Search`, Intent: `frequency`). Then fuzzy matching handles OCR and spelling variations."

### Q4: "Why did you use Streamlit and not React or Django?"
**Answer:**  
> "Streamlit is built specifically for Python machine learning workflows. It allows native generator streaming (`st.write_stream`) directly from Ollama's token buffer with zero WebSocket boilerplate, handles Altair chart serialization natively, and keeps the entire software stack in pure Python, making it easy to deploy on any university laptop."

### Q5: "What if an administrator uploads an invalid or malicious file?"
**Answer:**  
> "All file uploads go through `src/document_loader.py`, which sanitizes filenames using regex, strips path-traversal tricks (`../`), restricts file extensions strictly to `.pdf`, and enforces a password-protected administrator session."

---

## 9. Flawless 7-Minute Live Demo Script

Follow this exact order during your viva demonstration:

1. **Step 1: Student Login (1 min)**
   - Show the clean login screen. Log in as a student (`raghu@bu.ac.in`). Mention salted PBKDF2 password security and the 5-attempt lockout defense.
2. **Step 2: Study Assistant Query (2 mins)**
   - Select Subject: `Data Structures (DS)`.
   - Type Question: *"What is a Circular Queue and how often has it been asked?"*
   - Point out:
     - 📖 Card 1: Textbook-grounded notes definition.
     - 💡 Card 2: Simple real-world analogy.
     - 📊 Card 3: Deterministic Exam History badge (showing exact years and marks).
3. **Step 3: Negative Guard / Refusal Test (1 min)**
   - Ask an out-of-syllabus question: *"What is a Fenwick Tree?"*
   - Show how the system responds in **0.00 seconds**: *"Fenwick Tree does not appear in your MCA syllabus... declined."*
   - Explain to the examiner: *"This proves our guard stops hallucinations before wasting CPU cycles."*
4. **Step 4: Exam Intelligence Tab (2 mins)**
   - Switch to **Exam Intelligence**.
   - Show the **Topic Trend Chart** (e.g., *Normalization* in DBMS across 2020–2025).
   - Show the **Most Repeated Questions** view.
   - Click **Download Solutions PDF** $\rightarrow$ show that it instantly downloads model answers for every repeated question in that subject.
5. **Step 5: Conclude (1 min)**
   - Show `test_uniRAG.py` in terminal running **32 unit tests in ~3 seconds (All Passing)** to prove software engineering rigor.

---
*Good luck with your viva! You built a solid, well-engineered, and academically rigorous project.*
