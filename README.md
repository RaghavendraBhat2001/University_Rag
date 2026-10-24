# uniRAG — Offline Study Assistant for University Students

uniRAG answers a student's questions **only from the university's own notes, syllabus and past
question papers**, and it runs **100% locally** — no internet, no cloud API, no data leaves the
machine. It was built for MCA students at Bangalore University, and each university runs its own
separate install.

## What it does

| Feature | How it works |
|---|---|
| Explain a topic | Retrieves the student's own notes and writes a two-part answer: a textbook-style explanation and a plain-language analogy |
| "How many times has X been asked?" | Counts via **hybrid matching** (exact, subtopic & question vectorstore) over extracted exam questions — the language model never counts |
| Exact past questions, by year and marks | Shown verbatim from the extracted question bank |
| Trends, most-repeated questions, high-weightage topics | Computed from the same question bank (Exam Intelligence tab; a subject must be selected, so every number is per subject) |
| Study guidance | Priority tier and marks pattern derived from real paper data |
| Lab programs | Code written in C, following the lab manual's approach |
| Question Paper tab | Upload a paper → questions are extracted → each is answered → download all answers |
| Most Repeated Questions + solutions | Pick a subject: the questions repeated across two or more exam sittings are listed, and one PDF bundles a model answer for each (Exam Intelligence tab) |
| Scanned PDFs | Pages with no text layer are read with Tesseract OCR |
| Roles | Students self-register and log in; an administrator manages documents and rebuilds |

## Requirements

- Windows/Linux/macOS, Python 3.12 (developed on Windows 11), 16 GB RAM. No GPU needed (CPU inference, typically 1–4 minutes per answer on this CPU (median about 2 min in the eval run)).
- [Ollama](https://ollama.com) with two models:
  ```
  ollama pull mistral:7b
  ollama pull nomic-embed-text
  ```
- [Tesseract-OCR](https://github.com/UB-Mannheim/tesseract/wiki) (a separate system install, not just a pip package). Set `TESSERACT_CMD` in `config.py` if it is not on your PATH.
- Python packages: `pip install -r requirements.txt` (or `requirements.lock.txt` for the exact versions that were tested).

## Folder layout

```
University_Rag/
├── app.py                     Streamlit UI (all tabs, login, admin setup)
├── config.py                  Paths, model names, chunk sizes, admin password
├── eval_questions.py          Regression + calibration test harness
├── repeated_solutions.py      Solutions store + status / generate / export commands for the Most Repeated Questions bundle
├── build_solutions.py         Curated answers, written to data/repeated_solutions/solutions_index.json
├── rebuild_questions.py       Rebuilds the question bank in seconds (notes indexes untouched)
├── students.json              Student accounts (salted PBKDF2 password hashes)
├── university_settings.json   University name / department (written by the Setup tab)
├── src/
│   ├── document_loader.py     PDF loading, OCR fallback, question extraction, year detection
│   ├── query_understanding.py Turns a messy question into structured intents + topic
│   ├── rag_pipeline.py        Retrieval, guards, answer assembly
│   ├── question_index.py      Question bank (JSON) + its small semantic index
│   ├── question_matcher.py    Exact → subtopic → semantic question matching
│   └── exam_intelligence.py   Frequency, trends, ranking, marks analysis, syllabus check
├── data/bangalore_university_mca/
│   ├── notes/SEM<n>/(ABBR) Full Subject Name/*.pdf
│   ├── question_papers/SEM<n>/(ABBR) Full Subject Name.pdf     (several years merged per PDF)
│   └── syllabus/
├── data/repeated_solutions/   solutions_index.json (curated) + solutions_generated.json (model-generated, resumable)
└── faiss_store/               Generated indexes — safe to delete and rebuild
```

## Running it

```
venv\Scripts\activate
set ADMIN_PASSWORD=choose-a-password      (PowerShell: $env:ADMIN_PASSWORD="choose-a-password")
streamlit run app.py
```

Keep Ollama running in the background. If `ADMIN_PASSWORD` is not set, the fallback in `config.py`
is used — **change it before letting anyone else use the app.**

**First run / after adding documents:** log in as administrator → *Setup* → **Rebuild everything now**.
This OCRs scanned pages and embeds the notes, so on a CPU-only laptop it can take a long time.
Do it once after uploading, not per file.

## Maintenance

| Task | Command |
|---|---|
| Re-extract questions after changing the extractor | `python rebuild_questions.py` (seconds; notes indexes are preserved) |
| Re-extract questions by re-reading and re-OCRing the paper PDFs | `python rebuild_question_bank_only.py` (slower) |
| Check year detection / subject grouping | `python diagnose_corpus.py`, `check_subject_years.py`, `inspect_paper_years.py` |
| Test OCR on one page | `python test_ocr.py` |
| Full rebuild from scratch | delete `faiss_store/`, then rebuild from the Setup tab |
| Solutions coverage per subject | `python repeated_solutions.py status` |
| Generate missing solutions with the local model (hours, resumable) | `python repeated_solutions.py generate [SUBJECT ...]` |
| Write one clean .txt per subject | `python repeated_solutions.py export` |
| Add curated answers | `python build_solutions.py`, then the `add_*_solutions.py` scripts (they merge into the same index and win over generated answers) |

Changes to `rag_pipeline.py`, `query_understanding.py`, `exam_intelligence.py` or `app.py` need **no**
rebuild. Changes to how documents are chunked or what metadata each chunk carries do.

## Testing

`python eval_questions.py` runs 29 fixed cases (44 checks) through the same path the chat tab uses and reports
**PASS / FAIL / WARN** per rule (no topic carry-over into unrelated questions, out-of-scope topics
refused, sitting counts consistent with listed years, lab code in C, and so on). Answers are saved to
`eval_report.txt`. `--only <text>` runs a subset, `--calibrate` prints retrieval distances,
`--list` shows the case names.

## How an answer is produced

1. **Query understanding** — the model parses the question into JSON (topic, intents such as
   definition / frequency / exact questions / trend / ranking / lab program, subject). This call
   runs at temperature 0, so the same question always parses the same way. Deterministic code then
   corrects what a small model gets wrong: a topic is carried over from the previous message only if
   the new question actually points back at it.
2. **Retrieval** — notes come from a **per-subject FAISS index** (a merged index is the fallback for
   "All / Auto-detect"). Question papers are extracted into a structured **1,426-question bank** and
   searched using a **4-level hybrid matcher** (Exact / Subtopic / Dedicated Question Vectorstore / Filters)
   with automatic cross-subject fallback.
3. **Guards before the model writes anything** (below).
4. **Synthesis** — the model writes only the explanation/analogy or code. Exam counts, years, marks and
   priority tiers come from code, not from the model.
5. **Display** — the answer is split into sections (concept, exam history, trend, guidance, …), each
   shown in its own box.

## Guards against wrong answers

| Guard | What it stops |
|---|---|
| Relevance cutoff (L2 distance ≤ 1.15) | Answers built from unrelated text |
| Topic-in-context check | A topic whose own words never appear in the retrieved notes being answered from the model's memory |
| Out-of-syllabus gate (`SYLLABUS_GATE`) | Plain "what is X" questions about topics absent from both syllabus and past papers |
| Deterministic counting | Any invented frequency or year |
| Follow-up check | "How many subjects in 1st sem" inheriting the previous topic |
| Figure/table reference stripping | "Figure 3.1 shows…" when the student cannot see it |
| Empty-input guard | A bare "?" producing a random answer |

## Configuration

| Setting | Where | Meaning |
|---|---|---|
| `LLM_MODEL`, `EMBEDDING_MODEL` | `config.py` | Ollama model names |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` / `TOP_K` | `config.py` | 400 / 50 / 3 |
| `MARKS_MAX_TOKENS` | `config.py` | Caps answer length per marks level (bounds CPU latency) |
| `ADMIN_PASSWORD` | environment variable | Administrator login |
| `MAX_RELEVANT_DISTANCE` | `rag_pipeline.py` | Retrieval cutoff (1.15) |
| `SYLLABUS_GATE` | `rag_pipeline.py` | `"block"` or `"off"` |
| Answer temperature | `rag_pipeline.py` | 0.1 for answers, 0 for query parsing |

## Security notes

Student passwords are stored as salted PBKDF2-HMAC-SHA256 hashes (standard library only). Registration
is open and there is no email verification or password reset. Five wrong passwords lock that account for 60 seconds, and *Switch role* clears the previous user's chat and session. This is appropriate for an
offline single-machine deployment, **not** for a server exposed to the internet.

## Known limitations

- **Counting is literal.** A topic worded differently in the papers ("normalisation" vs a question that
  never uses the word) can be missed. This is deliberate: it is the price of never letting the model
  invent a count. Query understanding maps common synonyms and abbreviations to one canonical topic,
  which reduces but cannot remove this.
- **The model can still state a wrong fact** in an explanation (for example a wrong complexity case).
  Guards limit what it is asked to write; they cannot verify every sentence. Treat explanations as
  study aids and confirm key facts in the source book, which is listed under *Sources*.
- **OCR noise.** Scanned papers occasionally leave stray characters in a question; page-turn and scanner
  watermarks are stripped for display.
- **Some questions have no detected year** and are excluded from year-by-year answers.
- **Semantic question matching** uses a fixed distance cutoff that has not been calibrated against a
  labelled set, so an occasional loosely related question can appear marked `[Semantic Match]`.
- **No diagrams.** Figures are not extracted or drawn; requests to draw something are declined.
- **Comparisons and lab code** depend on retrieval finding the right chunk; similarly named topics
  (binary search vs binary search tree) can still be confused.
- **CPU-only** inference is slow, especially for the Question Paper tab (one model call per question).
- **Solutions bundle quality.** Of the 125 repeated questions across 16 subjects (ML and CNS have none), 67 have curated answers (drafted with an AI assistant and checked; numerical results were recomputed in code) and 58 were generated offline by the local 7B model from the notes. The model is unreliable on calculation and proof questions, so 41 of its first 99 answers were replaced; the 58 that remain are mostly descriptive theory and were only skimmed. Questions that refer to a figure missing from the extracted text get the method plus a worked example.