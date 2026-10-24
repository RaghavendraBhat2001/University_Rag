# uniRAG — Viva Notes

One idea to lead with: **the language model writes; code counts and checks.** Everything numeric
(how many times, which years, marks) is computed from the question papers by deterministic code, and
several code-level guards stop the model from answering when the notes don't support it.

---

## 60-second explanation

uniRAG is an offline study assistant. A student asks a question in ordinary, even messy, language.
The system understands what is being asked, finds the relevant passages in the university's own notes,
and writes a short explanation — and separately, from the past question papers, it tells the student
how often the topic was asked, in which years, and for how many marks. It runs entirely on a laptop
(Ollama with Mistral 7B, FAISS for search, Streamlit for the interface), so no student data leaves
the machine.

---

## Design questions

**Why RAG and not fine-tuning a model?**
Notes and papers change every year, and fine-tuning on a laptop CPU isn't practical. RAG lets the
university add a PDF and use it immediately, and every answer can point at its source files.

**Why fully offline?**
Privacy and cost: student queries stay local, there is no API bill, and it works without internet.
The trade-off is a smaller model and slower answers (typically 1–4 minutes per answer on this CPU (median about 2 min in the eval run), 16 GB RAM, CPU only).

**Why Mistral 7B?**
I tested smaller models (llama3.2:3b, qwen2.5:3b). They were faster but noticeably worse at writing
code, so I kept Mistral 7B. It is the largest model that runs acceptably on this hardware.

**Why one FAISS index per subject?**
A single combined index mixed subjects: a small subject's notes got crowded out by a much larger
one, and searching everything was slower. Per-subject indexes fix both; a merged index is the
fallback for "All / Auto-detect".

**How are question papers indexed and matched?**
Question papers are extracted into a structured Question Bank of 1,426 clean individual questions
(storing subject, semester, year, marks, and question text). Instead of lumping entire pages into a
generic notes index, questions are embedded into their own dedicated Question Vectorstore
(`_question_bank_faiss`). Retrieval uses a 4-level hybrid matcher:
1. Exact / Near-Exact keyword match (substring and high-ratio fuzzy match)
2. Subtopic match (e.g. matching 'Circular Queue' or 'Linear Queue' specifically)
3. Semantic Vector search via `nomic-embed-text` (finds conceptually related questions even when worded differently)
4. Marks & Year filters (e.g. 10-mark questions only)

**What happens if a student leaves the wrong subject selected in the dropdown? (Cross-Subject Fallback)**
If a student selects "Machine Learning" in the UI dropdown but asks a Data Structures question like
"What is Queue?", the system does not fail or return 0. The matcher detects that the ML question pool
has 0 matches and automatically falls back to search across the entire university question bank,
finding all 7 Data Structures questions. The notes retrieval similarly falls back to the authoritative
subject's notes. (The Exam Intelligence tabs are different: they require a subject, so counts, trends and the
solutions bundle are always per subject.)

**How do you get years out of scanned papers?**
Several years are merged into one PDF per subject, so the filename can't give the year. OCR reads each
page, and the year comes from the printed exam header ("Examination, June/July 2024"), carried
forward across continuation pages.

**What stops it answering something outside the MCA syllabus?**
A separate gate checks plain "what is X" questions against the syllabus PDF (read via OCR, since
it's a scan) and against past question papers. If X appears in neither, it declines instead of
answering from an unrelated book in the corpus — the notes folder isn't curated to be
syllabus-only, so without this a general-knowledge or unrelated-subject question could get a
confident answer sourced from the wrong book. Genuinely obscure syllabus content — for example a
cryptography topic worded unusually — is still answered normally if the syllabus or a past paper
actually mentions it; the gate only blocks when neither does.

**How does it cope with badly written questions?**
The first step is a model call that turns the question into structured fields (topic, what is being
asked, subject). It handles typos, abbreviations and mixed languages ("queue kitni baar aaya").
Then code corrects the cases where a small model is unreliable — for example, it only treats a
message as a follow-up if it actually refers back ("that", "it") or has no topic of its own.

---

## "How do you stop it hallucinating?" — the honest answer

**What is guarded (in code):**
1. A **relevance cutoff** on retrieval distance — no answer from unrelated text.
2. A **topic-in-context check** — if the topic's own words never appear in the retrieved notes, the
   system says it couldn't find it instead of answering from the model's memory.
3. An **out-of-syllabus gate** — a plain "what is X" for something in neither the syllabus nor any
   past paper is declined.
4. **Deterministic numbers** — counts, years and marks never come from the model.
5. The model is told not to mention exam frequency, importance or figure numbers in its prose, and
   figure references are removed by code.

**What is not guaranteed:** the model can still state a wrong fact inside an otherwise grounded
explanation. Example found in testing: it once gave the wrong time-complexity case for quick sort.
Guards limit what it is asked to write; they cannot fact-check every sentence. That is why every
answer lists its source files and why I would present the explanations as study aids.

Saying this plainly is stronger than claiming zero hallucination.

---

## Limitations to state before being asked

- Counting is literal, so a topic worded differently in a paper can be missed (deliberate trade-off).
- Some scanned questions have OCR noise or no detected year.
- The semantic-match cutoff is not calibrated on a labelled set. I measured retrieval distance on 8 topics from my notes (0.29–0.52); off-topic questions are mostly stopped earlier by the syllabus gate, and two that reached retrieval were at 0.66–0.69. The ceiling of 1.15 is deliberately permissive because precision comes from the syllabus gate and the grounding check.
- No diagrams.
- CPU-only, so slow on long batches (Question Paper tab).
- Open student registration with no email verification — fine for an offline single machine, not for
  an internet-facing server.
- Most Repeated Questions bundle: 67 of 125 answers are curated and checked; 58 were generated by the local
  7B model from the notes and are mostly descriptive theory that I only skimmed. The model is unreliable on
  calculation and proof questions (41 of its first 99 answers were wrong or did not solve the question and were
  replaced).
- Questions that refer to a figure missing from the extracted text get the method plus a worked example.

**Future work:** calibrate the matching cutoffs on a labelled question set; add a small verifier
that checks each explanation sentence against the retrieved text; GPU support for speed; diagram
extraction; a proper account system if deployed on a network.

---

## How it was evaluated

A regression suite (`eval_questions.py`, 29 fixed cases / 44 checks) runs through the same path the chat
uses and checks explicit rules: unrelated questions don't inherit the last topic, out-of-scope topics
are refused, counts agree with the years listed, lab code is in C, identical questions give identical
answers. Failures found during testing were fixed and the suite re-run. Known model-quality risks are
reported as warnings rather than hidden.

---

## Numbers (from the last startup log — re-check before the viva)

22 subject indexes (18 theory subjects with question papers + 4 lab subjects) · 630 question-paper chunks · 1426 extracted exam questions · chunk size 400 /
overlap 50 · top-3 retrieval · relevance cutoff 1.15 · Mistral 7B + nomic-embed-text.

---

## Live demo plan

**Safe, strong demos** (all tested):
1. `what is decision tree? how many times has it appeared? should I read it?` — shows the two-part answer, then exam history and guidance from real papers.
2. `queue kitni baar aaya hai exam me` — messy mixed-language input still resolved.
3. `explain hashing` → `what about its types` → `how many times has that appeared` — follow-ups.
4. `array vs linked list` — comparison covering both sides.
5. `write a program for binary search` — code in C.
6. `explain the French revolution` — shows the system **refusing** to answer something outside the syllabus, rather than answering from an unrelated book.
7. Toggle **Developer Mode** to show intent detection, LLM call count and retrieval distance.
8. Exam Intelligence → pick **DBMS or TOC** (not "All") → Most Repeated Questions → Find → download the Solutions PDF.

**Avoid live:** the solutions bundle for BDA, CC, CN or CA (model-generated, only skimmed), time-complexity questions (small-model factual risk), "trace/dry-run" requests
(returns code, not a trace), any question about a topic you haven't tested, and the Question Paper tab
on a full paper (slow on CPU) unless you have time to wait.

**If something goes wrong:** say what it is. "That is the small-model limitation I documented; the
counts and years below it are computed by code and are unaffected."

---

## Most Repeated Questions: solutions bundle

Pick a subject in Exam Intelligence → Most Repeated Questions → Find. The questions that appeared in two or more
exam sittings are listed (found by code), and one PDF below the list bundles a model answer for each.

- **Why per subject?** A bundle across subjects would merge different subjects' questions into one card, so the
  tab asks for a subject first.
- **Where do the answers come from?** 125 repeated questions across 16 subjects (ML and CNS have none). 67 are
  curated answers; 58 were generated offline by the same local model from the notes, in one overnight batch.
- **Were the generated answers checked?** I reviewed them. Calculation and proof questions came out wrong (for
  example the wrong largest number, an invented definition of a recursive language), so I replaced 41 of the first
  99 with curated answers. The 58 that remain are mostly descriptive theory and were only skimmed.
- **Why are answers precomputed?** On this CPU one answer takes minutes; precomputing makes the download instant.
- **Safest subjects to demo:** DBMS, OS, TOC, DM.

---

## Questions you should be ready for

- *What did you build yourself vs use off the shelf?* — Ollama, FAISS, Streamlit and LangChain are
  libraries. Mine: the document loader with OCR and year detection, question extraction, the question
  bank and its 4-level matcher, deterministic exam analytics, query understanding with code-level
  corrections, the guard layer, the UI, and the test harness.
- *Why not just use ChatGPT?* — It can't see this university's notes or papers, can't give real
  frequencies, and student data would leave the machine.
- *Is the exam-frequency number correct?* — It's exact for what is in the indexed papers and
  literally matches the topic wording; it can undercount paraphrases. It is never estimated.
- *Can another university use it?* — Yes: it is a per-university install. Upload that university's
  notes, papers and syllabus, set the name in Setup, and rebuild. It is single-tenant by design, not
  a shared cloud service.
- *What would you improve first?* — Calibrating the matching cutoffs on a labelled set, and a
  sentence-level verifier for explanations.