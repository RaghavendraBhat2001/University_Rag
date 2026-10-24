# uniRAG — Extra viva answers (add next to VIVA_NOTES.md)

Short, honest answers for the questions VIVA_NOTES.md does not cover yet.
Fill the numbers marked **[check]** from your last startup log / eval run just before the viva.

---

## 1. Corrections already applied to VIVA_NOTES.md (keep these wordings)

- "The semantic-match cutoff was set by inspection" → say:
  **"I measured retrieval distance on 8 topics from my notes: 0.29–0.52. Off-topic questions are mostly stopped earlier by the syllabus gate; two that reached retrieval were at 0.66–0.69. The ceiling is 1.15, deliberately permissive, because precision is enforced by the syllabus gate and the grounding check, not by distance alone."**
- "~28 fixed questions" → **29 cases / 44 checks, all passing [check]**.
- Numbers section → use the final question count from the OCR work **[check]**.

---

## 2. Evaluation and honesty

**How do you know it works?**
A regression suite (`eval_questions.py`): 29 cases, 44 checks, run through the same path the chat uses. It checks
intent classification, out-of-scope refusal, that counts equal the years listed, that code is C, and that identical
questions give identical answers. It also found real bugs (a false-pass rule, a false warning) that I fixed.

**What is your accuracy?**
I did not measure answer accuracy with a labelled set — that is future work. What I measured is behaviour that code
can guarantee (counts, years, refusals, determinism). Explanation quality is a 7B model's, so I show sources and call
the explanations study aids.

**How was the question bank checked?**
Subjects were verified word-for-word against the scanned papers **[check how many subjects are done]**; OCR errors in
the rest are a known limitation. Counting uses literal matches only, so an OCR typo can cause an undercount, never an
invented count.

---

## 3. Technical "why" questions

**What is an embedding / why FAISS?**
An embedding turns text into a vector so similar meanings are close. FAISS finds the nearest vectors quickly on a
laptop, with no server. Distance here is L2: smaller = more similar.

**Why chunk size 400, overlap 50, top-3?**
400 characters keeps one idea per chunk and fits the 2048-token prompt on CPU; overlap avoids cutting a definition in
half; top-3 keeps the prompt short. These were chosen for speed on this hardware, not tuned exhaustively.

**Why is it slow?**
CPU only, 7B model. Generation time grows with the number of tokens, so long answers take minutes. I cap tokens per
marks level, keep the model loaded (`keep_alive`), and stream the answer so the student sees text appear.

**Why did you not use a smaller model?**
I tried llama3.2:3b for query parsing: faster (14 s vs 33 s) but worse parses (lost follow-up topics, Hinglish, exam
intents; 39 pass / 2 fail), so I reverted to Mistral 7B.

**Why do counts come from code, not the model?**
Models invent numbers. Counting in code makes frequency, years and marks reproducible and checkable.

**What does "literal matches only" mean?**
Only exact/near-exact and subtopic matches are counted. Semantically similar questions are shown as related but not
counted, so the number is conservative.

**Why does the header say 22 subjects but Exam Intelligence says 18?**
22 is the number of subject indexes. 4 of them are lab subjects (programs from lab manuals, no written exam papers), so
they have no exam questions. Exam Intelligence counts the 18 theory subjects that have questions in the bank.

---

## 4. Security and privacy

**How are passwords stored?**
Salted PBKDF2-HMAC-SHA256 (100,000 rounds), compared in constant time. Five wrong tries lock that account for
60 seconds. Admin password comes from the `ADMIN_PASSWORD` environment variable.

**What are the limits?**
No email verification or password reset, and the lockout is per running app. That is fine for an offline single
machine; a networked deployment would need a proper account system.

**Can an admin upload break the folder structure?**
File and folder names are cleaned (path separators and `..` removed) before saving.

**What happens when someone switches user?**
*Switch role* and every new login clear the previous user's session (chat, last topic, cached answers). Login
lockout counters are kept separately, so switching role cannot reset them.

**Can a rebuild destroy corrected data?**
No: "Rebuild everything" backs up the question bank to `backups/` and restores it afterwards (switch in `config.py`).

---

## 5. Failure questions — say these calmly

- **Wrong fact in an explanation** → "Known small-model limit. Counts, years and marks below it are computed by code."
- **Slow answer** → "CPU-only 7B model; this is why the answer streams."
- **Topic not found** → "That is the guard working: it refuses instead of guessing."
- **OCR noise in an old question** → "Scanned papers; the verified subjects were checked word-for-word, the rest are
  listed as a limitation."

---

## 6. Safe demo order (≈8 minutes)

0. Before the viva: start Ollama, run `streamlit run app.py`, ask one throwaway question so the model is loaded.
1. `what is decision tree? how many times has it appeared? should I read it?`
2. `queue kitni baar aaya hai exam me`
3. `explain hashing` → `what about its types` → `how many times has that appeared`
4. `array vs linked list`
5. `write a program for binary search`
6. `explain the French revolution` (shows the refusal)
7. Turn on Developer Mode and point at: intent, LLM calls, retrieval distance.
8. Open Exam Intelligence → pick DS → show the ranking and the marks breakdown. Then Most Repeated Questions → Find →
   download the Solutions PDF (use DBMS, OS, TOC or DM; they are the most carefully checked). Pick a subject first:
   the tabs do not work in "All / Auto-detect" mode.

Avoid: the solutions bundle for BDA, CC, CN or CA, time-complexity questions, "trace/dry-run", untested topics, a full paper in the Question Paper tab.
Do NOT press "Rebuild everything now" during the demo.

---

## 7. Limitations (say them first)

Literal counting can miss paraphrases · 58 of 125 bundle answers are model-generated and only skimmed · some OCR noise · single-university install · no diagrams · long answers are
slow on CPU · explanations can contain small factual errors · open registration without email verification.

**Future work:** sentence-level verifier against retrieved text · labelled evaluation set for answer accuracy ·
calibrated matching cutoffs on a larger topic list · GPU support · diagram extraction · proper accounts if networked.

---

## 8. Solutions bundle (Most Repeated Questions)

**What is it?**
For a chosen subject, the questions that were asked in two or more exam sittings (found by code, never by the model),
each with a model answer, downloadable as one PDF. 125 repeated questions across 16 subjects; ML and CNS have none
because no question repeats across their sittings.

**How were the answers produced?**
67 are curated answers: drafted with an AI assistant, then checked by me; every numerical result was recomputed in code
(probabilities, base conversions, Huffman codes, shortest paths). The other 58 were generated offline by the app's own
model from my notes in one overnight batch. **[edit this line so it states exactly what you did]**

**Why not generate them all?**
I reviewed the generated answers. The 7B model gives good descriptive explanations but fails on calculations and
proofs: for example it gave 30 instead of 90 as the largest number, and defined a recursive language as one whose machine
never halts on non-members. 41 of the first 99 generated answers were wrong or did not solve the question, so I replaced
them with curated ones.

**Is the bundle fully verified?**
No. The 58 generated answers that remain are mostly descriptive theory and were only skimmed, so I call the bundle a
study aid. Questions that refer to a figure missing from the extracted text get the method plus a worked example.

**Why precompute?**
One answer takes minutes on this CPU. Precomputing makes the download instant and lets me review the answers first.

**Why does the tab ask for a subject?**
Counts and solutions are per subject; in "All" mode a card would merge questions from different subjects.