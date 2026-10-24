# config.py

from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "bangalore_university_mca"
NOTES_DIR = DATA_DIR / "notes"
PAPERS_DIR = DATA_DIR / "question_papers"
SYLLABUS_DIR = DATA_DIR / "syllabus"
FAISS_DIR = BASE_DIR / "faiss_store"

# Ollama models
LLM_MODEL = "mistral:7b"
EMBEDDING_MODEL = "nomic-embed-text"

# --- LLM speed tuning (CPU-only inference) ---
# num_ctx: how much context window Ollama allocates. Our prompts (context +
# question + guidance) are well under 1500 tokens even at the largest marks
# level, so 2048 gives headroom without over-allocating.
LLM_NUM_CTX = 3072
# num_thread: leave as None to let Ollama auto-detect your CPU core count
# (usually correct) — only set this if you've confirmed a specific value
# helps, since guessing wrong can make things slower, not faster.
LLM_NUM_THREAD = None
# Hard cap on how many tokens the model will generate, per marks level.
# This bounds worst-case latency directly — generation time scales with
# token count on CPU, so a runaway 10-mark answer can't take forever.
MARKS_MAX_TOKENS = {2: 250, 5: 600, 8: 850, 10: 1100}
# Code answers (lab programs / "write an algorithm") need more room than prose.
LAB_MAX_TOKENS = 900
# If the model hits the token cap mid-answer, ask it to finish (at most this many extra calls, each
# limited to CONTINUATION_TOKENS). If it is still cut off, the answer is trimmed to its last complete
# sentence so it never ends in the middle of a word, sentence or code block.
MAX_CONTINUATIONS = 1
CONTINUATION_TOKENS = 350

# Path to the Tesseract-OCR engine executable. Leave as None if `tesseract`
# is already on your system PATH. On Windows, if you installed it via the
# UB-Mannheim installer and it's not on PATH, set the actual install path
# here, e.g.:
#   TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# RAG settings
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
TOP_K = 3

# University info — overridable via university_settings.json (written by
# the in-app Setup screen), so a new university can be onboarded without
# editing source code. Falls back to these defaults if that file doesn't
# exist yet or can't be read.
UNIVERSITY_NAME = "Bangalore University"
DEPARTMENT = "MCA"

_SETTINGS_FILE = BASE_DIR / "university_settings.json"
if _SETTINGS_FILE.exists():
    try:
        import json as _json
        with open(_SETTINGS_FILE, "r", encoding="utf-8") as _f:
            _settings = _json.load(_f)
        UNIVERSITY_NAME = _settings.get("university_name", UNIVERSITY_NAME)
        DEPARTMENT = _settings.get("department", DEPARTMENT)
    except Exception:
        pass  # a corrupt settings file should never crash startup — fall back to defaults

# Shared admin password gating document upload/reindexing — NOT per-student
# accounts. Change this before handing the app to anyone else, or set the
# ADMIN_PASSWORD environment variable instead of editing this file.
import os as _os
ADMIN_PASSWORD = _os.environ.get("ADMIN_PASSWORD", "")
if not ADMIN_PASSWORD or ADMIN_PASSWORD == "bu":
    # SAFETY_FIX: never fall back to a guessable password
    import secrets as _secrets
    ADMIN_PASSWORD = _secrets.token_urlsafe(9)
    print(f"[uniRAG] ADMIN_PASSWORD is not set. Admin password for THIS run only: {ADMIN_PASSWORD}")
    print('[uniRAG] To choose your own, set it before starting:  $env:ADMIN_PASSWORD="your-password"')

# HARDENING_PATCH_APPLIED
# True  = the admin "Rebuild everything" button backs up and then RESTORES the existing
#         question bank (so verified/corrected questions are never lost).
# False = old behaviour (re-extract the question bank from the PDFs).
PRESERVE_QUESTION_BANK_ON_REBUILD = True

# Model used ONLY for query parsing (short JSON output). Smaller = faster on CPU.
# Leave as LLM_MODEL for no change. To try the faster parser, run `ollama pull llama3.2:3b`
# and change the next line to: PARSE_MODEL = "llama3.2:3b"  (then run eval_questions.py).
PARSE_MODEL = LLM_MODEL

# Keep models loaded between questions (Ollama unloads after 5 minutes by default,
# and reloading a 7B model costs 10-30 s).
LLM_KEEP_ALIVE = "30m"

# ANSWER_CACHE_PATCH: True = reuse saved answers when the model's input is identical (fast).
# Set to False when running eval_questions.py so every answer is generated fresh.
ANSWER_CACHE = True