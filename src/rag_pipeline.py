# src/rag_pipeline.py
"""
University Academic & Examination Assistant RAG Pipeline.

Orchestrates:
1. LLM Query Understanding (QueryNormalization, Intent Detection, Subtopics, Follow-up resolution)
2. Intent-Aware Multi-Source Retrieval:
   - Notes RAG (per-subject FAISS indexes)
   - Question Bank & Exam Paper Matching (Exact, Semantic Vector, Subtopic)
   - Syllabus Relevance & Module Verification
   - Lab Code Retrieval
3. Evidence Aggregation before Response Generation (preventing LLM hallucination of exam facts)
4. Fact-Grounded Response Synthesis (Explanations, Exam History, Syllabus Grounding, Study Guidance)
5. Streaming and Non-Streaming Interfaces with Transparency Debug Info
"""

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from config import (
    FAISS_DIR, LLM_MODEL, ALLOW_CROSS_SUBJECT_FALLBACK, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K,
    LLM_NUM_CTX, LLM_NUM_THREAD, MARKS_MAX_TOKENS, NOTES_DIR, PAPERS_DIR, SYLLABUS_DIR,
    PARSE_MODEL, LLM_KEEP_ALIVE, LAB_MAX_TOKENS, MAX_CONTINUATIONS, CONTINUATION_TOKENS, EMBEDDING_MODEL
)
from src.document_loader import load_documents, build_subject_full_names
from src.question_index import (
    QuestionRecord, build_question_bank, save_question_bank, load_question_bank,
    build_question_vectorstore, load_question_vectorstore, search_question_bank
)
from src.query_understanding import QueryUnderstanding, StructuredQuery
from src.question_matcher import QuestionMatcher, MatchResultSet
from src.exam_intelligence import ExamIntelligence, refers_to_missing_diagram  # CURRICULUM_PATCH_APPLIED

import time
import os
import re
import json
import difflib
import shutil
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any

# Length guidance injected into the prompt per marks level. Word targets (not vague adjectives) keep
# answers exam-sized; the token caps in config.py are only a safety net.
MARKS_GUIDANCE = {
    2: "About 70 words in total: 2-3 short sentences for the explanation and ONE short sentence for the analogy. This is a 2-3 mark answer.",
    5: "About 150 words in total: 4-6 sentences (or a short list) for the explanation and at most 2 sentences for the analogy. This is a 4-6 mark answer.",
    8: "About 250 words in total: a clear explanation covering the main points, and at most 3 sentences for the analogy. This is a 7-8 mark answer.",
    10: "About 350 words in total: a complete, well-organised explanation, and at most 3 sentences for the analogy. This is a 10-mark answer.",
}

# [patch_qp_fixes] language for lab-program answers
def _lab_language_rule(question, subject, context):
    q = (question or "").lower()
    s = (subject or "").lower()
    c = context or ""
    lang = None
    if re.search(r"\bjava\b|\brunnable\b|\bextends\s+thread\b|\bthread\s+class\b|\bjvm\b", q):
        lang = "Java"
    elif "c++" in q:
        lang = "C++"
    elif re.search(r"\bpython\b", q):
        lang = "Python"
    elif "java" in s or "object oriented" in s or re.search(r"\boop\b", s):
        lang = "Java"
    elif re.search(r"public\s+static\s+void\s+main|System\.out\.print", c):
        lang = "Java"
    if lang == "Java":
        return ("Language: write the program in Java. Put ALL import statements first, then use ONE public class "
                "(the one containing main); any other classes must be non-public, in the same file. "
                "Never switch to C or Python.")
    if lang:
        return f"Language: write the program in {lang}. Never switch to another language."
    return ("Language: use the same programming language as the code in the context (this is a C lab). "
            "If the context has no code, write C. Never switch to Python or Java.")


# [patch_lab_fence] wrap fence-less lab code in a code block
def _ensure_code_fence(text, lang_rule=""):
    if not text or "```" in text:
        return text
    m = re.search(r"^(?:import |package |public\s+class |class |#include|using |def |from \w+ import)", text, re.M)
    if not m:
        return text
    r = lang_rule or ""
    lang = "java" if "Java" in r else ("cpp" if "C++" in r else ("python" if "Python" in r else "c"))
    # [patch_lab_fence2] close the fence at the last top-level '}' so trailing prose stays outside
    body, tail = text[m.start():], ""
    if lang in ("java", "c", "cpp"):
        e = body.rfind("\n}")
        if e != -1:
            tail = body[e + 2:].strip()
            body = body[:e + 2]
    out = text[:m.start()] + "```" + lang + "\n" + body.rstrip() + "\n```\n"
    return out + ("\n" + tail + "\n" if tail else "")


def _marks_bucket(marks) -> int:
    """Maps any printed marks value onto a guidance bucket (2/5/8/10):
    1-3 -> 2, 4-6 -> 5, 7-8 -> 8, 9+ -> 10. (6-mark questions are about half of all questions and
    are answered in the same space as 5-mark ones, so they belong with the 5-mark bucket.)"""
    try:
        m = int(marks)
    except (TypeError, ValueError):
        return 5
    if m <= 3:
        return 2
    if m <= 6:
        return 5
    if m <= 8:
        return 8
    return 10


# L2 distance threshold for nomic-embed-text FAISS similarity
MAX_RELEVANT_DISTANCE = 1.15

# Out-of-syllabus gate for plain "what is X" questions.
#   "block": if X is not in the syllabus text AND no past-paper question literally
#            mentions it, say so instead of answering from the notes (the notes corpus
#            can contain non-syllabus books, e.g. a general-aptitude guide).
#   "off"  : never block.
# The gate switches itself off when the syllabus has no readable text.
SYLLABUS_GATE = "block"


def _is_substantive(question: str) -> bool:
    """Checks if a student query contains actual text beyond punctuation."""
    stripped = re.sub(r'[^A-Za-z0-9]', '', question)
    return len(stripped) >= 2


# ---- HARDENING_PATCH_APPLIED: protect the question bank during admin rebuilds ----
from pathlib import Path as _SnapPath
import re

# ---- ANSWER_CACHE_PATCH: reuse an earlier answer when the model would get the exact same input ----
import hashlib as _hl, json as _js, os as _os, threading as _th
_ac_lock = _th.Lock()
_ac_mem = None

def _ac_enabled():
    try:
        import config as _c
        return bool(getattr(_c, "ANSWER_CACHE", True))
    except Exception:
        return True

def _ac_path():
    from config import FAISS_DIR
    return _SnapPath(FAISS_DIR) / "_answer_cache.json"

def _ac_key(*parts):
    from config import LLM_MODEL as _m
    raw = "\x1f".join(str(x) for x in (_m, "complete-v2") + parts)   # v2: answers are never cut off mid-sentence
    return _hl.sha256(raw.encode("utf-8")).hexdigest()

def _ac_load():
    global _ac_mem
    if _ac_mem is None:
        try:
            _ac_mem = _js.loads(_ac_path().read_text(encoding="utf-8"))
        except Exception:
            _ac_mem = {}
    return _ac_mem

def _ac_get(key):
    if not _ac_enabled():
        return None
    with _ac_lock:
        return _ac_load().get(key)

def _ac_put(key, text):
    if not _ac_enabled() or not text or not str(text).strip():
        return
    try:
        with _ac_lock:
            d = _ac_load()
            d[key] = text
            while len(d) > 1000:              # bounded; oldest entry goes first
                d.pop(next(iter(d)))
            p = _ac_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = str(p) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                _js.dump(d, f, ensure_ascii=False)
            _os.replace(tmp, p)
    except Exception:
        pass                                   # the cache must never break an answer
# ---- end ANSWER_CACHE_PATCH ----


def _snapshot_question_data(faiss_dir):
    """Copy the question bank + its vector index to <project>/backups/. Returns the folder, or None."""
    faiss_dir = _SnapPath(faiss_dir)
    bank = faiss_dir / "_question_bank.json"
    vec = faiss_dir / "_question_bank_faiss"
    if not bank.exists():
        return None
    dest = faiss_dir.parent / "backups" / ("question_bank_" + time.strftime("%Y%m%d_%H%M%S"))
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(bank, dest / bank.name)
    if vec.is_dir():
        shutil.copytree(vec, dest / vec.name)
    return dest


def _restore_question_data(snapshot, faiss_dir):
    """Put the snapshot back into faiss_dir (bank json + vector index)."""
    snapshot = _SnapPath(snapshot)
    faiss_dir = _SnapPath(faiss_dir)
    faiss_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(snapshot / "_question_bank.json", faiss_dir / "_question_bank.json")
    snap_vec = snapshot / "_question_bank_faiss"
    target_vec = faiss_dir / "_question_bank_faiss"
    if snap_vec.is_dir():
        shutil.rmtree(target_vec, ignore_errors=True)
        shutil.copytree(snap_vec, target_vec)


# AUTO_MARKS_PATCH_APPLIED -- answer depth from the question wording (only when no marks were given)
_DEF_START_RE = re.compile(r"^\s*(what\s+(is|are|was|were)\b|what's\b|define\b|definition\s+of\b|meaning\s+of\b)", re.IGNORECASE)
_DEF_BLOCK_RE = re.compile(
    r"\b(explain|describe|discuss|example|examples|types?|kinds?|difference|differences|differentiate|compare|"
    r"how|why|steps?|algorithm|advantages?|disadvantages?|working|works|applications?|uses?|"
    r"program|code|implement\w*|detail\w*|elaborat\w*)\b", re.IGNORECASE)
_DETAIL_RE = re.compile(r"\b(in\s+detail|detailed|elaborate|elaborately|thoroughly|in\s+depth)\b", re.IGNORECASE)


def _auto_marks(question: str, default: int) -> int:
    q = question or ""
    if _DETAIL_RE.search(q):
        return 8
    if _DEF_START_RE.match(q) and not _DEF_BLOCK_RE.search(q):
        return 2
    return default


class RAGPipeline:
    def __init__(self):
        print("🔧 Initializing RAG Pipeline with Query Understanding & Exam Intelligence...")
        self._ollama_url = "http://127.0.0.1:11434"

        self.embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=self._ollama_url
        )
        self._base_llm_kwargs = {
            "model": LLM_MODEL,
            "temperature": 0.0,
            "base_url": self._ollama_url,
            "num_ctx": LLM_NUM_CTX,
            "keep_alive": LLM_KEEP_ALIVE
        }
        if LLM_NUM_THREAD:
            self._base_llm_kwargs["num_thread"] = LLM_NUM_THREAD
        self.llm = ChatOllama(**self._base_llm_kwargs)

        # Pre-built capped LLM instances
        self._llm_pool = {
            cap: ChatOllama(**{**self._base_llm_kwargs, "num_predict": cap})
            for cap in {*MARKS_MAX_TOKENS.values(), LAB_MAX_TOKENS, CONTINUATION_TOKENS}
        }

        # Initialize Query Understanding layer
        self.query_understanding = QueryUnderstanding(llm_callable=self._llm_parse_callable)

        # Initialize Exam Intelligence module
        self.exam_intelligence = ExamIntelligence(syllabus_dir=SYLLABUS_DIR)

        # Load or build all FAISS indexes and question bank
        (
            self.subject_indexes,
            self.all_notes_index,
            self._question_paper_docs,
            self._question_bank,
            self._question_vectorstore,
        ) = self._load_or_build_indexes()

        # Initialize Question Matcher with hybrid matching
        self.question_matcher = QuestionMatcher(
            question_bank=self._question_bank,
            question_vectorstore=self._question_vectorstore,
        )

        # Word -> set of indices into self._question_paper_docs for fast keyword pre-filtering
        self._paper_word_index = defaultdict(set)
        for i, doc in enumerate(self._question_paper_docs):
            for w in set(re.findall(r'\w+', doc.page_content.lower())):
                self._paper_word_index[w].add(i)

        # Abbreviation -> full descriptive name (e.g. 'ML' -> 'Machine Learning')
        self._subject_full_names = build_subject_full_names(NOTES_DIR, PAPERS_DIR)

        print(
            f"✅ RAG Pipeline ready! ({len(self.subject_indexes)} subject indexes, "
            f"{len(self._question_paper_docs)} paper chunks, {len(self._question_bank)} exam questions indexed)"
        )

    def _llm_with_cap(self, num_predict_cap: int):
        pooled = self._llm_pool.get(num_predict_cap)
        if pooled is not None:
            return pooled
        return ChatOllama(**{**self._base_llm_kwargs, "num_predict": num_predict_cap})

    # ------------------------------------------------------------
    # Never return an answer that stops in the middle of a sentence / code block
    # ------------------------------------------------------------
    @staticmethod
    def _unclosed_fence(text: str) -> bool:
        return text.count("```") % 2 == 1

    @classmethod
    def _was_cut(cls, text: str, done_reason=None) -> bool:
        """True when generation stopped because it hit the token cap (Ollama reports
        done_reason == "length"), or when the text visibly ends inside a code block."""
        return done_reason == "length" or cls._unclosed_fence(text or "")

    @staticmethod
    def _reason_of(msg):
        md = getattr(msg, "response_metadata", None) or {}
        return md.get("done_reason")

    @classmethod
    def _trim_to_complete(cls, text: str) -> str:
        """Last resort: cut back to the last finished sentence / line and close any open code block."""
        text = (text or "").rstrip()
        if cls._unclosed_fence(text):
            last = text.rsplit("\n", 1)[-1].rstrip()
            if last and last[-1] not in ";{}):,>":          # a half-written line: drop it
                cut = text.rfind("\n")
                text = text[:cut].rstrip() if cut > 0 else text
            if cls._unclosed_fence(text):
                text += "\n```"
            return text
        if text and text[-1] in ".!?)]}`*\"'":
            return text
        m = None
        for m in re.finditer(r"[.!?](?=\s|$)|\n", text):
            pass
        if m:                                   # always better to end on a finished sentence than mid-sentence
            return text[:m.end()].rstrip()
        return text

    def _complete(self, text: str, cut: bool, question: str, max_rounds=None):
        """If `text` was cut off by the token cap, let the model finish it (short prompt, so it fits the
        context window), then trim whatever is still unfinished. Returns (text, extra_llm_calls)."""
        extra = 0
        text = (text or "").rstrip()
        rounds = 0
        while cut and rounds < (MAX_CONTINUATIONS if max_rounds is None else max_rounds):
            rounds += 1
            if self._unclosed_fence(text):                      # drop the half-written code line
                k = text.rfind("\n")
                text = text[:k].rstrip() if k > 0 else text
                joiner = "\n"
            else:                                               # drop the half-written word
                text = re.sub(r"\S+$", "", text).rstrip() if text and not text[-1].isspace() and text[-1] not in ".!?:" else text
                joiner = " "
            tail = text[-1200:]
            prompt_text = (
                "You were writing an answer to this exam question for a student:\n"
                f"{question}\n\n"
                "Your answer was cut off. This is how it ends so far:\n---\n"
                f"{tail}\n---\n"
                "Continue EXACTLY from where it stops and finish the answer: complete the current list, sentence "
                "or code block, then stop. Do NOT repeat anything already written, do NOT start with a new "
                "introduction or heading, and keep it brief."
            )
            try:
                res = self._llm_with_cap(CONTINUATION_TOKENS).invoke(prompt_text)
            except Exception as e:
                print(f"⚠️ continuation failed: {e}")
                break
            extra += 1
            more = (res.content or "").strip("\r")
            if not more.strip():
                break
            if joiner == "\n":                                   # code: keep the model's indentation
                text = text + "\n" + more.lstrip("\n")
            else:
                text = text + (" " if not more.startswith("\n") else "") + more.lstrip(" ")
            cut = self._was_cut(text, self._reason_of(res))
        if cut or self._unclosed_fence(text):
            text = self._trim_to_complete(text)
        return text, extra

    def _llm_parse_callable(self, prompt_text: str) -> str:
        """Invokes LLM for Query Understanding with a tight token cap."""
        # temperature 0: the SAME question must always be parsed into the SAME
        # intents/topic. At the shared 0.3 setting, two identical questions could
        # be parsed differently and produce two different answers.
        if getattr(self, "_parse_llm", None) is None:
            self._parse_llm = ChatOllama(**{**self._base_llm_kwargs, "model": PARSE_MODEL, "temperature": 0, "num_predict": 350})
        # PARSE_CACHE_PATCH_APPLIED: same prompt -> same parse -> same retrieval -> same answer-cache key
        _pk = _ac_key("parse", PARSE_MODEL, prompt_text)
        _hit = _ac_get(_pk)
        if _hit is not None:
            return _hit
        res = self._parse_llm.invoke(prompt_text)
        if "{" in res.content and "}" in res.content:
            _ac_put(_pk, res.content)
        return res.content

    def get_available_subjects(self) -> List[str]:
        """Returns sorted list of indexed subjects for UI dropdown."""
        return sorted(s for s in self.subject_indexes.keys() if s != "UNKNOWN")

    def rebuild_everything(self) -> Dict[str, int]:
        """Full re-ingest + re-embed of every subject, syllabus, and question paper."""
        print("🔄 Rebuilding everything (triggered from Setup screen)...")
        import config as _cfg
        _keep_bank = getattr(_cfg, "PRESERVE_QUESTION_BANK_ON_REBUILD", True)
        _snapshot = _snapshot_question_data(FAISS_DIR) if _keep_bank else None
        if _snapshot is not None:
            print(f"🛡️ Question bank backed up to {_snapshot} (it will be restored after the rebuild).")
        if FAISS_DIR.exists():
            for entry in FAISS_DIR.iterdir():
                if entry.is_dir():
                    shutil.rmtree(entry, ignore_errors=True)
                elif entry.name in ("_papers.json", "_manifest.json", "_question_bank.json"):
                    entry.unlink(missing_ok=True)

        (
            self.subject_indexes,
            self.all_notes_index,
            self._question_paper_docs,
            self._question_bank,
            self._question_vectorstore,
        ) = self._build_all_indexes()

        if _snapshot is not None:
            try:
                # Keep the verified/corrected bank, but ADD questions from papers that were not in it
                # (newly uploaded PDFs). Restoring wholesale used to drop them silently.
                fresh_bank = list(self._question_bank)
                _restore_question_data(_snapshot, FAISS_DIR)
                kept = load_question_bank(FAISS_DIR / "_question_bank.json")
                known_sources = {r.source for r in kept}
                added = [r for r in fresh_bank if r.source not in known_sources]
                if added:
                    kept = kept + added
                    save_question_bank(kept, FAISS_DIR / "_question_bank.json")
                    self._question_vectorstore = build_question_vectorstore(
                        kept, self.embeddings, store_path=FAISS_DIR / "_question_bank_faiss")
                else:
                    self._question_vectorstore = load_question_vectorstore(
                        FAISS_DIR / "_question_bank_faiss", self.embeddings)
                self._question_bank = kept
                print(f"🛡️ Kept {len(kept) - len(added)} protected question(s) and added {len(added)} from new papers.")
            except Exception as e:
                print(f"⚠️ Could not restore the protected question bank ({e}). Backup is kept at {_snapshot}.")

        self.question_matcher = QuestionMatcher(
            question_bank=self._question_bank,
            question_vectorstore=self._question_vectorstore,
        )
        self._subject_full_names = build_subject_full_names(NOTES_DIR, PAPERS_DIR)

        self._paper_word_index = defaultdict(set)
        for i, doc in enumerate(self._question_paper_docs):
            for w in set(re.findall(r'\w+', doc.page_content.lower())):
                self._paper_word_index[w].add(i)

        print("✅ Rebuild complete.")
        return {
            "subjects": len(self.subject_indexes),
            "paper_chunks": len(self._question_paper_docs),
            "questions_indexed": len(self._question_bank),
        }

    # ------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------
    _QUERY_LOG_PATH = FAISS_DIR.parent / "logs" / "queries.jsonl"

    def _log_query(self, question: str, subject: Optional[str], structured_query: StructuredQuery, result: Dict[str, Any]):
        try:
            debug_info = result.get("debug_info") or {}
            entry = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "question": question,
                "subject": subject,
                "canonical_topic": structured_query.canonical_topic,
                "subtopics": structured_query.subtopics,
                "intents": structured_query.intents,
                "search_queries": structured_query.search_queries,
                "llm_calls": debug_info.get("llm_calls", 0),
            }
            self._QUERY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(self._QUERY_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"⚠️ Query logging failed: {e}")

    # ------------------------------------------------------------
    # Deterministic Paper Scanning & Trend Analysis
    # ------------------------------------------------------------
    def _correct_topic_spelling(self, topic: str) -> Tuple[str, bool]:
        words = topic.split()
        if not words:
            return topic, False
        corrected = []
        changed = False
        for w in words:
            w_lower = w.lower()
            if w_lower in self._paper_word_index:
                corrected.append(w)
                continue
            matches = difflib.get_close_matches(w_lower, self._paper_word_index.keys(), n=1, cutoff=0.82)
            if matches:
                corrected.append(matches[0])
                changed = True
            else:
                corrected.append(w)
        return " ".join(corrected), changed

    def _paper_pool_for_subject(self, subject: Optional[str]) -> Tuple[List[Document], bool]:
        pool = self._question_paper_docs
        subject_scoped = False
        if subject and subject != "All / Auto-detect":
            subject_upper = subject.strip().upper()
            scoped_pool = [
                d for d in pool
                if str(d.metadata.get("subject", "")).strip().upper() == subject_upper
            ]
            if scoped_pool:
                pool = scoped_pool
                subject_scoped = True
        return pool, subject_scoped

    def _questions_per_year(self, matches):
        """Year -> number of matching QUESTIONS (literal matches only; "related" semantic matches
        are never counted). Years with no match are included as 0 for every sitting on record for
        the same subject(s), so "asked every year or only sometimes?" shows its gaps."""
        literal = [m for m in matches if m.match_type != "Semantic Match"]
        counts: Dict[str, int] = {}
        undated = 0
        for m in literal:
            if m.record.year:
                counts[m.record.year] = counts.get(m.record.year, 0) + 1
            else:
                undated += 1
        if counts:
            subjects = {m.record.subject for m in literal}
            for r in self._question_bank:
                if r.year and r.subject in subjects:
                    counts.setdefault(r.year, 0)
        return dict(sorted(counts.items())), undated

    def analyze_topic_trend_by_questions(self, topic: str, subject: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Exam Intelligence tab: year -> number of matching questions in the question bank."""
        if not topic or len(topic.strip()) < 3:
            return None

        corrected_topic, spelling_corrected = self._correct_topic_spelling(topic.strip())
        pool, subject_scoped = self._paper_pool_for_subject(subject)

        sq = StructuredQuery(
            raw_query=corrected_topic, canonical_topic=corrected_topic,
            search_queries=[corrected_topic], subject=subject,
        )
        match_res = self.question_matcher.match(sq, allow_cross_subject_fallback=False)  # TREND_FIX
        counts, undated = self._questions_per_year(match_res.matches)

        base = {
            "topic": corrected_topic,
            "original_topic": topic,
            "spelling_corrected": spelling_corrected,
            "undated_count": undated,
            "subject_scoped": subject_scoped,
            "candidates_checked": len(pool),
        }
        if not counts:
            return {**base, "trend": "insufficient data", "year_counts": {}}
        return {**base, "trend": self._classify_trend(list(counts.keys()), counts), "year_counts": counts}

    def analyze_frequency(self, topic: str, subject: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Counts how many indexed question papers and questions contain this topic."""
        if not topic or len(topic) < 3:
            return None

        # Hybrid question matching from Question Bank
        sq = StructuredQuery(
            raw_query=topic,
            canonical_topic=topic,
            subtopics=[],
            intents=["frequency"],
            search_queries=[topic],
            subject=subject,
        )
        match_result = self.question_matcher.match(sq)

        corrected_topic, spelling_corrected = self._correct_topic_spelling(topic)
        pool, subject_scoped = self._paper_pool_for_subject(subject)

        # Direct chunk scan
        topic_lower = corrected_topic.lower()
        sources_seen = {}
        for doc in pool:
            if topic_lower in doc.page_content.lower():
                src = doc.metadata.get("source", "Unknown")
                if src not in sources_seen:
                    sources_seen[src] = doc.metadata.get("year")

        # Blend chunk scan and question bank scan for robust reporting
        all_papers = set(sources_seen.keys()) | {m.record.source for m in match_result.matches}
        all_years = sorted(list({y for y in sources_seen.values() if y} | set(match_result.distinct_years)))

        return {
            "topic": corrected_topic,
            "original_topic": topic,
            "spelling_corrected": spelling_corrected,
            "appearance_count": len(all_papers),
            "papers": sorted(list(all_papers)),
            "years": all_years,
            "candidates_checked": len(pool),
            "subject_scoped": subject_scoped,
            "total_papers_indexed": len(self._question_paper_docs),
            "question_matches_count": match_result.total_matching_questions,
            "distinct_question_papers": match_result.distinct_papers,
        }

    @staticmethod
    def _classify_trend(sorted_years: List[str], counts: Dict[str, int]) -> str:
        if len(sorted_years) < 2:
            return "insufficient data (only one year with a match)"
        values = [counts[y] for y in sorted_years]
        mid = len(values) // 2
        first_half = values[:mid] if mid > 0 else values[:1]
        second_half = values[mid:] if mid > 0 else values[1:]
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        if second_avg > first_avg + 0.5:
            return "rising"
        elif second_avg < first_avg - 0.5:
            return "falling"
        return "stable"

    def analyze_topic_trend(self, topic: str, subject: Optional[str] = None, match_result: Optional["MatchResultSet"] = None) -> Optional[Dict[str, Any]]:
        """Year-by-year trend for the chat answer. Uses the SAME literal matches as the frequency
        and exam-history sections (never a separate scan), counted as questions per year."""
        if not topic or len(topic) < 3:
            return None

        corrected_topic, spelling_corrected = self._correct_topic_spelling(topic)
        pool, subject_scoped = self._paper_pool_for_subject(subject)

        if match_result is None:
            sq = StructuredQuery(raw_query=topic, canonical_topic=topic, search_queries=[topic], subject=subject)
            match_result = self.question_matcher.match(sq)

        counts, undated = self._questions_per_year(match_result.matches)
        base = {
            "topic": corrected_topic,
            "original_topic": topic,
            "spelling_corrected": spelling_corrected,
            "undated_count": undated,
            "subject_scoped": subject_scoped,
            "candidates_checked": len(pool),
        }
        if not counts:
            return {**base, "trend": "insufficient data", "year_counts": {}}
        return {**base, "trend": self._classify_trend(list(counts.keys()), counts), "year_counts": counts}

    # ------------------------------------------------------------
    # Retrieval & Prompt Helpers
    # ------------------------------------------------------------
    def _expand_subject_query(self, question: str, subject: Optional[str]) -> str:
        if not subject:
            return question
        full_name = self._subject_full_names.get(subject)
        if not full_name:
            return question

        stripped = re.sub(r'[?!.]', ' ', question.lower())
        words = [w for w in re.split(r'\s+', stripped) if w]
        wrapper_words = {
            "what", "is", "are", "define", "explain", "describe", "the",
            "a", "an", "meaning", "of", "tell", "me", "about", "in"
        }
        core = [w for w in words if w not in wrapper_words]
        if len(core) == 1 and core[0] == subject.lower():
            return f"{question} ({full_name})"
        return question

    @staticmethod
    def _safe_similarity(store, query, k, errors):
        """similarity_search_with_score with one retry. Failures are recorded and printed,
        never silent — a swallowed embedding error used to look exactly like
        \"this topic isn't in your notes\"."""
        if store is None:
            errors.append("index not loaded")
            return []
        last = None
        for attempt in range(2):
            try:
                return store.similarity_search_with_score(query, k=k)
            except Exception as e:
                last = e
                if attempt == 0:
                    time.sleep(1.5)
        errors.append(f"{type(last).__name__}: {last}")
        print(f"⚠️ Retrieval failed for {query!r}: {type(last).__name__}: {last}")
        return []

    def _prepare_definition(
        self,
        question: str,
        subject: Optional[str],
        topic: Optional[str] = None,
        k: Optional[int] = None,
        context_chunks: int = 3,
        search_queries: Optional[List[str]] = None,
        check_grounding: bool = True,
        subtopics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Retrieves notes chunks from per-subject or all-notes FAISS indexes."""
        k = k or TOP_K
        definition_topic = topic or question

        queries_to_try = list(search_queries or ([question] if not topic else [f"What is {topic}?", topic, question]))
        # Always search the resolved topic itself too: the parser's search_queries sometimes describe
        # the OTHER clauses of a compound question ("how many times...") instead of the topic.
        if topic and topic.strip().lower() not in [q.strip().lower() for q in queries_to_try]:
            queries_to_try.append(topic.strip())
        retrieval_errors: List[str] = []
        scored_docs = []
        index_used = "_all_notes (merged, unscoped)"

        store = None
        if subject and subject != "All / Auto-detect" and subject in self.subject_indexes:
            store = self.subject_indexes[subject]
            index_used = subject
        else:
            store = self.all_notes_index

        seen_contents = set()
        for q in queries_to_try:
            expanded = self._expand_subject_query(q, subject)
            hits = self._safe_similarity(store, expanded, k, retrieval_errors)
            for doc, score in hits:
                if doc.page_content not in seen_contents:
                    seen_contents.add(doc.page_content)
                    scored_docs.append((doc, score))

        # Sort by distance (lower is better)
        scored_docs.sort(key=lambda item: item[1])

        all_docs = [doc for doc, score in scored_docs if score <= MAX_RELEVANT_DISTANCE]
        best_distance = round(scored_docs[0][1], 3) if scored_docs else None

        # Cross-subject fallback: if a specific subject was scoped (e.g. ML) but yielded poor or 0 matches,
        # check all_notes_index for a significantly better authoritative match (e.g. from DS).
        if ALLOW_CROSS_SUBJECT_FALLBACK and subject and subject != "All / Auto-detect" and self.all_notes_index is not None:
            if not all_docs or (best_distance is not None and best_distance > 0.80):
                fallback_scored = []
                for q in queries_to_try:
                    hits = self._safe_similarity(self.all_notes_index, q, k, retrieval_errors)
                    for doc, score in hits:
                        if doc.page_content not in seen_contents:
                            seen_contents.add(doc.page_content)
                            fallback_scored.append((doc, score))
                fallback_scored.sort(key=lambda item: item[1])
                fb_valid = [doc for doc, score in fallback_scored if score <= MAX_RELEVANT_DISTANCE]
                if fb_valid and (not all_docs or fallback_scored[0][1] < (best_distance or 999.0) - 0.10):
                    all_docs = fb_valid
                    best_distance = round(fallback_scored[0][1], 3)
                    fb_subject = all_docs[0].metadata.get("subject", "all_notes")
                    index_used = f"{fb_subject} (auto-fallback from {subject})"

        if not all_docs:
            retrieval_failed = bool(retrieval_errors) and not scored_docs
            if retrieval_failed:
                msg = ("Retrieval failed — the embedding model (Ollama) didn't respond, so I couldn't search your notes. "
                       "Check that Ollama is running, then ask again. "
                       f"(Details: {retrieval_errors[0][:200]})")
            else:
                msg = "I don't have information on this in your indexed notes"
                if subject and subject != "All / Auto-detect":
                    msg += f" for {subject}"
                msg += " — this might be outside what's been uploaded or phrased differently from your notes."
            return {"error": {
                "answer": msg,
                "sources": [],
                "topic": definition_topic,
                "debug_info": {
                    "intent": "definition",
                    "llm_calls": 0,
                    "method": "retrieval ERROR (embedding call failed)" if retrieval_failed else "retrieval found nothing close enough to be relevant",
                    "retrieval_errors": retrieval_errors[:3],
                    "queries_tried": queries_to_try,
                    "index_used": index_used,
                    "chunks_retrieved": 0,
                    "best_distance": best_distance,
                    "relevance_threshold": MAX_RELEVANT_DISTANCE
                }
            }}

        # Subtopic-aware chunk selection: ensure chunks covering subtopics are prioritized in context_docs
        effective_context_chunks = max(context_chunks, 4) if subtopics else context_chunks
        selected_docs = []
        if subtopics:
            for st in subtopics:
                st_tokens = [w for w in re.findall(r"[a-z]{3,}", st.lower()) if w not in {"the", "and", "for", "with"}]
                for doc in all_docs:
                    doc_lower = doc.page_content.lower()
                    if any(tok in doc_lower for tok in st_tokens):
                        if doc not in selected_docs:
                            selected_docs.append(doc)
                            break

        for doc in all_docs:
            if doc not in selected_docs:
                selected_docs.append(doc)
            if len(selected_docs) >= effective_context_chunks:
                break

        context_docs = selected_docs
        context = "\n\n".join([doc.page_content for doc in context_docs])

        # Deterministic grounding check: a vector distance under the threshold only
        # means "nearby wording", not "this is about the asked topic". If most of the
        # topic's own words never appear in the retrieved text, the model would be
        # answering from its own memory (e.g. "Fenwick tree", absent from the notes) —
        # say so instead.
        if check_grounding and topic and not self._topic_grounded(topic, [d.page_content for d in all_docs]):
            msg = f"I couldn't find '{topic}' in your indexed notes"
            if subject and subject != "All / Auto-detect":
                msg += f" for {subject}"
            msg += " — it may be outside what's been uploaded, or worded differently in your notes."
            return {"error": {
                "answer": msg,
                "sources": [],
                "topic": definition_topic,
                "debug_info": {
                    "intent": "definition",
                    "llm_calls": 0,
                    "method": "topic words not present in retrieved notes (grounding check)",
                    "index_used": index_used,
                    "chunks_retrieved": len(all_docs),
                    "best_distance": best_distance,
                    "relevance_threshold": MAX_RELEVANT_DISTANCE
                }
            }}

        unique_sources = []
        seen = set()
        for doc in all_docs:
            source = doc.metadata.get("source", "Unknown")
            if source not in seen:
                seen.add(source)
                unique_sources.append({"source": source})

        return {
            "context": context,
            "sources": unique_sources,
            "topic": definition_topic,
            "index_used": index_used,
            "chunks_retrieved": len(all_docs),
            "best_distance": best_distance,
        }

    _DIAGRAM_ONLY_RE = re.compile(
        r"^\s*(please\s+)?(draw|sketch|diagram\b|show\s+(me\s+)?(the\s+)?(diagram|figure|picture|image))",
        re.IGNORECASE,
    )

    @classmethod
    def _is_diagram_only_request(cls, question: str) -> bool:
        return bool(cls._DIAGRAM_ONLY_RE.match(question))

    _FIG_REF = r"(?:Fig(?:ure|\.)?|Table|Tab\.)\s*\d+(?:[.\-]\d+)*"

    @staticmethod
    def _format_trend(tr: Optional[Dict[str, Any]], topic: str) -> str:
        """Renders the year-by-year trend block. Used both when trend is the only
        intent and when it is one of several (it used to be dropped silently in the
        multi-intent case, even though the trend had been computed)."""
        if not tr or not tr.get("year_counts"):
            return f"I don't have enough year-tagged exam papers to determine a trend for **{topic}**."
        lines = [
            f"### 📈 EXAM TREND: {tr['topic']}",
            f"- **Overall Trend:** **{tr['trend'].upper()}**",
            f"- **Yearly Appearances:**"
        ]
        for y, count in tr["year_counts"].items():
            lines.append(f"  - **{y}:** {count} question(s)")
        return "\n".join(lines)

    @classmethod
    def _strip_figure_refs(cls, text: str) -> str:
        """Removes references to figures/tables/section marks from LLM-written
        text ("Figure 3.1 provides an example", "as shown in Fig. 2", "§.33") — the
        student can't see them, and the model ignores the prompt rule against them.
        Only ever applied to the model's own prose, never to stored exam questions."""
        if not text:
            return text
        # hallucinated markdown image links (offline product: they can never resolve)
        text = re.sub(r"!\[[^\]]*\]\([^)]*\)?", "", text)
        # "(see Figure 3.1)" / "(Table 2)" parentheticals
        text = re.sub(r"\s*\((?:see |refer to |cf\. )?" + cls._FIG_REF + r"[^)]*\)", "", text, flags=re.IGNORECASE)
        # ", as shown in Figure 3.1" style tails
        text = re.sub(r",?\s*(?:as\s+)?(?:shown|illustrated|depicted|given|seen|described)\s+(?:in|by)\s+" + cls._FIG_REF, "", text, flags=re.IGNORECASE)
        # stray section marks like "§.33" or "§3.3"
        text = re.sub(r"\s*§\s*[\d.]+", "", text)
        # any remaining sentence that still mentions a figure/table number is dropped
        cleaned_lines = []
        for line in text.split("\n"):
            parts = re.split(r"(?<=[.!?])\s+", line)
            kept = [p for p in parts if not re.search(cls._FIG_REF, p, re.IGNORECASE)]
            cleaned_lines.append(" ".join(kept))
        return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned_lines)).strip()

    @staticmethod
    def _topic_grounded(topic: str, texts: List[str]) -> bool:
        """True when a strict majority of the topic's significant words appear
        (5-letter prefix match, so plurals/spelling variants still count) in the
        retrieved text. Purely lexical — no LLM involved."""
        stop = {"the", "and", "for", "with", "from", "into", "using", "what", "how", "are", "its", "vs", "versus"}
        tokens = [w for w in re.findall(r"[a-z]{3,}", topic.lower()) if w not in stop]
        if not tokens:
            return True
        blob = " ".join(texts).lower()
        keys = [w if len(w) < 5 else w[:5] for w in tokens]
        hits = sum(1 for k in keys if k in blob)
        return hits * 2 > len(keys)

    @staticmethod
    def _definition_prompt():
        return ChatPromptTemplate.from_template("""You are a study assistant for MCA university students.
Use ONLY the context below — never invent facts not present in it.

Structure your answer under EXACTLY these two headings, and nothing else. Write each heading exactly as shown. Never write the words "Part 1", "Part 2", "Part A" or any numbering:

📘 As per your notes:
(your precise, textbook-style explanation, grounded strictly in the context)

💡 Simple way to understand it:
(the SAME idea in plain, everyday language using one short analogy; no new facts here)

Guidance — follow silently and NEVER copy, quote or mention these lines in your answer:
- Define the concept and state how it works.
- If the Topic to Explain names specific types, variants or operations, explain each one briefly.
- If it compares two things, explain each, then give the key differences as short bullet points.
- Length: {marks_guidance} Finish every sentence; keep the whole answer within this length.

Rules:
- Do NOT mention or guess how many times this topic has appeared in exams, whether the student should read it, how important it is, or any study advice — that comes from real exam data elsewhere.
- Do NOT write meta-commentary about the context (never say "in the provided context" or "the context does not discuss").
- If the question asks for a diagram/figure, explain it in words; do NOT draw ASCII art or give image links.
- Never mention figure, table, page or section numbers (like "Figure 5.14" or "§.33").
- Do NOT mention time or space complexity (Big-O) unless the Topic to Explain explicitly asks about it. If it does, state only complexities the context itself states, with the exact case label it uses (best / average / worst).
- Never add a closing remark about what the notes do not cover. Answer only what was asked.
- If the context doesn't contain enough to answer, say "I don't have enough information in the indexed notes" instead of guessing.

Context from notes:
{context}

Topic to Explain: {question}

Answer:""")

    @staticmethod
    def _definition_focused_question(sq: "StructuredQuery") -> str:
        """Builds a clean, definition-only question to feed the definition prompt.

        Strips away frequency, marks, year, and importance clauses from compound questions
        so the LLM focuses purely on delivering an authoritative conceptual explanation.
        """
        topic = sq.canonical_topic or sq.raw_query
        subtopics = [st for st in (sq.subtopics or []) if st and st.lower() != topic.lower()]

        if subtopics:
            return f"Explain {topic}, covering: {', '.join(subtopics)}."

        # Only rewrite when OTHER intents (frequency/importance/...) were detected alongside
        # the definition — never from substring keywords in the raw text ("exam" also matches
        # "example"/"examine", which used to replace real questions with a generic one).
        other_intents = [i for i in (sq.intents or []) if i != "definition"]
        if other_intents and sq.canonical_topic:
            return f"What is {topic}? Explain it in detail."

        return sq.raw_query

    @staticmethod
    def _lab_program_prompt():
        return ChatPromptTemplate.from_template("""You are a programming lab assistant for MCA students. The student is asking for actual working CODE for the SPECIFIC program named in the Student Question below.

FIRST, check whether the context actually describes THAT SAME program — not merely one with a similar or overlapping name. Similar-sounding names are almost always DIFFERENT programs:
- "binary search" (searching a sorted array/list) is NOT "binary search tree" (a linked data structure with insert/traversal).
- "linked list insertion" is NOT "linked list traversal" or "linked list deletion".
- "quick sort" is NOT "merge sort", even though both are sorting algorithms.
If the context is actually about a different specific operation, algorithm, or data structure than what was asked — even if it looks related or shares words — do NOT reuse its code, its worked example, or copy its own description/comment. In that case, write standard, correct code for the ACTUAL thing asked, and say so plainly in your one-line comment (e.g. "// Note: lab manual covers binary search trees, not binary search — standard binary search below").

If the context DOES genuinely cover the same program, follow its SAME algorithm, approach, and operation order (e.g. insert-at-head vs insert-at-end/tail, iterative vs recursive, which helper functions it defines) — do not substitute a different generic textbook approach just because it's more common.

{language_rule}
Sample data must satisfy the algorithm's preconditions (for example, binary search needs a SORTED array).
Cover EVERY part of the question that asks for code, and print only what is asked (for example, if asked for prime numbers, print only the primes).
Use the data types the question names (for example, 'day, month and year numbers' means three int values, not month-name text).
Prefer simple textbook logic (a days-in-month array, plain if/else) over clever one-line formulas, and double-check boundary cases (for dates: 31 days in Jan/Mar/May/Jul/Aug/Oct/Dec, 30 in Apr/Jun/Sep/Nov, Feb with leap years).

Output ONLY:
1. A one-line comment naming the language and the program objective — matching what was actually asked, never a substituted topic.
2. The complete, working code in a single code block.

Do not explain the algorithm in prose, with ONE exception: if the Student Question says "algorithm", first give the algorithm as a short numbered list of steps; if it also says "trace" or gives sample input, add a compact step-by-step trace table for that exact input (use the real values from the question). Keep both short, then the code. Do not add conversational commentary. Do not add any remark about how often this has been asked in exams, how common it is, or its importance — that is never something you determine; it comes from real exam-history data elsewhere, never from you.

Context from lab notes:
{context}

Student Question: {question}

Answer:""")

    @staticmethod
    def _safe_dirname(name: str) -> str:
        cleaned = re.sub(r'[^A-Za-z0-9_\-]', '_', name)[:80]
        return cleaned or "UNKNOWN"

    def _embed_in_batches(self, chunks: List[Document]) -> FAISS:
        batch_size = 100
        total = len(chunks)
        store = None
        for i in range(0, total, batch_size):
            batch = chunks[i:i + batch_size]
            try:
                if store is None:
                    store = FAISS.from_documents(batch, self.embeddings)
                else:
                    store.add_documents(batch)
                time.sleep(2)
            except Exception as e:
                print(f"      ⚠️ Error in batch: {e} — retrying smaller...")
                for j in range(0, len(batch), 20):
                    sub_batch = batch[j:j + 20]
                    if store is None:
                        store = FAISS.from_documents(sub_batch, self.embeddings)
                    else:
                        store.add_documents(sub_batch)
                    time.sleep(3)
        return store

    def _load_or_build_indexes(self):
        """Loads per-subject FAISS indexes, question bank, and question vectorstore from disk.
        If per-subject indexes already exist, reuses them instead of re-embedding the entire corpus.
        """
        all_notes_path = FAISS_DIR / "_all_notes"
        papers_path = FAISS_DIR / "_papers.json"
        manifest_path = FAISS_DIR / "_manifest.json"
        question_bank_path = FAISS_DIR / "_question_bank.json"
        qb_faiss_path = FAISS_DIR / "_question_bank_faiss"

        subject_dirs = [
            d for d in FAISS_DIR.iterdir()
            if d.is_dir() and d.name not in ("_all_notes", "_question_bank_faiss")
        ] if FAISS_DIR.exists() else []

        if subject_dirs:
            print(f"📂 Found {len(subject_dirs)} existing subject FAISS folders. Loading...")
            subject_indexes = {}
            for entry in sorted(subject_dirs):
                try:
                    store = FAISS.load_local(str(entry), self.embeddings, allow_dangerous_deserialization=True)
                    sample = next(iter(store.docstore._dict.values()), None)
                    subject = sample.metadata.get("subject", entry.name) if sample else entry.name
                    subject_indexes[subject] = store
                except Exception as e:
                    print(f"   ⚠️ Skipping {entry.name}: {e}")

            # Merged All-Notes Index
            if all_notes_path.exists():
                all_notes_index = FAISS.load_local(str(all_notes_path), self.embeddings, allow_dangerous_deserialization=True)
            else:
                print("💾 Generating merged all-notes index from loaded subject indexes...")
                all_notes_index = None
                for s_name, store in subject_indexes.items():
                    reloaded = FAISS.load_local(
                        str(FAISS_DIR / self._safe_dirname(s_name)),
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    if all_notes_index is None:
                        all_notes_index = reloaded
                    else:
                        all_notes_index.merge_from(reloaded)
                if all_notes_index:
                    all_notes_index.save_local(str(all_notes_path))

            # Question Papers and Question Bank
            paper_docs = []
            if papers_path.exists():
                try:
                    with open(papers_path, "r", encoding="utf-8") as f:
                        raw_papers = json.load(f)
                    paper_docs = [Document(page_content=p["page_content"], metadata=p["metadata"]) for p in raw_papers]
                except Exception:
                    paper_docs = []

            question_bank = []
            if question_bank_path.exists():
                question_bank = load_question_bank(question_bank_path)

            if not paper_docs or not question_bank:
                print("📄 Extracting question papers and building structured question bank...")
                from src.document_loader import load_pdf
                qp_pages = []
                if PAPERS_DIR.exists():
                    for pdf in sorted(PAPERS_DIR.glob("**/*.pdf")):
                        qp_pages.extend(load_pdf(pdf, "question_paper"))

                if qp_pages:
                    question_bank = build_question_bank(qp_pages)
                    save_question_bank(question_bank, question_bank_path)

                    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
                    paper_docs = splitter.split_documents(qp_pages)
                    with open(papers_path, "w", encoding="utf-8") as f:
                        json.dump([{"page_content": c.page_content, "metadata": c.metadata} for c in paper_docs], f)

            # Question Vectorstore
            question_vectorstore = load_question_vectorstore(qb_faiss_path, self.embeddings)

            # Save manifest
            manifest = {
                "version": "2.0",
                "built_at": datetime.now().isoformat(),
                "subjects": {s: {"chunk_count": len(store.docstore._dict)} for s, store in subject_indexes.items()},
                "paper_chunks": len(paper_docs),
                "question_bank_size": len(question_bank),
                "status": "complete",
            }
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

            print(f"📄 Loaded {len(question_bank)} indexed question(s) from question bank")
            return subject_indexes, all_notes_index, paper_docs, question_bank, question_vectorstore

        print("📝 No existing subject indexes found — building fresh from source...")
        return self._build_all_indexes()

    def _build_all_indexes(self):
        """Builds all subject indexes, question bank, and question vector store."""
        print("🔍 Loading documents...")
        docs = load_documents()
        if not docs:
            raise Exception("❌ No PDF documents found in data folder!")

        print(f"📄 Loaded {len(docs)} pages")

        print("📋 Building question bank from question-paper pages (per-question, before chunking)...")
        question_paper_pages = [d for d in docs if d.metadata.get("type") == "question_paper"]
        question_bank = build_question_bank(question_paper_pages)
        print(f"   Extracted {len(question_bank)} individual question(s)")

        print("✂️ Splitting into chunks...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_documents(docs)
        print(f"   Created {len(chunks)} chunks")

        notes_by_subject = {}
        paper_chunks = []
        for c in chunks:
            if c.metadata.get("type") == "question_paper":
                paper_chunks.append(c)
            else:
                subj = c.metadata.get("subject", "UNKNOWN")
                notes_by_subject.setdefault(subj, []).append(c)

        FAISS_DIR.mkdir(parents=True, exist_ok=True)

        print(f"💾 Building {len(notes_by_subject)} per-subject indexes...")
        subject_indexes = {}
        total_subjects = len(notes_by_subject)
        for idx, (subject, subj_chunks) in enumerate(notes_by_subject.items(), 1):
            print(f"   [{idx}/{total_subjects}] {subject}: embedding {len(subj_chunks)} chunks...")
            store = self._embed_in_batches(subj_chunks)
            store.save_local(str(FAISS_DIR / self._safe_dirname(subject)))
            subject_indexes[subject] = store

        print("💾 Building merged 'All subjects' fallback index...")
        all_notes_index = None
        for subject in subject_indexes:
            reloaded = FAISS.load_local(
                str(FAISS_DIR / self._safe_dirname(subject)), self.embeddings,
                allow_dangerous_deserialization=True
            )
            if all_notes_index is None:
                all_notes_index = reloaded
            else:
                all_notes_index.merge_from(reloaded)
        all_notes_index.save_local(str(FAISS_DIR / "_all_notes"))

        print(f"💾 Saving {len(paper_chunks)} question-paper chunks as plain text...")
        with open(FAISS_DIR / "_papers.json", "w", encoding="utf-8") as f:
            json.dump([{"page_content": c.page_content, "metadata": c.metadata} for c in paper_chunks], f)

        print(f"💾 Saving {len(question_bank)} question-bank record(s)...")
        save_question_bank(question_bank, FAISS_DIR / "_question_bank.json")

        print("💾 Embedding Question Bank for semantic question retrieval...")
        qb_vectorstore = build_question_vectorstore(
            question_bank, self.embeddings, store_path=FAISS_DIR / "_question_bank_faiss"
        )

        manifest = {
            "version": "2.0",
            "built_at": datetime.now().isoformat(),
            "subjects": {
                subject: {
                    "folder_name": self._safe_dirname(subject),
                    "chunk_count": len(subj_chunks),
                    "source_pdfs": sorted(set(c.metadata.get("source", "unknown") for c in subj_chunks)),
                }
                for subject, subj_chunks in notes_by_subject.items()
            },
            "paper_chunks": len(paper_chunks),
            "question_bank_size": len(question_bank),
            "question_vectorstore_built": qb_vectorstore is not None,
            "status": "complete",
        }
        with open(FAISS_DIR / "_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print("✅ All indexes and Question Bank successfully built and saved!")
        return subject_indexes, all_notes_index, paper_chunks, question_bank, qb_vectorstore

    # ------------------------------------------------------------
    # Core Answer Synthesis
    # ------------------------------------------------------------
    def _execute_intent_aware_retrieval(self, sq: StructuredQuery, marks: int, subject: Optional[str], skip_syllabus_gate: bool = False) -> Dict[str, Any]:
        """Collects evidence across Notes, Question Bank, Syllabus, and Lab Manuals.

        skip_syllabus_gate=True bypasses the out-of-syllabus check entirely — for the
        Question Paper tab (plain_answer_only), where every question comes from a REAL
        uploaded past paper from this university. By definition it's in-syllabus; running
        the check there is redundant at best, and at worst could wrongly refuse a
        legitimate question on a false-negative syllabus/topic match against short or
        OCR-garbled extracted question text.
        """
        evidence: Dict[str, Any] = {
            "structured_query": sq,
            "topic": sq.canonical_topic,
            "subtopics": sq.subtopics,
            "intents": sq.intents,
            "notes_prep": None,
            "question_matches": None,
            "syllabus_info": None,
            "ranking_results": None,
            "trend_result": None,
            "sources": [],
        }

        # 0. Out-of-syllabus gate (plain definition questions only)
        if not skip_syllabus_gate and SYLLABUS_GATE == "block" and sq.intents == ["definition"] and sq.canonical_topic:
            if self.exam_intelligence.syllabus_scope(sq.canonical_topic) is False:
                # Only a genuinely EXACT past-paper mention keeps a topic in scope —
                # "Subtopic Match" is also a loose, partial-word match (e.g. "Fenwick
                # Tree" subtopic-matching an unrelated "AVL Tree" question on the
                # shared word "tree" alone), not proof the topic itself was examined.
                # Excluding only "Semantic Match" let Fenwick Tree and Kubernetes both
                # slip through the gate via a Subtopic Match — a real regression, caught
                # by eval_questions.py's ungrounded_fenwick/ungrounded_kubernetes cases.
                qb = self.question_matcher.match(sq)
                literal = [m for m in qb.matches if m.match_type == "Exact / Near-Exact"]
                if not literal:
                    topic = sq.canonical_topic
                    evidence["notes_prep"] = {"error": {
                        "answer": f"'{topic}' doesn't appear in your MCA syllabus or in any past question paper, "
                                  f"so I haven't answered it — it looks outside what you're examined on.",
                        "sources": [],
                        "topic": topic,
                        "debug_info": {
                            "intent": "definition",
                            "llm_calls": 0,
                            "method": "out-of-syllabus gate (topic absent from syllabus text and past papers)",
                            "relevance_threshold": MAX_RELEVANT_DISTANCE,
                        },
                    }}
                    return evidence

        # 1. Question Bank Matching
        if sq.needs_question_bank:
            match_res = self.question_matcher.match(sq)
            evidence["question_matches"] = match_res
            for m in match_res.matches:
                if m.record.source:
                    evidence["sources"].append({"source": m.record.source})

        # 2. Ranking Intent
        if "ranking" in sq.intents:
            evidence["ranking_results"] = self.exam_intelligence.get_most_repeated_topics_from_bank(
                self._question_bank, subject=sq.subject or subject
            )

        # 3. Trend Intent
        if "trend" in sq.intents:
            evidence["trend_result"] = self.analyze_topic_trend(
                sq.canonical_topic, subject=sq.subject or subject, match_result=evidence.get("question_matches")
            )

        # 4. Syllabus Check
        if sq.needs_syllabus or "importance" in sq.intents or "syllabus" in sq.intents:
            evidence["syllabus_info"] = self.exam_intelligence.check_syllabus_relevance(sq.canonical_topic)

        # 5. Notes / Lab retrieval
        if sq.needs_notes or sq.needs_lab or "definition" in sq.intents:
            k = 6 if sq.needs_lab else TOP_K
            chunks = 6 if sq.needs_lab else 3
            prep = self._prepare_definition(
                sq.raw_query,
                subject=sq.subject or subject,
                topic=sq.canonical_topic,
                k=k,
                context_chunks=chunks,
                search_queries=sq.search_queries,
                check_grounding=not sq.needs_lab and not skip_syllabus_gate,
                subtopics=sq.subtopics,
            )
            evidence["notes_prep"] = prep
            if "error" not in prep:
                for s in prep.get("sources", []):
                    if s not in evidence["sources"]:
                        evidence["sources"].append(s)

        return evidence

    def _format_grounded_response(self, evidence: Dict[str, Any], marks: int, subject: Optional[str], plain_answer_only: bool = False) -> Tuple[str, int]:
        """Synthesizes structured university assistant response from factual evidence."""
        sq: StructuredQuery = evidence["structured_query"]
        intents = sq.intents
        topic = sq.canonical_topic or "Topic"
        llm_calls = 0

        # Dedicated Single-Intent Handlers (kept only for intents that don't
        # meaningfully compose with other sections — ranking/trend are
        # freestanding analyses, not something you'd merge with a code
        # block or an exam-history section):
        # A. Ranking / Most repeated topics
        if intents == ["ranking"]:
            rankings = evidence["ranking_results"] or []
            if not rankings:
                return f"No repeated question clusters found yet for {subject or 'the indexed papers'}.", 0
            lines = [f"### 🏆 MOST FREQUENTLY ASKED QUESTIONS & REPEATED TOPICS ({subject or 'All Subjects'})\n"]
            for idx, r in enumerate(rankings[:10], 1):
                years_str = ", ".join(r['years'][:3]) if r['years'] else "N/A"
                marks_str = f" [{r['marks']} Marks]" if r.get('marks') else ""
                lines.append(
                    f"{idx}. **{r['representative_question']}**{marks_str}  \n"
                    f"   - **Appearances:** {str(len(r['years'])) + ' exam sitting(s)' if r['years'] else 'exam year not detected'} ({r['total_occurrences']} time(s)) | Years: {years_str}"
                )
            return "\n".join(lines), 0

        # B. Trend only
        if intents == ["trend"]:
            return self._format_trend(evidence.get("trend_result"), topic), 0

        # C. Pure exact_questions request
        if intents == ["exact_questions"]:
            match_res: MatchResultSet = evidence["question_matches"]
            return self.exam_intelligence.format_exam_history_section(match_res), 0

        # D. Pure frequency request
        if intents == ["frequency"]:
            match_res: MatchResultSet = evidence["question_matches"]
            if match_res and match_res.total_matching_questions > 0:
                years_str = ", ".join(match_res.distinct_years) if match_res.distinct_years else "N/A"
                n_years = len(match_res.distinct_years or [])
                if n_years > match_res.distinct_papers:
                    appear_str = f"Found across **{n_years}** examination sitting(s) "
                else:
                    appear_str = f"Found in **{match_res.distinct_papers}** distinct examination paper(s) "
                return (
                    f"### 📊 EXAM FREQUENCY: {topic}\n\n"
                    f"- **Appearances:** {appear_str}"
                    f"(**{match_res.total_matching_questions}** total matching question(s)).\n"
                    f"- **Exam Years:** {years_str}\n\n"
                    f"*(Ask 'show me the exact questions' to see the verbatim questions asked.)*"
                ), 0
            return f"I didn't find any questions matching **'{topic}'** in the indexed question papers.", 0

        # ------------------------------------------------------------
        # Compound Intent (Comprehensive Student-Friendly Response)
        # e.g., Definition + Frequency + Exact Questions + Guidance
        # ------------------------------------------------------------
        sections = []

        # Section 1: Concept Definition & Explanation (from Notes RAG), OR
        # Lab Program Code. lab_program takes priority when present — it
        # used to be a mutually-exclusive dedicated branch (`intents ==
        # ["lab_program"]`, early return) that silently produced NO code at
        # all whenever another intent (e.g. "frequency") was also detected
        # alongside it, since this compound builder never checked for
        # "lab_program" here. Folding it in here means "write a program for
        # X, and how many times has it appeared" can never drop one half.
        if "lab_program" in intents:
            prep = evidence["notes_prep"]
            if prep and "error" in prep:
                sections.append(f"## {topic.upper()}\n\n{prep['error']['answer']}")
            elif prep:
                prompt = self._lab_program_prompt()
                _lang_rule = _lab_language_rule(sq.raw_query, subject, prep["context"])
                _key = _ac_key("lab", str(prompt), prep["context"], sq.raw_query, _lang_rule)
                _code = _ac_get(_key)
                if _code is None:
                    chain = prompt | self._llm_with_cap(LAB_MAX_TOKENS)
                    _msg = chain.invoke({"context": prep["context"], "question": sq.raw_query, "language_rule": _lang_rule})
                    _code, _extra = self._complete(_msg.content, self._was_cut(_msg.content, self._reason_of(_msg)), sq.raw_query)
                    _ac_put(_key, _code)
                    llm_calls += 1 + _extra
                _code = _ensure_code_fence(_code, _lang_rule)
                sections.append(f"## {topic.upper()}\n\n{_code}")
        elif "definition" in intents or "importance" in intents or not intents:
            prep = evidence["notes_prep"]
            if prep and "error" not in prep:
                prompt = self._definition_prompt()
                target_marks = sq.marks or marks
                cap = MARKS_MAX_TOKENS[_marks_bucket(target_marks)]
                _q = self._definition_focused_question(sq)
                _mg = MARKS_GUIDANCE[_marks_bucket(target_marks)]
                _key = _ac_key("def", str(prompt), prep["context"], _q, _mg, cap)
                explanation = _ac_get(_key)
                if explanation is None:
                    chain = prompt | self._llm_with_cap(cap)
                    _msg = chain.invoke({
                        "context": prep["context"],
                        "question": _q,
                        "marks_guidance": _mg,
                    })
                    explanation = _msg.content.strip()
                    explanation, _extra = self._complete(explanation, self._was_cut(explanation, self._reason_of(_msg)), _q,
                                                         max_rounds=(0 if _marks_bucket(target_marks) == 2 else None))
                    explanation = self._strip_figure_refs(explanation)
                    _ac_put(_key, explanation)
                    llm_calls += 1 + _extra
                sections.append(f"## {topic.upper()}\n\n{explanation}")
            elif prep and "error" in prep:
                sections.append(f"## {topic.upper()}\n\n{prep['error']['answer']}")

        # Sections 2-4: Exam History / Trend / Syllabus / Study Guidance.
        # Skipped entirely in plain_answer_only mode (Question Paper tab): the
        # student is answering a SPECIFIC real question already in front of
        # them — "this also appeared in 2022/2023" and study-priority
        # commentary is noise there, not the chat tab's "explore what's
        # worth studying" use case these sections were built for.
        match_res: MatchResultSet = evidence.get("question_matches")
        syllabus_info = evidence.get("syllabus_info")
        if not plain_answer_only:
            # Section 2: Exam History (from Question Bank)
            if match_res is not None and (
                any(i in intents for i in ("frequency", "exact_questions", "years", "importance"))
                or match_res.total_matching_questions > 0
            ):
                exam_sec = self.exam_intelligence.format_exam_history_section(match_res)
                sections.append(exam_sec)

            # Section 2b: Year-by-year trend (only when the question asked for it)
            if "trend" in intents:
                sections.append(self._format_trend(evidence.get("trend_result"), topic))

            # Section 3: Syllabus Relevance
            if syllabus_info and syllabus_info.get("in_syllabus"):
                snippets = syllabus_info.get("matches", [])
                s_lines = ["### 📋 SYLLABUS & NOTES RELEVANCE", "- **Status:** Topic verified present in official MCA syllabus."]
                if snippets and snippets[0].get("snippets"):
                    s_lines.append(f"- **Module Context:** *\"{snippets[0]['snippets'][0]}\"*")
                sections.append("\n".join(s_lines))

            # Section 4: Grounded Study & Revision Guidance
            if "importance" in intents or (match_res and match_res.total_matching_questions > 0 and len(intents) > 1):
                guidance_sec = self.exam_intelligence.generate_study_guidance(match_res, syllabus_info)
                sections.append(guidance_sec)

        full_response = "\n\n---\n\n".join(sections)
        return full_response, llm_calls

    # ------------------------------------------------------------
    # Public Entry Points: ask & ask_stream
    # ------------------------------------------------------------
    def _topicless_reply(self, sq):
        """Asks the student for a topic when a topic-dependent question names none."""
        t = (sq.canonical_topic or "").strip().lower()
        if t and not t.startswith("not specified"):
            return None
        if not set(sq.intents) & {"trend", "frequency", "years", "exact_questions", "importance"}:
            return None
        return {"answer": "Which topic would you like me to check? This question needs a specific topic name.",
                "sources": [], "topic": None,
                "debug_info": {"intent": "topicless", "llm_calls": 1, "method": "rejected: no topic in query"}}

    def _curriculum_reply(self, question: str):
        """'how many subjects in 1st sem' etc. - answered from the folder structure, no model call."""
        try:
            from src.curriculum import curriculum_answer
            from config import DEPARTMENT
            text = curriculum_answer(question, NOTES_DIR, PAPERS_DIR, DEPARTMENT)
        except Exception as e:  # never let this feature break normal answering
            print(f"curriculum lookup skipped: {e}")
            return None
        if not text:
            return None
        return {
            "answer": text,
            "sources": [],
            "topic": None,
            "debug_info": {"intent": "curriculum", "llm_calls": 0,
                           "method": "read from the notes / question-paper folder structure (no model call)"},
        }

    def ask(self, question: str, marks: int = 8, subject: Optional[str] = None, context_topic: Optional[str] = None, plain_answer_only: bool = False) -> Dict[str, Any]:
        """Non-streaming entry point.

        plain_answer_only=True skips exam-history/trend/syllabus/study-guidance
        sections entirely, returning just the concept explanation or code —
        for callers (the Question Paper tab) answering a SPECIFIC real question
        already in hand, where that extra commentary is noise, not insight.
        """
        if not _is_substantive(question):
            return {
                "answer": "I didn't catch an actual question there — could you type what you'd like to ask?",
                "sources": [],
                "topic": None,
                "debug_info": {"intent": "invalid_input", "llm_calls": 0, "method": "non-substantive query"}
            }

        if not plain_answer_only:
            _cur = self._curriculum_reply(question)
            if _cur:
                return _cur

        if not plain_answer_only and self._is_diagram_only_request(question):
            return {
                "answer": "Diagrams aren't available in this version of uniRAG — I can only give text answers. "
                          "Try asking me to *explain* the same topic in words (for example, \"explain AVL tree rotations\").",
                "sources": [],
                "topic": None,
                "debug_info": {"intent": "diagram_request", "llm_calls": 0, "method": "diagram requests are not supported"}
            }

        # 1. Query Understanding
        sq = self.query_understanding.parse(question, context_topic=context_topic, subject_context=subject)
        understanding_llm_calls = 1 if getattr(self.query_understanding, "last_parse_used_llm", True) else 0  # REVIEW_FIXES_APPLIED

        if not plain_answer_only:
            _r = self._topicless_reply(sq)
            if _r:
                return _r

        # Question Paper tab: the student is answering ONE real exam question. Exam-meta
        # intents (trend/frequency/...) are never wanted there — words like "increasing
        # order" or "consistent" inside a question must not turn it into a trend report.
        if plain_answer_only:
            _meta = {"frequency", "trend", "importance", "ranking", "exact_questions", "years", "similar_questions", "syllabus"}
            sq.intents = [i for i in sq.intents if i not in _meta] or ["definition"]
            sq.needs_question_bank = False
            sq.needs_syllabus = False
            if "lab_program" not in sq.intents:
                sq.needs_notes = True

        # 2. Intent-Aware Retrieval & Evidence Gathering
        evidence = self._execute_intent_aware_retrieval(sq, marks, subject, skip_syllabus_gate=plain_answer_only)

        # 3. Grounded Response Synthesis
        full_answer, synthesis_llm_calls = self._format_grounded_response(evidence, marks, subject, plain_answer_only=plain_answer_only)
        if plain_answer_only and full_answer and refers_to_missing_diagram(question):
            full_answer = full_answer.rstrip() + (
                "\n\n> ℹ️ This question refers to a figure/graph that is not part of the paper's text, "
                "so this answer explains the method only and does not use the actual figure.")
        total_llm_calls = understanding_llm_calls + synthesis_llm_calls

        match_res: Optional[MatchResultSet] = evidence.get("question_matches")
        notes_prep = evidence.get("notes_prep") or {}
        result = {
            "answer": full_answer,
            "sources": evidence.get("sources", []),
            "topic": sq.canonical_topic or context_topic,
            "debug_info": {
                "canonical_topic": sq.canonical_topic,
                "subtopics": sq.subtopics,
                "intents": sq.intents,
                "search_queries": sq.search_queries,
                "question_scope": sq.question_scope,
                "is_followup": sq.is_followup,
                "resolved_reference": sq.resolved_reference,
                "llm_calls": total_llm_calls,
                "matching_questions": match_res.total_matching_questions if match_res else 0,
                "distinct_papers": match_res.distinct_papers if match_res else 0,
                "distinct_years": match_res.distinct_years if match_res else [],
                "best_distance": notes_prep.get("best_distance", notes_prep.get("error", {}).get("debug_info", {}).get("best_distance")),
                "relevance_threshold": MAX_RELEVANT_DISTANCE,
                "method": "LLM Query Understanding → Multi-Level Hybrid Retrieval → Evidence Aggregation → Grounded Synthesis",
            }
        }
        self._log_query(question, subject, sq, result)
        return result

    def ask_stream(self, question: str, marks: int = 8, subject: Optional[str] = None, context_topic: Optional[str] = None):
        """Streaming entry point yielding (kind, payload) tuples for progressive rendering."""
        if not _is_substantive(question):
            yield ("done", {
                "answer": "I didn't catch an actual question there — could you type what you'd like to ask?",
                "sources": [],
                "topic": None,
                "debug_info": {"intent": "invalid_input", "llm_calls": 0, "method": "non-substantive query"}
            })
            return

        _cur = self._curriculum_reply(question)
        if _cur:
            yield ("done", _cur)
            return

        if self._is_diagram_only_request(question):
            yield ("done", {
                "answer": "Diagrams aren't available in this version of uniRAG — I can only give text answers. "
                          "Try asking me to *explain* the same topic in words (for example, \"explain AVL tree rotations\").",
                "sources": [],
                "topic": None,
                "debug_info": {"intent": "diagram_request", "llm_calls": 0, "method": "diagram requests are not supported"}
            })
            return

        # 1. Query Understanding
        sq = self.query_understanding.parse(question, context_topic=context_topic, subject_context=subject)
        understanding_llm_calls = 1 if getattr(self.query_understanding, "last_parse_used_llm", True) else 0  # REVIEW_FIXES_APPLIED

        if sq.marks is None:
            marks = _auto_marks(question, marks)  # AUTO_MARKS_PATCH_APPLIED
        _r = self._topicless_reply(sq)
        if _r:
            yield ("done", _r)
            return

        # 2. Intent-Aware Retrieval & Evidence Gathering
        evidence = self._execute_intent_aware_retrieval(sq, marks, subject)

        yield ("meta", {
            "sources": evidence.get("sources", []),
            "topic": sq.canonical_topic or context_topic
        })

        # 3. Stream or assemble response
        # If response has a definition component, stream definition tokens first
        intents = sq.intents
        prep = evidence.get("notes_prep")
        match_res: Optional[MatchResultSet] = evidence.get("question_matches")

        running_text = ""
        synthesis_llm_calls = 0

        if ("lab_program" not in intents) and ("definition" in intents or "importance" in intents or not intents) and prep and "error" not in prep:
            topic = sq.canonical_topic or "Topic"
            prefix = f"## {topic.upper()}\n\n"
            running_text = prefix
            yield ("token", running_text)

            prompt = self._definition_prompt()
            target_marks = sq.marks or marks
            cap = MARKS_MAX_TOKENS[_marks_bucket(target_marks)]
            _q = self._definition_focused_question(sq)
            _mg = MARKS_GUIDANCE[_marks_bucket(target_marks)]
            _key = _ac_key("def", str(prompt), prep["context"], _q, _mg, cap)   # same key as ask()
            _hit = _ac_get(_key)
            if _hit is not None:
                running_text = prefix + _hit
                yield ("token", running_text)
            else:
                chain = prompt | self._llm_with_cap(cap)
                _reason = None
                for chunk in chain.stream({
                    "context": prep["context"],
                    "question": _q,
                    "marks_guidance": _mg,
                }):
                    running_text += chunk.content
                    _r = self._reason_of(chunk)
                    if _r:
                        _reason = _r
                    yield ("token", running_text)
                synthesis_llm_calls += 1
                _body = running_text[len(prefix):]
                _body, _extra = self._complete(_body, self._was_cut(_body, _reason), _q)
                synthesis_llm_calls += _extra
                running_text = prefix + self._strip_figure_refs(_body)
                _ac_put(_key, running_text[len(prefix):].strip())
                yield ("token", running_text)

            # Append deterministic sections
            tail_sections = []
            if match_res is not None and (
                any(i in intents for i in ("frequency", "exact_questions", "years", "importance"))
                or match_res.total_matching_questions > 0
            ):
                tail_sections.append(self.exam_intelligence.format_exam_history_section(match_res))

            if "trend" in intents:
                tail_sections.append(self._format_trend(evidence.get("trend_result"), topic))

            syllabus_info = evidence.get("syllabus_info")
            if syllabus_info and syllabus_info.get("in_syllabus"):
                snippets = syllabus_info.get("matches", [])
                s_lines = ["### 📋 SYLLABUS & NOTES RELEVANCE", "- **Status:** Topic verified present in official MCA syllabus."]
                if snippets and snippets[0].get("snippets"):
                    s_lines.append(f"- **Module Context:** *\"{snippets[0]['snippets'][0]}\"*")
                tail_sections.append("\n".join(s_lines))

            if "importance" in intents or (match_res and match_res.total_matching_questions > 0 and len(intents) > 1):
                tail_sections.append(self.exam_intelligence.generate_study_guidance(match_res, syllabus_info))

            if tail_sections:
                running_text += "\n\n---\n\n" + "\n\n---\n\n".join(tail_sections)
                yield ("token", running_text)

        else:
            # Fully deterministic or lab/ranking intent
            full_ans, calls = self._format_grounded_response(evidence, marks, subject)
            synthesis_llm_calls += calls
            running_text = full_ans
            yield ("token", running_text)

        total_llm_calls = understanding_llm_calls + synthesis_llm_calls
        notes_prep = prep or {}
        final_result = {
            "answer": running_text,
            "sources": evidence.get("sources", []),
            "topic": sq.canonical_topic or context_topic,
            "debug_info": {
                "canonical_topic": sq.canonical_topic,
                "subtopics": sq.subtopics,
                "intents": sq.intents,
                "search_queries": sq.search_queries,
                "question_scope": sq.question_scope,
                "is_followup": sq.is_followup,
                "resolved_reference": sq.resolved_reference,
                "llm_calls": total_llm_calls,
                "matching_questions": match_res.total_matching_questions if match_res else 0,
                "distinct_papers": match_res.distinct_papers if match_res else 0,
                "distinct_years": match_res.distinct_years if match_res else [],
                "best_distance": notes_prep.get("best_distance", notes_prep.get("error", {}).get("debug_info", {}).get("best_distance")),
                "relevance_threshold": MAX_RELEVANT_DISTANCE,
                "method": "LLM Query Understanding → Multi-Level Hybrid Retrieval → Evidence Aggregation → Grounded Synthesis",
            }
        }
        self._log_query(question, subject, sq, final_result)
        yield ("done", final_result)