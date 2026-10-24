# src/question_index.py
"""
Structured Question Bank and Vector Index for University Exam Papers.

During document ingestion or rebuild:
1. Every question paper is parsed page-by-page.
2. Individual examination questions are extracted deterministically along with
   metadata: subject, year, marks, page number, question number, semester, university, and source PDF.
3. Structured QuestionRecords are stored in `_question_bank.json`.
4. A dedicated FAISS vector index (`_question_bank_faiss`) is built using question embeddings,
   enabling semantic, subtopic, and hybrid question retrieval alongside exact matching.
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any, Tuple

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from src.document_loader import extract_questions_from_text
from config import UNIVERSITY_NAME


@dataclass
class QuestionRecord:
    subject: str
    question: str
    year: Optional[str]
    marks: int
    page: int
    source: str
    semester: Optional[str] = None
    university: Optional[str] = UNIVERSITY_NAME
    question_number: Optional[str] = None
    subtopics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuestionRecord":
        return cls(
            subject=data.get("subject", "UNKNOWN"),
            question=data.get("question", ""),
            year=data.get("year"),
            marks=int(data.get("marks", 0) or 0),
            page=int(data.get("page", 0) or 0),
            source=data.get("source", "unknown"),
            semester=data.get("semester"),
            university=data.get("university", UNIVERSITY_NAME),
            question_number=data.get("question_number"),
            subtopics=data.get("subtopics") or [],
        )


_LAB_RE = re.compile(r'\blabs?\b|\blaborator(?:y|ies)\b|\bpracticals?\b', re.IGNORECASE)


def build_question_bank(question_paper_page_docs, university: Optional[str] = None) -> List[QuestionRecord]:
    """Extracts structured questions from whole-page question paper Documents.
    Groups pages by source PDF to accurately preserve page numbers and multi-page questions.
    """
    by_source: Dict[str, List[Any]] = {}
    for doc in question_paper_page_docs:
        src = doc.metadata.get("source", "unknown")
        by_source.setdefault(src, []).append(doc)

    records: List[QuestionRecord] = []
    effective_university = university or UNIVERSITY_NAME

    for source, page_docs in by_source.items():
        subject = page_docs[0].metadata.get("subject", "UNKNOWN") if page_docs else "UNKNOWN"
        semester = page_docs[0].metadata.get("semester") if page_docs else None

        # Exclude laboratory practical manuals/assessments from university theory exam bank
        if _LAB_RE.search(source) or _LAB_RE.search(subject):
            continue

        page_texts = [d.page_content for d in page_docs]
        questions = extract_questions_from_text(page_texts)

        for q in questions:
            page_idx = q.get("page", 0)
            year = page_docs[page_idx].metadata.get("year") if 0 <= page_idx < len(page_docs) else None
            q_num = str(q.get("number", "")).strip() or None
            q_text = str(q.get("text", "")).strip()

            records.append(QuestionRecord(
                subject=subject,
                question=q_text,
                year=str(year) if year else None,
                marks=int(q.get("marks", 0) or 0),
                page=page_idx,
                source=source,
                semester=str(semester) if semester else None,
                university=effective_university,
                question_number=q_num,
                subtopics=[],
            ))

    return records


def build_question_vectorstore(records: List[QuestionRecord], embeddings, store_path: Optional[Path] = None) -> Optional[FAISS]:
    """Builds a FAISS vector store embedding each question's text with attached metadata."""
    if not records or embeddings is None:
        return None

    docs = []
    for idx, r in enumerate(records):
        if not r.question or len(r.question.strip()) < 5:
            continue
        metadata = {
            "record_id": idx,
            "subject": r.subject,
            "year": r.year,
            "marks": r.marks,
            "page": r.page,
            "source": r.source,
            "semester": r.semester,
            "university": r.university,
            "question_number": r.question_number,
        }
        docs.append(Document(page_content=r.question, metadata=metadata))

    if not docs:
        return None

    try:
        store = FAISS.from_documents(docs, embeddings)
        if store_path:
            store_path = Path(store_path)
            store_path.mkdir(parents=True, exist_ok=True)
            store.save_local(str(store_path))
        return store
    except Exception as e:
        print(f"⚠️ Failed to build question vector store: {e}")
        return None


def load_question_vectorstore(store_path: Path, embeddings) -> Optional[FAISS]:
    """Loads existing FAISS vector store for questions if available."""
    p = Path(store_path)
    if not p.exists() or embeddings is None:
        return None
    try:
        return FAISS.load_local(str(p), embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"⚠️ Failed to load question vector store from {store_path}: {e}")
        return None


def save_question_bank(records: List[QuestionRecord], path: Path):
    """Saves QuestionRecord list to JSON."""
    import os as _os
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # SAFETY_FIX: write to a temp file first; the real file is only replaced once the write is complete
    tmp = p.with_name(p.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in records], f, indent=2, ensure_ascii=False)
            f.flush()
            _os.fsync(f.fileno())
        _os.replace(tmp, p)
    except BaseException:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise


def load_question_bank(path: Path) -> List[QuestionRecord]:
    """Loads QuestionRecord list from JSON, returns [] if file does not exist."""
    p = Path(path)
    if not p.exists():
        return []
    try:
        with open(p, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return [QuestionRecord.from_dict(r) for r in raw]
    except Exception as e:
        print(f"⚠️ Failed to load question bank ({e}) — treating as empty until next rebuild")
        return []


def search_question_bank(records: List[QuestionRecord], topic: str, subject: Optional[str] = None) -> Tuple[List[QuestionRecord], int]:
    """Deterministic substring search over question text (backward compatible).
    Returns (matching_records, distinct_paper_count).
    """
    if not topic:
        return [], 0
    topic_lower = topic.lower().strip()
    subject_upper = subject.strip().upper() if subject and subject != "All / Auto-detect" else None

    matches = []
    for r in records:
        if subject_upper and r.subject.strip().upper() != subject_upper:
            continue
        if topic_lower in r.question.lower():
            matches.append(r)

    distinct_papers = len({r.source for r in matches})
    return matches, distinct_papers
