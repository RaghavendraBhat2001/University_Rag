# CURRICULUM_PATCH_APPLIED
# src/exam_intelligence.py
"""
Exam Intelligence & Student Study Guidance Module.

Analyzes question paper patterns, frequencies, year distributions, and syllabus
alignment to provide fact-grounded study guidance without hallucination.

Provides:
- Distinction between total matching questions and distinct examination papers
- Topic exam history formatting
- Grounded revision guidance separating factual evidence from recommendations
- Repeated questions / top topics identification across the question bank
- Marks-specific question analysis (e.g., 10-mark questions)
- Syllabus cross-referencing
"""

import re
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path

from src.question_index import QuestionRecord
from src.question_matcher import MatchResultSet, QuestionMatch
from config import SYLLABUS_DIR


_SCAN_NOISE = re.compile(
    r"\b(?:P\.?\s?T\.?\s?O\.?|F\.?\s?T\.?\s?O\.?)\s*\.?|Scanned\s+with\s+\w+(?:\s+Scanner)?|\bGE\b(?=\s+Scanned)",
    re.IGNORECASE,
)


# Questions that point at a figure/diagram the paper text does not contain
_DIAGRAM_WORDS = (r"(?:graph(?!\s+(?:traversals?|algorithms?|problems?|types|terminolog\w+))|figure|diagram|"
                  r"tree(?!\s+(?:traversals?|types|operations?|algorithms?|structures?))|circuit|chart|dfa|nfa|"
                  r"automaton|flowchart|game\s+tree)")
_MISSING_DIAGRAM_RES = [
    re.compile(r"\b(?:following|below|given|shown|above)\s+(?:\w+\s+){0,2}?" + _DIAGRAM_WORDS + r"\b", re.IGNORECASE),
    re.compile(r"\b" + _DIAGRAM_WORDS + r"\s+(?:given\s+)?(?:below|above|shown)\b", re.IGNORECASE),
    re.compile(r"\bfor\s+(?:the\s+)?(?:following|below|given)\s*[.:]", re.IGNORECASE),
    re.compile(r"\bfig(?:ure|\.)?\s*\d", re.IGNORECASE),
]


def refers_to_missing_diagram(text: str) -> bool:
    """True when a question depends on a figure/graph/tree that is not part of its extracted text."""
    return bool(text) and any(p.search(text) for p in _MISSING_DIAGRAM_RES)


def clean_question_display(text: str, topic: Optional[str] = None) -> str:
    """Display-only cleanup: removes scanner watermarks, page-turn markers,
    section instructions, and isolates the relevant question clause when multiple
    questions were merged due to OCR noise."""
    cleaned = _SCAN_NOISE.sub(" ", text or "")
    cleaned = re.sub(r'SECTION\s*[-\u2013\u2014:]?\s*[A-Z]\s*(?:Answer\s+any\s+\w+\s+questions)?.*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    # OCR sometimes glues the NEXT question onto this one after a stray marks digit:
    # "...stack and queues. 4 Define Binary Search Tree. Construct...". Split on
    # sentence-end + 1-2 digit number + capitalised word, and keep only the part(s)
    # that mention the topic (or the first part when none does). Display only —
    # the stored question and all counting stay untouched.
    segs = re.split(r'(?<=[.?!])\s+\d{1,2}\s+(?=[A-Z][a-z])', cleaned)
    if len(segs) > 1:
        keep = [s for s in segs if topic and topic.lower().strip() in s.lower()] or segs[:1]
        cleaned = " ".join(s.strip() for s in keep)

    if topic and len(cleaned) > 160:
        topic_lower = topic.lower().strip()
        sentences = re.split(r'(?<=[.!?])\s+|(?<=[a-z]\))\s+|(?<=\d\.)\s+', cleaned)
        matching = [s.strip(' ._-') for s in sentences if topic_lower in s.lower() and len(s.strip()) > 10]
        if matching:
            return ". ".join(matching) + "."

    return cleaned


class ExamIntelligence:
    """Analyzes examination patterns and formulates evidence-based study guidance."""

    def __init__(self, syllabus_dir: Optional[Path] = None):
        self.syllabus_dir = syllabus_dir or SYLLABUS_DIR
        self._syllabus_texts: Optional[Dict[str, str]] = None

    def _ensure_syllabus_loaded(self):
        """Loads text from syllabus PDFs lazily on demand, cached in memory.

        Falls back to OCR per page, same as the main notes/papers loader —
        many of this university's PDFs (including, it turns out, syllabus
        PDFs) are scanned images with no real text layer. Without this, every
        page silently extracts as empty text, self._syllabus_texts ends up
        with entries but no real content, and syllabus_scope() can never
        distinguish "not in the syllabus" from "couldn't read the syllabus" —
        which is exactly why the out-of-syllabus gate never fired on
        anything: it correctly refused to guess when the text was empty.
        """
        if self._syllabus_texts is not None:
            return
        self._syllabus_texts = {}
        if not self.syllabus_dir.exists():
            return

        cache_path = self.syllabus_dir.parent.parent / "faiss_store" / "_syllabus_cache.json"
        if cache_path.exists():
            try:
                import json
                with open(cache_path, "r", encoding="utf-8") as f:
                    self._syllabus_texts = json.load(f)
                if self._syllabus_texts:
                    return
            except Exception:
                pass

        try:
            from pypdf import PdfReader
            from src.document_loader import ocr_page, MIN_TEXT_LENGTH
            for pdf in self.syllabus_dir.glob("**/*.pdf"):
                try:
                    reader = PdfReader(str(pdf))
                    n_pages = len(reader.pages)
                    text_parts = []
                    for i, p in enumerate(reader.pages):
                        if n_pages > 5 and i % 10 == 0:
                            print(f"   📖 Syllabus OCR: page {i + 1}/{n_pages} of {pdf.name}...")
                        txt = (p.extract_text() or "").strip()
                        if len(txt) < MIN_TEXT_LENGTH:
                            ocr_text = ocr_page(pdf, i, dpi=150)
                            if len(ocr_text) > len(txt):
                                txt = ocr_text
                        if txt:
                            text_parts.append(txt)
                    self._syllabus_texts[pdf.name] = "\n".join(text_parts)
                    print(f"   ✅ Syllabus OCR done: {pdf.name} ({n_pages} pages)")
                except Exception:
                    pass

            if self._syllabus_texts:
                try:
                    import json
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(self._syllabus_texts, f)
                except Exception:
                    pass
        except Exception as e:
            print(f"⚠️ Syllabus loading skipped: {e}")

    def syllabus_scope(self, topic: str) -> Optional[bool]:
        """Is this topic inside the syllabus? True / False, or None when that cannot
        be judged (no syllabus uploaded, or its PDFs have no readable text layer —
        e.g. scans, which this loader does not OCR). None must be treated as
        "don't know" — never as "out of syllabus".

        Deliberately looser than check_syllabus_relevance: a strict majority of the
        topic's significant words (5-letter prefixes, so "Quicksort"/"Quick Sort"
        and plurals still match) must appear anywhere in the syllabus text."""
        self._ensure_syllabus_loaded()
        if not self._syllabus_texts:
            return None
        blob = " ".join(self._syllabus_texts.values()).lower()
        if len(blob) < 3000:   # too little text to trust an absence
            return None
        stop = {"the", "and", "for", "with", "from", "into", "using", "what", "how", "are", "its", "vs", "versus"}
        tokens = [w for w in re.findall(r"[a-z]{3,}", (topic or "").lower()) if w not in stop]
        if not tokens:
            return None
        keys = [w if len(w) < 5 else w[:5] for w in tokens]
        hits = sum(1 for k in keys if k in blob)
        return hits * 2 > len(keys)

    def check_syllabus_relevance(self, topic: str) -> Optional[Dict[str, Any]]:
        """Checks if a canonical topic is explicitly mentioned in the university syllabus."""
        if not topic:
            return None
        self._ensure_syllabus_loaded()
        if not self._syllabus_texts:
            return None

        topic_clean = topic.lower().strip()
        matches = []
        for pdf_name, text in self._syllabus_texts.items():
            if topic_clean in text.lower():
                # Extract surrounding context (e.g. unit/module name)
                pattern = rf'(.{{0,80}}{re.escape(topic_clean)}.{{0,120}})'
                found = re.findall(pattern, text, re.IGNORECASE)
                snippets = [re.sub(r'\s+', ' ', s).strip() for s in found[:2]]
                matches.append({
                    "syllabus_file": pdf_name,
                    "snippets": snippets,
                })

        if matches:
            return {
                "in_syllabus": True,
                "matches": matches,
            }
        return {"in_syllabus": False, "matches": []}

    @staticmethod
    def format_exam_history_section(match_result: MatchResultSet, max_display: int = 10) -> str:
        """Formats the EXAM HISTORY section with factual paper and question metrics."""
        topic = match_result.canonical_topic
        total_q = match_result.total_matching_questions
        distinct_p = match_result.distinct_papers
        years = match_result.distinct_years

        if total_q == 0:
            msg = (
                f"### 📊 EXAM HISTORY\n\n"
                f"- **Exam Appearances:** No questions matching **'{topic}'** were found in the currently indexed past examination papers.\n"
                f"- *(Note: This topic may be tested under broader questions, or is from an unindexed sitting/syllabus module.)*"
            )
            return msg

        years_str = ", ".join(years) if years else "Years not detected"
        # Use the SAME distinct-year count as generate_study_guidance (below)
        # for "sittings" language — these two sections used to compute this
        # number two different ways (this one additionally counted each
        # undated/"Year N/A" record as its own phantom extra "sitting" via a
        # (source, year-or-'unknown') set), so they could show two different
        # sitting counts for the same topic in the same answer. Only a real
        # detected year counts as a sitting now, in both places.
        distinct_year_count = len(years) if years else 0
        sitting_str = (
            f"Appeared across **{distinct_year_count}** examination sitting(s) / paper(s)"
            if distinct_year_count > distinct_p
            else f"Appeared in **{distinct_p}** distinct examination paper(s)"
        )
        lines = [
            f"### 📊 EXAM HISTORY",
            f"- **Frequency:** {sitting_str} (**{total_q}** total question(s) found).",
            f"- **Exam Years:** {years_str}",
            f"\n**Previous Exam Questions:**",
        ]

        _tot = getattr(match_result, "total_sittings", 0)
        _csub = getattr(match_result, "coverage_subject", None)
        if _tot and _csub and 0 < distinct_year_count <= _tot:
            lines.insert(3, f"- **Coverage:** asked in **{distinct_year_count}** of **{_tot}** sittings on record for {_csub} ({round(100 * distinct_year_count / _tot)}%).")

        # Display questions sorted by year descending
        literal_matches = [m for m in match_result.matches if m.match_type != "Semantic Match"]
        literal_matches.sort(
            key=lambda m: (int(m.record.year) if m.record.year and m.record.year.isdigit() else 0, m.score),
            reverse=True,
        )
        shown = literal_matches[:max_display]
        for idx, m in enumerate(shown, 1):
            r = m.record
            y_str = f"**{r.year}**" if r.year else "Year N/A"
            m_str = f"{r.marks} Marks" if r.marks > 0 else "Marks N/A"
            sub_tag = f" `[{m.match_type}]`" if m.match_type != "Exact / Near-Exact" else ""
            # Some extracted "question" records span more than one actual
            # question glued together by OCR noise on the source scan (a
            # missed number boundary). Cap what's displayed so one bad
            # extraction doesn't dump an unrelated wall of text under this
            # topic — the underlying grouping is a separate, deeper fix.
            q_display = clean_question_display(r.question, topic=topic)
            q_display = q_display if len(q_display) <= 280 else q_display[:280].rstrip() + "…"
            diag_tag = " 📎 *(refers to a diagram not shown here)*" if refers_to_missing_diagram(r.question) else ""
            lines.append(f"{idx}. [{y_str} — {m_str}] {q_display}{diag_tag} *({r.source})*{sub_tag}")

        if len(literal_matches) > len(shown):
            lines.append(f"\n*(...plus {len(literal_matches) - len(shown)} more question(s) in the examination database)*")

        return "\n".join(lines)

    def generate_study_guidance(
        self,
        match_result: MatchResultSet,
        syllabus_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """Produces evidence-based revision and exam preparation guidance grounded in data."""
        topic = match_result.canonical_topic
        distinct_p = match_result.distinct_papers
        total_q = match_result.total_matching_questions
        years = match_result.distinct_years
        # NOTE: question papers merge multiple exam sittings/years into one PDF
        # per subject (see project convention), so distinct_papers (distinct
        # source FILES) is not the same as the number of separate exam
        # sittings the topic actually appeared in — that's the number of
        # distinct years instead. Use distinct_year_count for any "sittings"
        # language and for tiering; keep distinct_p only for "paper(s)" wording.
        distinct_year_count = len(years) if years else 0
        breakdown = match_result.marks_breakdown

        lines = ["### 💡 STUDY & REVISION GUIDANCE"]

        # 1. Priority Tier
        if distinct_year_count >= 3 or total_q >= 4:
            priority = "**HIGH EXAM PRIORITY (Repeated Topic)**"
            years_display = ", ".join(years) if years else "multiple years"
            advice = (
                f"**'{topic}'** has appeared repeatedly across **{distinct_year_count}** separate exam sittings "
                f"({years_display}). "
                "Examiners frequently test this topic. Ensure you can write full, structured definitions, "
                "draw neat architectural or functional diagrams, and clearly explain operational workflows."
            )
        elif distinct_year_count >= 1 or total_q >= 1:
            priority = "**MODERATE EXAM PRIORITY (Core Topic)**"
            sittings_disp = f"across **{distinct_year_count}** examination sitting(s)" if distinct_year_count > 1 else f"in **{distinct_p}** examination paper(s)"
            advice = (
                f"**'{topic}'** has appeared {sittings_disp}. "
                "While not appearing in every consecutive year, it is a core syllabus concept. "
                "Focus on understanding fundamental definitions, key differences, and standard operations."
            )
        else:
            priority = "**SYLLABUS-LED PRIORITY (Not in Past Papers)**"
            advice = (
                f"**'{topic}'** did not match any indexed past questions directly. "
                "This does not imply it can be skipped: verify its weightage in your syllabus unit. "
                "It may be an emerging concept, a lab-oriented task, or framed under a broader heading."
            )

        lines.append(f"- **Exam Priority Tier:** {priority}")
        lines.append(f"- **Evidence-Based Recommendation:** {advice}")

        # 2. Marks Distribution Insight
        if breakdown:
            # Exact breakdown of how this topic was actually marked (papers use 4/5/6/10
            # marks, so fixed 2-5 / 8-10 buckets used to ignore most questions).
            parts = [f"{mk} marks × {cnt}" for mk, cnt in sorted(breakdown.items())]
            marks_notes = [f"Asked as {', '.join(parts)}."]
            long_count = sum(cnt for mk, cnt in breakdown.items() if mk >= 8)
            if long_count:
                marks_notes.append(f"Tested as a long-answer question (**8+ marks**) {long_count} time(s). Prepare detailed explanations.")
            if marks_notes:
                lines.append(f"- **Marks Pattern:** {' '.join(marks_notes)}")

        # 3. Syllabus Grounding
        if syllabus_info and syllabus_info.get("in_syllabus"):
            lines.append("- **Syllabus Status:** Verified present in official university syllabus documents.")
            snippets = syllabus_info.get("matches", [])
            if snippets and snippets[0].get("snippets"):
                lines.append(f"  - Context: *\"{snippets[0]['snippets'][0]}\"*")

        lines.append("\n> [!NOTE]\n> *Study guidance is computed strictly from your university's indexed examination papers and syllabus documents. It distinguishes observed factual trends from preparation advice.*")

        return "\n".join(lines)

    @staticmethod
    def get_most_repeated_topics_from_bank(
        question_bank: List[QuestionRecord],
        subject: Optional[str] = None,
        top_k: int = 10,
        min_sittings: int = 2
    ) -> List[Dict[str, Any]]:
        """Analyzes question bank to identify frequently repeated questions and themes."""
        if not question_bank:
            return []

        pool = question_bank
        if subject and subject != "All / Auto-detect":
            subj_clean = subject.strip().upper()
            scoped_pool = [r for r in question_bank if r.subject.strip().upper() == subj_clean]
            # Same fallback pattern used everywhere else subject-scoping
            # happens in this project (_paper_pool_for_subject,
            # QuestionMatcher.match): the LLM's extracted `subject` field is
            # free text from the question ("Data Structures") and often
            # won't match the stored abbreviation code ("DS") exactly. If
            # scoping produces zero results, that almost always means a
            # mismatched subject string, not an actually-empty subject —
            # so fall back to the full pool instead of silently returning
            # no clusters at all.
            if scoped_pool:
                pool = scoped_pool

        if not pool:
            return []

        # Find questions with high similarity across different papers/years
        clusters: List[Dict[str, Any]] = []

        for r in pool:
            q_clean = re.sub(r'[^a-zA-Z0-9\s]', '', r.question.lower()).strip()
            if len(q_clean) < 15:
                continue

            # Check if fits existing cluster
            placed = False
            for cl in clusters:
                rep_clean = cl["rep_clean"]
                words1 = set(q_clean.split())
                words2 = set(rep_clean.split())
                jaccard = len(words1 & words2) / max(len(words1 | words2), 1)
                if jaccard >= 0.55:
                    cl["questions"].append(r)
                    cl["papers"].add(r.source)
                    if r.year:
                        cl["years"].add(r.year)
                    placed = True
                    break

            if not placed:
                clusters.append({
                    "representative_question": r.question,
                    "rep_clean": q_clean,
                    "subject": r.subject,
                    "marks": r.marks,
                    "questions": [r],
                    "papers": {r.source},
                    "years": {r.year} if r.year else set(),
                })

        # Filter clusters to only those that appeared in at least min_sittings distinct exam sittings (years)
        repeated_clusters = [cl for cl in clusters if len(cl["years"]) >= min_sittings]

        # Sort clusters by number of distinct exam sittings (years), then question count
        repeated_clusters.sort(key=lambda c: (len(c["years"]), len(c["questions"])), reverse=True)

        results = []
        for cl in repeated_clusters[:top_k]:
            results.append({
                "representative_question": cl["representative_question"],
                "subject": cl["subject"],
                "marks": cl["marks"],
                "distinct_papers": len(cl["papers"]),
                "total_occurrences": len(cl["questions"]),
                "years": sorted(list(cl["years"]), reverse=True),
                "sample_papers": sorted(list(cl["papers"]))[:3],
            })

        return results

    @staticmethod
    def get_questions_by_marks(question_bank: List[QuestionRecord], marks: int, subject: Optional[str] = None, limit: int = 15) -> List[QuestionRecord]:
        """Returns questions matching a specific marks value."""
        pool = question_bank
        if subject and subject != "All / Auto-detect":
            pool = [r for r in question_bank if r.subject.strip().upper() == subject.strip().upper()]

        matching = [r for r in pool if r.marks == marks]
        matching.sort(key=lambda r: (r.year or "", r.source), reverse=True)
        return matching[:limit]