# src/question_matcher.py
"""
Multi-Level Hybrid Question Matcher for University Exam Papers.

Supports:
1. Exact / Near-Exact Question Match:
   High lexical overlap, fuzzy token ratio, verbatim substring matching.
2. Semantic / Topic Match:
   FAISS vector similarity retrieval across embedded past exam questions.
3. Subtopic Match:
   Finds questions targeting specific subtopics (e.g., 'Circular Queue', 'Linear Queue').
4. Metadata Filtering:
   Scoping by subject, target marks (e.g., 10 marks), and year constraints.
5. Distinct Frequency Reporting:
   Accurately separates total matching questions from distinct examination papers and distinct exam years.
"""

import re
import difflib
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any
from langchain_community.vectorstores import FAISS

from src.question_index import QuestionRecord
from src.query_understanding import StructuredQuery


@dataclass
class QuestionMatch:
    record: QuestionRecord
    match_type: str  # "Exact / Near-Exact", "Semantic Match", "Subtopic Match"
    score: float     # 0.0 to 1.0 (higher = better)
    matched_subtopics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.record.question,
            "year": self.record.year,
            "marks": self.record.marks,
            "page": self.record.page,
            "source": self.record.source,
            "subject": self.record.subject,
            "semester": self.record.semester,
            "question_number": self.record.question_number,
            "match_type": self.match_type,
            "score": round(self.score, 3),
            "matched_subtopics": self.matched_subtopics,
        }


@dataclass
class MatchResultSet:
    canonical_topic: str
    total_matching_questions: int
    distinct_papers: int
    distinct_years: List[str]
    matches: List[QuestionMatch]
    by_year: Dict[str, List[QuestionMatch]] = field(default_factory=dict)
    marks_breakdown: Dict[int, int] = field(default_factory=dict)
    subject_scoped: bool = False
    total_sittings: int = 0
    coverage_subject: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "canonical_topic": self.canonical_topic,
            "total_matching_questions": self.total_matching_questions,
            "distinct_papers": self.distinct_papers,
            "distinct_years": self.distinct_years,
            "matches": [m.to_dict() for m in self.matches],
            "distinct_paper_count": self.distinct_papers,
            "marks_breakdown": self.marks_breakdown,
            "subject_scoped": self.subject_scoped,
        }


def _token_set(text: str) -> Set[str]:
    return set(re.findall(r'[a-zA-Z0-9]+', text.lower()))


def _token_overlap_score(s1: str, s2: str) -> float:
    t1 = _token_set(s1)
    t2 = _token_set(s2)
    if not t1 or not t2:
        return 0.0
    intersection = t1 & t2
    return len(intersection) / len(t1)


def _topic_in_text(topic: str, text: str) -> bool:
    """True when `topic` occurs in `text` starting at a word boundary (both lower-case).
    Longer topics may continue into a suffix ("tree" -> "trees", "hash" -> "hashing"); topics of
    3 letters or fewer ("os", "ai", "ml") must be a whole word, so "os" no longer matches "most"
    and "ai" no longer matches "explain"."""
    if not topic:
        return False
    if len(topic) <= 3:
        pattern = r"(?<![a-z0-9])" + re.escape(topic) + r"s?(?![a-z0-9])"
    else:
        pattern = r"(?<![a-z0-9])" + re.escape(topic)
    return re.search(pattern, text) is not None


class QuestionMatcher:
    """Performs multi-level hybrid retrieval across indexed examination questions."""

    def __init__(self, question_bank: List[QuestionRecord], question_vectorstore: Optional[FAISS] = None):
        self.question_bank = question_bank
        self.vectorstore = question_vectorstore

    def set_vectorstore(self, store: Optional[FAISS]):
        self.vectorstore = store

    def _match_pool(
        self,
        pool: List[QuestionRecord],
        structured_query: StructuredQuery,
        subject_filter: Optional[str] = None,
        top_n: int = 25
    ) -> Dict[Tuple[str, Optional[str], str], QuestionMatch]:
        matches_dict: Dict[Tuple[str, Optional[str], str], QuestionMatch] = {}

        topic = structured_query.canonical_topic or structured_query.raw_query
        topic_lower = topic.lower().strip()
        subtopics = [st.strip() for st in structured_query.subtopics if st.strip()]

        # -------------------------------------------------------------
        # LEVEL 1: Exact and Near-Exact Lexical Matching
        # -------------------------------------------------------------
        for r in pool:
            q_text = r.question
            q_lower = q_text.lower()
            key = (r.source, r.year, q_text)

            # Verbatim topic substring match
            if topic_lower and _topic_in_text(topic_lower, q_lower):
                score = 0.95 if topic_lower == q_lower else 0.85
                matches_dict[key] = QuestionMatch(
                    record=r,
                    match_type="Exact / Near-Exact",
                    score=score,
                    matched_subtopics=[topic] if topic else [],
                )
                continue

            # Fuzzy sequence similarity
            ratio = difflib.SequenceMatcher(None, topic_lower, q_lower).ratio()
            if ratio >= 0.75:
                matches_dict[key] = QuestionMatch(
                    record=r,
                    match_type="Exact / Near-Exact",
                    score=ratio,
                    matched_subtopics=[topic],
                )

        # -------------------------------------------------------------
        # LEVEL 2: Subtopic Matching
        # -------------------------------------------------------------
        if subtopics:
            for r in pool:
                key = (r.source, r.year, r.question)
                q_lower = r.question.lower()
                matched_subs = []
                for st in subtopics:
                    st_lower = st.lower()
                    if _topic_in_text(st_lower, q_lower) or difflib.SequenceMatcher(None, st_lower, q_lower).ratio() > 0.8:
                        matched_subs.append(st)

                if matched_subs:
                    if key in matches_dict:
                        # Upgrade existing match with matched subtopics
                        matches_dict[key].matched_subtopics.extend([s for s in matched_subs if s not in matches_dict[key].matched_subtopics])
                        matches_dict[key].score = max(matches_dict[key].score, 0.90)
                    else:
                        matches_dict[key] = QuestionMatch(
                            record=r,
                            match_type="Subtopic Match",
                            score=0.88,
                            matched_subtopics=matched_subs,
                        )

        # -------------------------------------------------------------
        # LEVEL 3: Semantic Vector Matching
        # -------------------------------------------------------------
        if self.vectorstore is not None:
            search_phrases = structured_query.search_queries or [topic]
            for phrase in search_phrases[:3]:
                try:
                    scored_docs = self.vectorstore.similarity_search_with_score(phrase, k=top_n)
                    for doc, score in scored_docs:
                        if score > 0.75:  # CUTOFF_PATCH_APPLIED (calibrated, was 1.15)
                            continue

                        meta = doc.metadata or {}
                        rec_id = meta.get("record_id")
                        r = None
                        if rec_id is not None and 0 <= rec_id < len(self.question_bank):
                            r = self.question_bank[rec_id]
                        else:
                            for cand in pool:
                                if cand.question == doc.page_content:
                                    r = cand
                                    break

                        if r is None:
                            continue

                        if subject_filter and r.subject.strip().upper() != subject_filter.strip().upper():
                            continue

                        key = (r.source, r.year, r.question)
                        norm_score = max(0.5, 1.0 - (score / 2.0))

                        if key in matches_dict:
                            matches_dict[key].score = max(matches_dict[key].score, norm_score)
                        else:
                            matches_dict[key] = QuestionMatch(
                                record=r,
                                match_type="Semantic Match",
                                score=norm_score,
                                matched_subtopics=[],
                            )
                except Exception as e:
                    print(f"⚠️ Vector search error in QuestionMatcher: {e}")
        else:
            search_phrases = structured_query.search_queries or [topic]
            for r in pool:
                key = (r.source, r.year, r.question)
                if key in matches_dict:
                    continue
                q_lower = r.question.lower()
                for phrase in search_phrases:
                    overlap = _token_overlap_score(phrase, q_lower)
                    if overlap >= 0.70:
                        matches_dict[key] = QuestionMatch(
                            record=r,
                            match_type="Semantic Match",
                            score=overlap * 0.85,
                            matched_subtopics=[],
                        )
                        break

        return matches_dict

    def match(self, structured_query: StructuredQuery, top_n: int = 25,
              allow_cross_subject_fallback: bool = True) -> MatchResultSet:
        """Executes multi-level hybrid matching for a StructuredQuery."""
        if not self.question_bank:
            return MatchResultSet(
                canonical_topic=structured_query.canonical_topic,
                total_matching_questions=0,
                distinct_papers=0,
                distinct_years=[],
                matches=[],
            )

        # 1. Subject pool filtering
        subject_filter = structured_query.subject
        pool = self.question_bank
        subject_scoped = False
        if subject_filter and subject_filter != "All / Auto-detect":
            subj_clean = subject_filter.strip().upper()
            filtered_pool = [r for r in self.question_bank if r.subject.strip().upper() == subj_clean]
            if filtered_pool:
                pool = filtered_pool
                subject_scoped = True

        matches_dict = self._match_pool(
            pool, structured_query, subject_filter if subject_scoped else None, top_n=top_n
        )

        # Cross-subject fallback: if subject filter was applied but yielded 0 matches,
        # fall back to search across all subjects in the question bank.
        # TREND_FIX: the Exam Intelligence trend passes allow_cross_subject_fallback=False so a subject
        # with no matches shows "no matches" instead of silently reporting another subject's questions.
        if subject_scoped and not matches_dict and allow_cross_subject_fallback:
            fallback_matches = self._match_pool(self.question_bank, structured_query, None, top_n=top_n)
            if fallback_matches:
                matches_dict = fallback_matches
                subject_scoped = False

        topic = structured_query.canonical_topic or structured_query.raw_query

        # -------------------------------------------------------------
        # LEVEL 4: Explicit Filter Application (Marks, Years)
        # -------------------------------------------------------------
        all_matches = list(matches_dict.values())

        # If specific marks requested (e.g. 10 marks), prioritize or filter
        if structured_query.marks:
            target_marks = structured_query.marks
            # Boost matches matching target marks
            for m in all_matches:
                if m.record.marks == target_marks:
                    m.score += 0.05

        # If specific year constraint
        if structured_query.year_constraint:
            target_year = str(structured_query.year_constraint).strip()
            if target_year.isdigit():
                # Filter to matching year if available
                year_filtered = [m for m in all_matches if m.record.year == target_year]
                if year_filtered:
                    all_matches = year_filtered

        # Sort matches by score descending, then by year descending
        all_matches.sort(
            key=lambda m: (m.score, int(m.record.year) if m.record.year and m.record.year.isdigit() else 0),
            reverse=True
        )

        # Headline statistics (how many questions, which sittings, marks pattern) count
        # ONLY literal matches: "Exact / Near-Exact" and "Subtopic Match". A "Semantic Match"
        # is just a similar-sounding question (e.g. "Define data structures" for a query on
        # "queue"), so it is still shown to the student as a related question but is never
        # counted as an appearance of the topic. Counted BEFORE the top_n cut so related
        # questions can never push a real match out of the totals.
        counted = [m for m in all_matches if m.match_type != "Semantic Match"]

        # Keep EVERY literal match (needed for exact per-year counts); only the loosely related
        # semantic extras are capped at top_n.
        semantic_extras = [m for m in all_matches if m.match_type == "Semantic Match"][:top_n]
        all_matches = sorted(
            counted + semantic_extras,
            key=lambda m: (m.score, int(m.record.year) if m.record.year and m.record.year.isdigit() else 0),
            reverse=True,
        )

        # Calculate metrics
        distinct_papers = len({m.record.source for m in counted if m.record.source})
        years_set = {m.record.year for m in counted if m.record.year}
        distinct_years = sorted(list(years_set), reverse=True)

        by_year: Dict[str, List[QuestionMatch]] = {}
        marks_breakdown: Dict[int, int] = {}

        for m in all_matches:
            y = m.record.year or "Unknown Year"
            by_year.setdefault(y, []).append(m)
        for m in counted:
            if m.record.marks > 0:
                marks_breakdown[m.record.marks] = marks_breakdown.get(m.record.marks, 0) + 1

        # Sittings on record for the subject(s) these matches belong to (for "asked in N of M sittings").
        _subjects = {m.record.subject for m in counted if m.record.subject}
        total_sittings, coverage_subject = 0, None
        if len(_subjects) == 1:
            coverage_subject = next(iter(_subjects))
            total_sittings = len({r.year for r in self.question_bank if r.year and r.subject == coverage_subject})

        return MatchResultSet(
            canonical_topic=topic,
            total_matching_questions=len(counted),
            distinct_papers=distinct_papers,
            distinct_years=distinct_years,
            matches=all_matches,
            by_year=by_year,
            marks_breakdown=marks_breakdown,
            subject_scoped=subject_scoped,
            total_sittings=total_sittings,
            coverage_subject=coverage_subject,
        )