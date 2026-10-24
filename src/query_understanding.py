# src/query_understanding.py
"""
LLM-based Query Understanding & Query Normalization Layer.

Transforms raw, informal, or compound student queries into structured retrieval
instructions (StructuredQuery) without hardcoding subject- or topic-specific rules.

Handles:
- Canonical topic extraction and normalization (typo correction, standard academic terminology)
- Subtopic identification (e.g., Linear Queue, Circular Queue)
- Multi-intent classification (definition, frequency, exact_questions, years, importance, trend, ranking, lab_program)
- Search query expansion for hybrid vector/lexical retrieval
- Conversational follow-ups and reference resolution (e.g., "How many times was this asked?")
- Explicit constraint extraction (marks, years, subject)
- Robust offline/fallback parsing when the LLM is unreachable
"""

import json
import re
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from langchain_core.prompts import ChatPromptTemplate


@dataclass
class StructuredQuery:
    raw_query: str
    canonical_topic: str
    subtopics: List[str] = field(default_factory=list)
    intents: List[str] = field(default_factory=lambda: ["definition"])
    search_queries: List[str] = field(default_factory=list)
    question_scope: str = "related"  # "exact", "subtopic", "related"
    subject: Optional[str] = None
    marks: Optional[int] = None
    year_constraint: Optional[str] = None
    is_followup: bool = False
    resolved_reference: Optional[str] = None
    needs_notes: bool = True
    needs_question_bank: bool = False
    needs_syllabus: bool = False
    needs_lab: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Unambiguous hard signals checked quickly before calling the LLM
LAB_PROGRAM_PATTERNS = [
    r'\bwrite (a |the )?(c |c\+\+|java|python )?program\b',
    r'\bwrite (the )?code\b',
    r'\bcode for\b',
    r'\bimplement (a |the )?\w+ in (c|c\+\+|java|python)\b',
]

RANKING_PATTERNS = [
    r'most (frequently|often) (asked|appeared|repeated|occurring)',
    r'which .{0,15}(is|are) (the )?most (frequent|common|repeated|important|asked)',
    r'top (question|topic)s?\b',
    r'most (common|frequent|repeated) (question|topic)',
]

ALL_INTENTS = {
    "definition",
    "frequency",
    "exact_questions",
    "years",
    "importance",
    "trend",
    "ranking",
    "lab_program",
    "similar_questions",
    "syllabus",
}

_STOPWORDS = {
    "how", "many", "times", "time", "has", "had", "have", "having", "did", "does", "do", "is", "it", "this",
    "that", "these", "those", "was", "were", "been", "being", "should", "i", "we", "you", "they", "to", "for",
    "the", "a", "an", "of", "in", "on", "and", "or", "will", "come", "comes", "exam", "exams", "examination",
    "examinations", "question", "questions", "paper", "papers", "topic", "important", "necessary", "pass", "read",
    "study", "studying", "prepare", "learn", "learning", "revise", "revising", "reading",
    "need", "worth", "appeared", "asked", "repeated", "can", "skip", "me", "what", "are",
    "define", "explain", "describe", "meaning", "tell", "about", "give", "trend", "trends",
    "rising", "falling", "increasing", "decreasing", "becoming", "less", "more", "over",
    "years", "year", "recent", "wise", "by", "past", "show", "exact", "which",
}

_FILLER_WORDS = {
    "one", "ones", "too", "also", "again", "before", "same", "thing",
    "things", "there", "here", "way", "its", "so", "now", "still",
    "up", "ever", "any", "anything", "out",
}

# Words that genuinely point back at the previous topic.
_FOLLOWUP_CUES = re.compile(
    r"\b(it|its|this|that|these|those|them|they|same|above|previous|earlier|again)\b", re.IGNORECASE
)

# Deterministic backstop for "trend" — a 7B model does not reliably catch every
# phrasing of "is this asked consistently across years" from the description alone.
_TREND_CUES = re.compile(
    r"\b(trend|rising|falling|increasing|decreasing|declining|recurring|consisten\w*|"
    r"regularly|every\s+year|each\s+year|only\s+sometimes|how\s+often|"
    r"getting\s+(?:more|less)\s+common|still\s+(?:relevant|asked|being\s+asked|common)|"
    r"over\s+the\s+years?)\b",
    re.IGNORECASE,
)

# Real lab-program requests reliably say "write a/an/the program", "write code",
# "implement X", "algorithm in <language>", "pseudocode", or "using <language>".
# If a question was classified lab_program but contains none of these, it's a plain
# explain/describe/write-a-note-on question the classifier got wrong — seen repeatedly
# in real papers ("write brief notes on X", "write in detail about X") where "write"
# means "explain in writing", not "write source code". Unconditional once lab_program
# is present and this doesn't match — no separate "does it look like an explain verb"
# check needed, since a genuine lab request always has one of these markers.
_LAB_REAL_RE = re.compile(
    r"\b((write|develop|design|create)\s+(a\s+|an\s+|the\s+)?(c\s+|c\+\+\s+|java\s+|python\s+)?program|"
    r"implement|"
    r"(write|develop|design|create)\s+(a\s+|an\s+|the\s+)?(c\s+|c\+\+\s+|java\s+|python\s+)?(code|algorithm)|"
    r"algorithm\s+in\s+(c|c\+\+|java|python)|pseudo ?code|using\s+(c|c\+\+|java|python))\b",
    re.IGNORECASE,
)


def _veto_false_lab(sq: StructuredQuery, question: str) -> StructuredQuery:
    if "lab_program" in sq.intents and not _LAB_REAL_RE.search(question):
        sq.intents = [i for i in sq.intents if i != "lab_program"] or ["definition"]
        sq.needs_lab = False
        sq.needs_notes = True
    return sq


# LAB_SUBJECT_PATCH_APPLIED -- lab subjects treat program-style questions as lab requests
_LAB_SUBJECT_RE = re.compile(r"\blabs?\b|\bpracticals?\b|\blaborator(?:y|ies)\b", re.IGNORECASE)
_THEORY_START_RE = re.compile(
    r"^\s*(what\s+(is|are)|define|explain|describe|differentiate|difference\s+between|compare|why|"
    r"advantages?|disadvantages?)\b", re.IGNORECASE)
_CODE_WORD_RE = re.compile(r"\b(program|code|algorithm|implement\w*)\b", re.IGNORECASE)
_STATS_RE = re.compile(r"\b(how\s+many\s+times|how\s+often|frequen\w*|appeared|which\s+years?|trend)\b", re.IGNORECASE)
_NOT_CODE_INTENTS = {"frequency", "exact_questions", "years", "trend", "ranking",
                     "similar_questions", "syllabus", "importance"}


def _force_lab_for_lab_subject(sq: "StructuredQuery", question: str, subject_context: Optional[str]) -> "StructuredQuery":
    if not subject_context or not _LAB_SUBJECT_RE.search(subject_context):
        return sq
    if "lab_program" in sq.intents:
        return sq
    if any(i in _NOT_CODE_INTENTS for i in sq.intents) or _STATS_RE.search(question):
        return sq
    if _THEORY_START_RE.match(question) and not _CODE_WORD_RE.search(question):
        return sq
    sq.intents = ["lab_program"]
    sq.needs_lab = True
    sq.needs_notes = True
    sq.needs_question_bank = False
    sq.needs_syllabus = False
    return sq


def _topic_matches_query(topic: str, query: str) -> bool:
    """False when the assigned canonical_topic shares no real word (and no close
    spelling-variant prefix) with the actual question — e.g. a header reading
    "NORMALIZATION" over an answer about "Data cleaning". True (permissive) whenever
    either side has no extractable words, so this never fires on something it can't
    meaningfully judge."""
    t = set(re.findall(r"[a-z]{3,}", (topic or "").lower()))
    q = set(re.findall(r"[a-z]{3,}", (query or "").lower()))
    if not t or not q:
        return True
    return bool(t & q) or any(a.startswith(b[:5]) or b.startswith(a[:5]) for a in t for b in q)


_COMPARE_PATTERNS = [
    re.compile(r"\b(?:difference|differences|diff|compare|comparison)\b.*?\bbetween\s+(.+?)\s+(?:and|&|vs\.?|versus)\s+(.+?)\s*[?.!]*$", re.IGNORECASE),
    re.compile(r"^(?:compare\s+)?(.+?)\s+(?:vs\.?|versus)\s+(.+?)(?:\s+(?:diff|difference|differences|important|imp)\b.*?)?\s*[?.!]*$", re.IGNORECASE),
]


def _looks_like_followup(question: str) -> bool:
    """A question is a genuine follow-up only if it points back with a pronoun
    ("it", "that"...) OR has no content words of its own ("how many times?",
    "show exact questions"). A new question that merely lacks a topic name
    ("how many subjects in 1st sem") is NOT a follow-up."""
    return bool(_FOLLOWUP_CUES.search(question)) or not _clean_fallback_topic(question)


def _extract_comparison(question: str):
    for pat in _COMPARE_PATTERNS:
        m = pat.search(question.strip())
        if m:
            a, b = m.group(1).strip(" ,.?"), m.group(2).strip(" ,.?")
            if a and b and len(a) < 40 and len(b) < 40:
                return a, b
    return None


def _clean_fallback_topic(text: str) -> str:
    """Offline heuristic topic extractor when LLM is unavailable."""
    q = re.sub(r'[?!.,\'"]', ' ', text.lower())
    words = [w for w in re.split(r'\s+', q) if w]
    keep = [w for w in words if w not in _STOPWORDS and w not in _FILLER_WORDS]
    return " ".join(keep).strip()


# FAST_PARSE_PATCH_APPLIED
FAST_PARSE_DEFINITIONS = True   # False = always use the LLM parser (old behaviour)
_FAST_DEF_RE = re.compile(
    r"^\s*(?:what\s+is|what\s+are|define|explain|describe)\s+(?:a\s+|an\s+|the\s+)?"
    r"([A-Za-z][A-Za-z0-9+#\-]*(?:\s+[A-Za-z][A-Za-z0-9+#\-]*){0,3})\s*[?.]?\s*$",
    re.IGNORECASE,
)
# anything that hints at a compound / exam-meta / comparison / lab question -> keep the LLM parser
_FAST_DEF_BLOCK = re.compile(
    r"\b(of|in|and|or|vs|versus|between|for|with|using|how|why|which|when|many|times|exam|exams|"
    r"year|years|marks?|important|types?|diagram|program|code|algorithm|syllabus|trend|"
    r"difference|differences|compare|write|show|list)\b",
    re.IGNORECASE,
)


def _restore_case(topic: str, question: str) -> str:
    """'avl tree' -> 'AVL Tree' (keeps words the student typed in capitals, e.g. AVL, OSI, TCP)."""
    typed = {w.lower(): w for w in re.findall(r"[A-Za-z0-9+#\-]+", question)}
    out = []
    for w in topic.split():
        t = typed.get(w.lower(), w)
        out.append(t if (t.isupper() and len(t) > 1) else w.capitalize())
    return " ".join(out)


class QueryUnderstanding:
    """Interprets student natural language queries using a local LLM."""

    def __init__(self, llm_callable=None):
        self._llm_callable = llm_callable

    def _build_prompt(self, question: str, context_topic: Optional[str] = None, subject_context: Optional[str] = None) -> str:
        context_instruction = ""
        if context_topic:
            context_instruction = (
                f'\nCONVERSATION CONTEXT:\nThe student was recently discussing: "{context_topic}".\n'
                f'Only treat this question as a follow-up (set canonical_topic to "{context_topic}" and '
                f'is_followup to true) if it CLEARLY continues that same topic — e.g. it uses a pronoun or '
                f'implicit reference like "it", "this", "that", "this topic", "these questions" that can only '
                f'sensibly mean "{context_topic}", or is a short continuation like "how many times?", "show '
                f'exact questions", "is it important?" that makes no sense as a standalone question about '
                f'anything else.\n'
                f'Do NOT treat it as a follow-up merely because it does not explicitly name a topic — a new, '
                f'unrelated, or administrative question (e.g. about timetables, subjects offered, general '
                f'university info) is NEVER a follow-up to "{context_topic}" just because it is short or vague. '
                f'In that case extract canonical_topic from the question itself (or leave it as whatever the '
                f'question is actually about) and set is_followup to false.\n'
            )

        subject_instruction = ""
        if subject_context and subject_context != "All / Auto-detect":
            subject_instruction = f'\nCURRENT SUBJECT SCOPE: "{subject_context}".\n'

        return f"""You are the Query Understanding & Normalization Engine for a University Academic & Examination Assistant.
Your task is to analyze a student's natural language query and convert it into a structured JSON query object.

Do NOT answer the question. Only extract and structure the student's request.
{context_instruction}{subject_instruction}
Analyze the question for:
1. "canonical_topic": The standard academic subject or concept (e.g. "Queue", "Stack", "Binary Search Tree", "Decision Tree", "Normalization", "Process Scheduling", "Quick Sort"). Correct any typos. If a follow-up refers to the prior topic, use that.
2. "subtopics": List of specific subtopics, components, or variants mentioned (e.g. ["Linear Queue", "Circular Queue"]). Empty list if none.
3. "intents": A list containing one or more of:
   - "definition": asks for explanation, definition, concepts, or what something is
   - "frequency": asks how many times or how often it appeared in previous exams
   - "exact_questions": asks to see the actual past exam questions verbatim
   - "years": asks which years or exam sessions it appeared in
   - "importance": asks if they should study it, if it is important, or asks for revision guidance
   - "trend": asks about the pattern of appearance across years — becoming more/less frequent,
     asked every year vs only some years, how consistently/regularly it comes up, or whether it is
     "still" being asked
   - "ranking": asks for most repeated or top topics across the subject
   - "lab_program": asks for code, programming implementation, or algorithm execution
   - "similar_questions": asks for questions similar to a given question
   - "syllabus": asks whether it is in the syllabus or which unit/module
4. "search_queries": 2 to 4 diverse search phrases to retrieve relevant notes and past exam questions (e.g. ["queue", "linear queue", "circular queue", "operations of queue"]).
5. "question_scope": "exact" (if looking for a specific question), "subtopic" (if focused on specific subtopics), or "related" (general topic search).
6. "marks": Integer if the student explicitly asked for a specific marks level (e.g. 2, 5, 8, 10), else null.
7. "year_constraint": Specific year string if mentioned (e.g. "2024", "last 5 years"), else null.
8. "subject": Subject name if explicitly mentioned in the query (e.g. "Data Structures", "DBMS", "Operating Systems"), else null.
9. "is_followup": boolean indicating if this is a follow-up referencing the previous topic.

OUTPUT FORMAT:
Return ONLY a valid JSON object matching this schema. No markdown fences, no explanation:
{{
  "canonical_topic": "...",
  "subtopics": [...],
  "intents": ["..."],
  "search_queries": ["..."],
  "question_scope": "...",
  "marks": null,
  "year_constraint": null,
  "subject": null,
  "is_followup": false
}}

EXAMPLES:
Query: "what is queue? linear and circular. how many times it appeared. show exact questions and years."
Response:
{{
  "canonical_topic": "Queue",
  "subtopics": ["Linear Queue", "Circular Queue"],
  "intents": ["definition", "frequency", "exact_questions", "years"],
  "search_queries": ["queue", "linear queue", "circular queue", "linear and circular queue"],
  "question_scope": "subtopic",
  "marks": null,
  "year_constraint": null,
  "subject": null,
  "is_followup": false
}}

Query: "what is decision tree? how many times has this question appeared? should i read this to pass the exam?"
Response:
{{
  "canonical_topic": "Decision Tree",
  "subtopics": [],
  "intents": ["definition", "frequency", "importance"],
  "search_queries": ["decision tree", "decision tree classification", "decision tree algorithm"],
  "question_scope": "related",
  "marks": null,
  "year_constraint": null,
  "subject": null,
  "is_followup": false
}}

Query: "show me the exact questions" (with context "Decision Tree")
Response:
{{
  "canonical_topic": "Decision Tree",
  "subtopics": [],
  "intents": ["exact_questions"],
  "search_queries": ["decision tree"],
  "question_scope": "exact",
  "marks": null,
  "year_constraint": null,
  "subject": null,
  "is_followup": true
}}

Student Query: {question}
JSON:"""

    def parse(self, question: str, context_topic: Optional[str] = None, subject_context: Optional[str] = None) -> StructuredQuery:
        """Parses a student query into a StructuredQuery object."""
        sq = self._parse_core(question, context_topic, subject_context)
        return _force_lab_for_lab_subject(sq, question, subject_context)  # LAB_SUBJECT_PATCH_APPLIED

    def _parse_core(self, question: str, context_topic: Optional[str] = None, subject_context: Optional[str] = None) -> StructuredQuery:
        """The original parse (unchanged)."""
        q_clean = question.strip()
        q_lower = q_clean.lower()
        self.last_parse_used_llm = False  # REVIEW_FIXES_APPLIED: true only if the LLM answered

        # Hard signals check (zero-cost instant bypass if completely explicit)
        is_lab = any(re.search(pat, q_lower) for pat in LAB_PROGRAM_PATTERNS)
        is_ranking = any(re.search(pat, q_lower) for pat in RANKING_PATTERNS)

        if is_lab and not any(w in q_lower for w in ["frequency", "how many times", "exam", "years"]):
            topic = _clean_fallback_topic(question) or (context_topic or "Program")
            return StructuredQuery(
                raw_query=question,
                canonical_topic=topic,
                subtopics=[],
                intents=["lab_program"],
                search_queries=[question, topic],
                question_scope="related",
                subject=subject_context,
                needs_notes=False,
                needs_question_bank=False,
                needs_lab=True,
            )

        if is_ranking and not any(w in q_lower for w in ["what is", "define", "explain"]):
            return StructuredQuery(
                raw_query=question,
                canonical_topic=context_topic or "",
                subtopics=[],
                intents=["ranking"],
                search_queries=["frequently asked questions", "top questions"],
                question_scope="related",
                subject=subject_context,
                needs_notes=False,
                needs_question_bank=True,
            )

        # FAST_PARSE_PATCH_APPLIED: plain one-topic definition question -> no LLM parse needed
        if (FAST_PARSE_DEFINITIONS and _FAST_DEF_RE.match(q_clean)
                and not _FAST_DEF_BLOCK.search(q_clean) and not _looks_like_followup(q_clean)):
            _fast = self._heuristic_fallback(question, context_topic, subject_context)
            if _fast.intents == ["definition"] and _fast.canonical_topic and not _fast.is_followup:
                _fast.canonical_topic = _restore_case(_fast.canonical_topic, q_clean)
                _fast.search_queries = [f"What is {_fast.canonical_topic}?", _fast.canonical_topic]
                return self._post_process(_fast, question, context_topic)

        # Primary path: LLM Query Understanding
        if self._llm_callable:
            try:
                prompt_text = self._build_prompt(
                    question, context_topic if _looks_like_followup(question) else None, subject_context)  # REPEAT_STABLE_PATCH_APPLIED
                raw_response = self._llm_callable(prompt_text)
                cleaned_response = self._clean_json(raw_response)
                parsed = json.loads(cleaned_response)
                sq = self._build_structured_query(question, parsed, context_topic, subject_context)
                self.last_parse_used_llm = True
                return self._post_process(sq, question, context_topic)
            except Exception as e:
                print(f"⚠️ LLM Query Understanding fallback triggered: {e}")

        # Fallback offline heuristic
        return self._heuristic_fallback(question, context_topic, subject_context)

    def _post_process(self, sq: StructuredQuery, question: str, context_topic: Optional[str]) -> StructuredQuery:
        """Deterministic corrections applied after the LLM parse — the LLM is
        good at reading messy phrasing but unreliable at knowing when NOT to
        reuse the previous topic."""
        # 1. Stop topic carry-over on questions that are not follow-ups.
        if (
            context_topic
            and sq.canonical_topic
            and sq.canonical_topic.strip().lower() == context_topic.strip().lower()
            and not _looks_like_followup(question)
        ):
            own_topic = _clean_fallback_topic(question)
            if own_topic and sq.canonical_topic.strip().lower() not in question.lower():  # REPEAT_STABLE_PATCH_APPLIED
                sq.search_queries = [own_topic if q.strip().lower() == context_topic.strip().lower() else q for q in sq.search_queries] or [own_topic]
                sq.canonical_topic = own_topic
                sq.subtopics = []
                sq.is_followup = False
                sq.resolved_reference = None

        # 1b. Trend backstop — add it if the question clearly asks about it,
        # even when the LLM classified it as something else instead (e.g.
        # frequency/years). Never REMOVES an intent the LLM found; only adds.
        if _TREND_CUES.search(question) and "trend" not in sq.intents:
            sq.intents.append("trend")

        # 2. Comparison questions ("stack vs queue"): retrieve BOTH sides.
        cmp = _extract_comparison(question)
        if cmp and sq.intents == ["definition"]:
            a, b = cmp
            sq.search_queries = [a, b] + [q for q in sq.search_queries if q.lower() not in (a.lower(), b.lower())]
            sq.canonical_topic = f"{a.title()} vs {b.title()}"
            sq.subtopics = []

        # 3. A question classified lab_program that doesn't actually ask for code.
        sq = _veto_false_lab(sq, question)

        # 4. Header/answer mismatch: canonical_topic drifted away from what was
        # actually asked (seen on very short/truncated questions, e.g. "Data
        # cleaning" getting assigned "Normalization"). Falls back to the
        # question's own words rather than the LLM's possibly-wrong guess.
        # Skipped for genuine follow-ups ("what about its types", "how many
        # times has that appeared") — there the topic is SUPPOSED to come
        # from conversation context, not this message's own words, so it
        # will never share words with a carried-over topic by design. This
        # guard wrongly overrode "Hashing" on exactly such a follow-up
        # before — caught by eval_questions.py's chain_followup_kept case.
        if not sq.is_followup and not _topic_matches_query(sq.canonical_topic, question):
            fallback = _clean_fallback_topic(question)
            if fallback:
                sq.canonical_topic = fallback
                sq.search_queries = [fallback] + [q for q in sq.search_queries if q.lower() != (sq.canonical_topic or "").lower()]

        return sq

    @staticmethod
    def _clean_json(raw: str) -> str:
        """Strips markdown code fences and isolates JSON payload."""
        text = raw.strip()
        text = re.sub(r'^```(json)?', '', text, flags=re.MULTILINE).strip()
        text = re.sub(r'```$', '', text, flags=re.MULTILINE).strip()
        start = text.find('{')
        if start == -1:
            return text
        try:
            _, end = json.JSONDecoder().raw_decode(text, start)
            return text[start:end]
        except json.JSONDecodeError:
            end = text.rfind('}')
            return text[start:end + 1] if end > start else text

    def _build_structured_query(self, raw_query: str, data: dict, context_topic: Optional[str], subject_context: Optional[str]) -> StructuredQuery:
        canonical_topic = str(data.get("canonical_topic", "") or "").strip()
        if not canonical_topic and context_topic:
            canonical_topic = context_topic

        raw_subtopics = data.get("subtopics", [])
        subtopics = [str(st).strip() for st in raw_subtopics if str(st).strip()] if isinstance(raw_subtopics, list) else []

        raw_intents = data.get("intents", [])
        if isinstance(raw_intents, list):
            intents = [i.strip().lower() for i in raw_intents if i.strip().lower() in ALL_INTENTS]
        elif isinstance(raw_intents, str) and raw_intents.strip().lower() in ALL_INTENTS:
            intents = [raw_intents.strip().lower()]
        else:
            intents = []

        if not intents:
            intents = ["definition"]

        # "show/list ... questions", "past/previous/exact questions" -> exact_questions intent
        if re.search(r'\b(show|list|give|display|see)\b.*\bquestions?\b|\b(previous|past|exact)\s+(year\s+)?questions?\b', raw_query, re.I):
            if "exact_questions" not in intents:
                intents.append("exact_questions")
            if not re.search(r'\b(explain|define|what is|describe|meaning)\b', raw_query, re.I):
                intents = [i for i in intents if i != "definition"] or ["exact_questions"]

        # Search queries
        raw_sq = data.get("search_queries", [])
        search_queries = [str(sq).strip() for sq in raw_sq if str(sq).strip()] if isinstance(raw_sq, list) else []
        if not search_queries:
            search_queries = [canonical_topic] if canonical_topic else [raw_query]
            if subtopics:
                search_queries.extend([f"{canonical_topic} {st}" for st in subtopics])

        scope = str(data.get("question_scope", "related")).lower()
        if scope not in ("exact", "subtopic", "related"):
            scope = "related"

        # Explicit marks extraction (from JSON or regex e.g. "for 10 marks")
        marks = data.get("marks")
        if marks is not None:
            try:
                marks = int(marks)
            except (ValueError, TypeError):
                marks = None
        if marks is None:
            m = re.search(r'\b(2|5|8|10)\s*marks?\b', raw_query.lower())
            if m:
                marks = int(m.group(1))

        # Year constraint
        year_constraint = data.get("year_constraint")
        if not year_constraint:
            year_match = re.search(r'\b(19|20)\d{2}\b', raw_query)
            if year_match:
                year_constraint = year_match.group(0)

        subject = data.get("subject") or subject_context
        is_followup = bool(data.get("is_followup", False))

        # Determine target source requirements based on detected intents
        needs_notes = any(i in intents for i in ("definition", "importance", "lab_program", "syllabus"))
        needs_qb = any(i in intents for i in ("frequency", "exact_questions", "years", "importance", "trend", "ranking", "similar_questions"))
        needs_syllabus = "syllabus" in intents or "importance" in intents
        needs_lab = "lab_program" in intents

        return StructuredQuery(
            raw_query=raw_query,
            canonical_topic=canonical_topic,
            subtopics=subtopics,
            intents=intents,
            search_queries=search_queries,
            question_scope=scope,
            subject=subject,
            marks=marks,
            year_constraint=str(year_constraint) if year_constraint else None,
            is_followup=is_followup,
            resolved_reference=context_topic if is_followup else None,
            needs_notes=needs_notes,
            needs_question_bank=needs_qb,
            needs_syllabus=needs_syllabus,
            needs_lab=needs_lab,
        )

    def _heuristic_fallback(self, question: str, context_topic: Optional[str], subject_context: Optional[str]) -> StructuredQuery:
        """Pure python fallback when LLM is unavailable."""
        q_lower = question.lower()
        intents = []

        if any(w in q_lower for w in ["what is", "define", "explain", "describe", "meaning", "how does"]):
            intents.append("definition")
        if any(w in q_lower for w in ["how many times", "frequency", "how often", "repeated"]):
            intents.append("frequency")
        if any(w in q_lower for w in ["exact question", "exact questions", "show questions", "what questions", "previous questions", "past questions"]):
            intents.append("exact_questions")
        if re.search(r'\byears?\b', q_lower):
            intents.append("years")
        if any(w in q_lower for w in ["should i study", "should i read", "is it important", "pass the exam", "priority", "importance"]):
            intents.append("importance")
        if any(w in q_lower for w in [
            "trend", "rising", "falling", "increasing", "over years", "over the years",
            "every year", "each year", "only sometimes", "consistently", "regularly asked",
            "how often", "getting more common", "getting less common", "still relevant",
            "still asked", "still being asked", "declining", "recurring",
        ]):
            intents.append("trend")
        if any(w in q_lower for w in ["most frequent", "top question", "top topics", "ranking"]):
            intents.append("ranking")
        if any(w in q_lower for w in ["program", "code for", "implement"]):
            intents.append("lab_program")
        if any(w in q_lower for w in ["syllabus", "which unit", "module"]):
            intents.append("syllabus")

        if not intents:
            intents = ["definition"]

        # Topic detection
        topic = _clean_fallback_topic(question)
        is_followup = False
        if (not topic or len(topic) < 3 or topic.lower() in ("it", "this", "that", "questions", "question", "topic")) and context_topic:
            topic = context_topic
            is_followup = True

        subtopics = []
        if "linear" in q_lower and "circular" in q_lower:
            subtopics = ["Linear Queue", "Circular Queue"]
            topic = re.sub(r'\b(linear|circular)\b', '', topic, flags=re.IGNORECASE).strip()
            topic = re.sub(r'\s+', ' ', topic)
            if not topic:
                topic = "Queue"

        search_queries = [topic] if topic else [question]
        if subtopics:
            search_queries.extend([f"{topic} {st}" for st in subtopics])

        marks_match = re.search(r'\b(2|5|8|10)\s*marks?\b', q_lower)
        marks = int(marks_match.group(1)) if marks_match else None

        needs_notes = any(i in intents for i in ("definition", "importance", "lab_program", "syllabus"))
        needs_qb = any(i in intents for i in ("frequency", "exact_questions", "years", "importance", "trend", "ranking"))

        return StructuredQuery(
            raw_query=question,
            canonical_topic=topic,
            subtopics=subtopics,
            intents=intents,
            search_queries=search_queries,
            question_scope="subtopic" if subtopics else "related",
            subject=subject_context,
            marks=marks,
            year_constraint=None,
            is_followup=is_followup,
            resolved_reference=context_topic if is_followup else None,
            needs_notes=needs_notes,
            needs_question_bank=needs_qb,
            needs_syllabus="syllabus" in intents or "importance" in intents,
            needs_lab="lab_program" in intents,
        )