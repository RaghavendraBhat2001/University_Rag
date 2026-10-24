# src/document_loader.py

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from config import NOTES_DIR, PAPERS_DIR
import re
import io
import os
import json
from config import BASE_DIR

# OCR fallback deps — only needed for scanned PDFs with no real text layer
# (common in older question papers). Guarded so a missing PyMuPDF install
# degrades to "no OCR" instead of breaking ingestion entirely.
try:
    import fitz  # PyMuPDF
    import pytesseract
    from PIL import Image
    from config import TESSERACT_CMD
    if TESSERACT_CMD and os.path.exists(TESSERACT_CMD):
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False

# Below this many characters, PyPDFLoader's output is treated as "no real
# text" (covers both fully blank pages and stamp-only pages like "Library
# DSCASC") and OCR is attempted instead.
MIN_TEXT_LENGTH = 30
OCR_DPI = 300

# Phrases that match the numbered-line pattern but are instructions, not
# actual questions ("1. Answer any 4 from Section B.") — filtered out
_NON_QUESTION_PHRASES = (
    "answer any", "answer all", "answer the following", "note :",
    "note:", "instructions", "time :", "time:", "max. marks",
    "max marks", "each question carries", "section -",
    "section a", "section b", "part -", "part a", "part b",
    "cbcs scheme", "computer science", "p.t.o", "hours max",
)

_DASH_CHARS = re.compile(r"[\u2010-\u2015\u2212]")  # hyphen/en/em dash, minus sign


def _normalize_dashes(text: str) -> str:
    """Collapses every dash-like unicode character OCR might produce (en dash
    "–", em dash "—", minus sign "−", ...) to a plain "-", so phrase checks
    against _NON_QUESTION_PHRASES only need one canonical form. Without this,
    "Part — B : Answer any four questions" (em dash) silently failed to match
    the "part -" entry (plain hyphen) and got kept as a fake "question" —
    a real, confirmed bug from a Question Paper tab upload."""
    return _DASH_CHARS.sub("-", text)


def _is_non_question(text: str) -> bool:
    return any(_normalize_dashes(text).lower().startswith(p) for p in _NON_QUESTION_PHRASES)


# "Part—A: Answer any five..." / "Part—B: Answer any four..." — OCR renders the dash with NO
# surrounding spaces, which _NON_QUESTION_PHRASES's "part -" (space before the dash) never
# matched. A dedicated regex, independent of exact spacing, catches this reliably.
_SECTION_HEADER_RE = re.compile(
    r"^\s*part\s*[-\u2013\u2014:]?\s*[a-z]\b|"
    r"^\s*section\s*[-\u2013\u2014:]?\s*[a-z]\b",
    re.IGNORECASE,
)


def _is_section_header(text: str) -> bool:
    return bool(_SECTION_HEADER_RE.match(text))


# Administrative/cover-page phrases that should never appear inside a real
# exam QUESTION. Checked against the whole assembled candidate text (not
# per-line, since OCR line-breaks land unpredictably — a stray "22)" from
# "(2021 — 22)" can start a physical OCR line purely by accident of page
# layout and get mistaken for a new numbered question). Deliberately specific
# phrases only, to avoid rejecting a real question that happens to mention
# "computer" or "science" in passing.
_COVER_PAGE_MARKERS = re.compile(
    r"examination\s*,|\bcbcs\b|semester\s+m\.?\s*c\.?\s*a\.?|max\.?\s*marks\s*:|"
    r"time\s*:\s*\d|scanned\s+with|\(elective\)|two\s+years?\s+course|"
    r"\bmca\s*\d\s*e\s*\d|\bpg\s*[-\u2013\u2014]\s*\d+|\bill\s+semester\b|"
    r"\biii\s+semester\b|\bsemester\s+m\b",
    re.IGNORECASE,
)

# Page-turn/scan-batch watermark seen trailing a question across EVERY paper from this
# university ("P.T.O. JP – 763 *JP763*", "-2- *JP766*", or a bare "*JP766*"). A stray
# number immediately before it ("6 *JP766*") is often the question's real marks value,
# which the plain marks-extraction regex below could never reach since it requires the
# number to be the very last thing in the string — this watermark was always in the way.
# Footer markers seen on every page of these papers ("P.T.O.", "JP - 185", "*JP763*"). OCR also
# reads the barcode/header next to them as junk ("I]", "{ANY IO ~", "2. NUN", "pa ees ; 0"),
# so the marker is NOT the last thing in the text. If only a short tail follows the first
# marker it is all footer junk -> cut it. A long tail means the question continues onto the
# next page -> keep the text and only remove the markers themselves.
_PAGE_WATERMARK_RE = re.compile(r'(?:\bP\.\s*T\.\s*O\b\.?|\bJP\s*[-\u2013\u2014]?\s*\d{3}\b|\*JP\d{3}\*|-\s*\d{1,2}\s*-(?=\s|$))', re.IGNORECASE)
_WM_SHORT_TAIL = 80


def _strip_page_watermark(text):
    """Returns (text with the watermark removed, recovered marks value or None)."""
    m = _PAGE_WATERMARK_RE.search(text)
    if not m:
        return text, None
    if len(text) - m.end() <= _WM_SHORT_TAIL:
        head = text[:m.start()].rstrip()
    else:
        head = re.sub(r'\s+', ' ', _PAGE_WATERMARK_RE.sub(' ', text)).strip()
        return head, None
    mm = re.search(r'\s(\d{1,2})$', head)
    recovered = int(mm.group(1)) if mm else None
    if mm:
        head = head[:mm.start()].rstrip()
    return head, recovered

_WORD_TO_NUM = {
    'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'eight': 8, 'ten': 10,
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '8': 8, '10': 10
}


# --- Figure / diagram OCR noise -------------------------------------------------
# A question that says "for the following tree / graph / circuit" is usually followed by
# a drawing. OCR reads the drawing as junk tokens (e.g. 'mr" {" Vv', or vertex labels
# 'A B D CE F H GP Q S RT U W V') which end up glued to the end of the question text.
# Strip that trailing junk ONLY when all of these hold, so real formulas survive
# ("Simplify the following expression. AB + A'C" keeps its expression):
#   - the text before it contains a figure cue word (following/given/tree/graph/...)
#   - every trailing token has no 3-letter word, no digit and no operator/bracket symbol
_FIGURE_CUE = re.compile(r"\b(following|given|below|above|shown|figure|diagram|tree|graph|graphs|circuit|network|"
                         r"automaton|machine|matrix|table|flowchart)\b", re.IGNORECASE)
_MATHY = re.compile(r"[+=*^/<>'\u2019\u2229\u222a\u2227\u2228\u00ac\u2192\u2264\u2265()]")


def _strip_figure_noise(text):
    if not text:
        return text
    text = re.sub(r'\s+\.\s*$', '', text)   # a lone stray "." after the junk would hide the junk from the check below
    idx = max(text.rfind(c) for c in ".?!:")
    if idx < 15:
        return text
    head, tail = text[:idx + 1], text[idx + 1:].strip()
    # a bare closing quote/bracket after the final period is real text, not figure noise
    if not tail or not re.search(r"[A-Za-z]", tail) or not _FIGURE_CUE.search(head):
        return text
    toks = tail.split()
    # FIX: drawing junk like '\\ ®@ © & | pr," {* Vv ©® © oO .' -- symbols that never occur in a real
    # exam question (® © ™ @) and no real word. Stripped even though it contains '*' / ',' (_MATHY).
    if len(toks) <= 30 and re.search(r'[\u00ae\u00a9\u2122@\u2022\u25cf]', tail) \
            and not any(re.search(r'[A-Za-z]{3,}', tk) for tk in toks):
        return head.rstrip()
    if len(toks) > 25 or _MATHY.search(tail):
        return text
    if any(re.search(r"[A-Za-z]{3,}", tk) for tk in toks):
        return text
    stray_digits = [tk for tk in toks if re.fullmatch(r"\d{1,2}[.,;]?", tk)]
    if len(stray_digits) > 2 or any(re.search(r"\d", tk) for tk in toks if tk not in stray_digits):
        return text
    if sum(1 for tk in toks if re.search(r"[A-Za-z]", tk)) < 0.6 * len(toks):
        return text
    return head.rstrip()


_BARE_NUM_ONLY = re.compile(r'^\d{1,2}$')
_STARTS_LIKE_QUESTION = re.compile(r'^(?:\(?[a-d][\.\)]\s|[A-Z][a-z]{2,})')


_SUBPART_RE = re.compile(r'(?:^|\s)\(?([a-d])[\.\)]\s+(.+?)(?=\s\(?[a-d][\.\)]\s|\Z)')  # case-sensitive on purpose
# [patch_qp_fixes] '(2.5x4=10)' style marks formula that sits in the stem of a multi-part question
_MARKS_FORMULA_RE = re.compile(r'\(?\s*(\d+(?:\.\d+)?)\s*[×xX*]\s*(\d+)\s*=\s*\d+\s*\)?')



# FIX: on some scans Tesseract returns a run of bare question numbers ("9." "10." "11." ...)
# first and the question texts afterwards, un-numbered, in the same order. Pair them back up.
_BARE_NUM_LINE = re.compile(r'^[_~\s]*(\d{1,2})\s*[.;:]?\s*$')
_NUMBERED_WITH_TEXT = re.compile(r'^\s*(?:Q\.?\s*)?\d{1,2}\s*[.)]\s*\S')
_CONT_START = re.compile(r'^(?:[b-eB-E][.)]\s|\(?[b-e]\)|[ivx]{1,4}\)|(?!a[.)])[a-z]|\d|\()')


def _repair_orphan_numbers(page_str):
    lines = page_str.split('\n')
    nums, run_start, run_end = [], None, None
    for i, l in enumerate(lines):
        if not l.strip():
            continue
        m = _BARE_NUM_LINE.match(l)
        if m:
            if run_start is None:
                run_start = i
            nums.append(int(m.group(1)))
            run_end = i
        elif run_start is not None:
            if len(nums) >= 2:
                break
            nums, run_start, run_end = [], None, None   # lone stray digit in header junk, keep looking
    if len(nums) < 2 or nums != list(range(nums[0], nums[0] + len(nums))):
        return page_str
    blocks, cur, prev_blank = [], [], True
    for l in lines[run_end + 1:]:
        if not l.strip():
            prev_blank = True
            continue
        if _NUMBERED_WITH_TEXT.match(l):
            break
        _ls = l.strip()
        _prev_has_marks = bool(cur) and bool(re.search(r'\s\d{1,2}\s*\W*$', cur[-1]))
        _is_a = bool(re.match(r'^a[.)]\s', _ls))
        _cur_has_a = any(re.match(r'^a[.)]\s', c) for c in cur)
        _prev_junk = bool(cur) and not re.search(r'[A-Za-z]{3,}', cur[-1])
        _stem_open = bool(cur) and cur[-1].rstrip().endswith(':') and not _cur_has_a
        if _is_a and _stem_open:
            pass                                    # "Write notes on :" + blank + "a) ..." stays together
        elif _is_a and not _cur_has_a and _prev_junk:
            blocks.append(cur); cur = []            # figure junk line(s) separate the stem from the next question
        elif cur and not _CONT_START.match(_ls) and (prev_blank or _prev_has_marks):
            blocks.append(cur); cur = []
        elif cur and _is_a and _cur_has_a:
            blocks.append(cur); cur = []
        cur.append(_ls); prev_blank = False
    if cur:
        blocks.append(cur)
    blocks = [b for b in blocks if len(' '.join(b)) >= 12]
    if len(blocks) < len(nums):
        return page_str
    out = lines[:run_start]
    for n, b in zip(nums, blocks):
        out.append(f"{n}. " + ' '.join(b))
    for b in blocks[len(nums):]:
        out.extend(b)
    # keep any numbered remainder (e.g. "13. Find MSP ...")
    rest_idx = None
    seen = 0
    for j in range(run_end + 1, len(lines)):
        if _NUMBERED_WITH_TEXT.match(lines[j]):
            rest_idx = j; break
    if rest_idx is not None:
        out.extend(lines[rest_idx:])
    return '\n'.join(out)


def extract_questions_from_text(page_texts):
    """Extracts cleanly segmented individual examination questions from per-page text strings.
    Accurately recognizes:
    - Section marks declarations (e.g. 'Each question carries six marks', 'carries 10 marks', '(5x6=30)')
    - Top-level question boundaries (Q1., 1., 2.) without bleeding into subsequent questions or cover headers
    - Subparts (a), b)) with individual marks annotation
    - Page number recorded per question (kept in the data model; no
      longer used for diagram association since that feature was removed)
    """
    if isinstance(page_texts, str):
        raise TypeError("extract_questions_from_text() expects a LIST of per-page strings, not a joined string.")

    questions = []
    current_section_marks = 5
    q_blocks = []
    cur_q_num = None
    cur_lines = []
    cur_start_page = None
    last_num = 0

    for page_idx, page_str in enumerate(page_texts):
        page_str = _repair_orphan_numbers(page_str)
        cleaned = re.sub(r'[\t\r]+', ' ', page_str)
        lines = cleaned.split('\n')

        # FIX: a page that opens with a cover header (Examination ... / Max. Marks) is a NEW paper.
        # Close the question still open from the previous page, otherwise the cover text gets glued
        # onto it and the cover-page filter later throws the whole (real) question away.
        _head = ' '.join(page_str.split())[:700]
        if re.search(r'examin\w*\s*,?\s*[-\u2013\u2014]?\s*(?:[A-Za-z./]+\s*){0,3}?20[12]\d|max\.?\s*marks\s*:?\s*\d+', _head, re.IGNORECASE):
            if cur_q_num and cur_lines:
                q_blocks.append((cur_q_num, '\n'.join(cur_lines), current_section_marks, cur_start_page))
            cur_q_num = None; cur_lines = []; cur_start_page = None
            last_num = 0
            current_section_marks = 5

        in_instructions = False

        for li, line in enumerate(lines):
            line_str = line.strip()
            if not line_str:
                continue

            # A lone number with no dot/bracket ("2" on its own line, then "a) Perform ..." or
            # a capitalised sentence) is a question start whose punctuation OCR dropped. Guarded:
            # it must be the next number in sequence (or 1) AND be followed by question-like text,
            # so a stray marks digit or page number is not promoted to a question.
            if _BARE_NUM_ONLY.match(line_str):
                _n = int(line_str)
                _nxt = next((l.strip() for l in lines[li + 1:li + 4] if l.strip()), "")
                if ((_n == last_num + 1 or _n == 1) and _STARTS_LIKE_QUESTION.match(_nxt)
                        and not _is_non_question(_nxt) and not _is_section_header(_nxt)):
                    line_str = f"{_n}."

            # FIX: marks declarations often sit INSIDE the "Note : ... Each question carries 10 marks"
            # line, which the instruction-block `continue` below used to skip entirely.
            _m1 = re.search(r'each\s+question\s+carries\s+(\w+)\s+marks?', line_str, re.IGNORECASE)
            if _m1 and _m1.group(1).lower() in _WORD_TO_NUM:
                current_section_marks = _WORD_TO_NUM[_m1.group(1).lower()]
            _m2 = re.search(r'\(\s*\d+\s*[xX*]\s*(\d+)\s*=\s*\d+\s*\)', line_str)
            if _m2:
                current_section_marks = int(_m2.group(1))

            # Detect instruction blocks to avoid capturing "1) Answer any 5" as Question 1
            if re.match(r'^(?:Instructions|Note\s*:)', line_str, re.IGNORECASE):
                in_instructions = True
                continue
            if in_instructions:
                if re.match(r'^(?:SECTION|PART)', line_str, re.IGNORECASE):
                    in_instructions = False
                elif re.match(r'^\d+[\.\)]\s*(?:Answer|Note)', line_str, re.IGNORECASE):
                    continue

            # Check section marks headers
            m_sec = re.search(r'each\s+question\s+carries\s+(\w+)\s+marks?', line_str, re.IGNORECASE)
            if m_sec:
                w = m_sec.group(1).lower()
                if w in _WORD_TO_NUM:
                    current_section_marks = _WORD_TO_NUM[w]
            m_sec2 = re.search(r'\(\d+\s*[xX*]\s*(\d+)\s*=\s*\d+\)', line_str)
            if m_sec2:
                current_section_marks = int(m_sec2.group(1))

            # Boundary indicator (Sections, Parts, Exam Sitting Headers, horizontal lines)
            is_boundary = bool(re.search(
                r'^(?:SECTION|PART|Instructions|\*PG|\bExamination\b|Time\s*:|Max\.\s*Marks|CBCS|_{3,}|-{3,})',
                line_str, re.IGNORECASE
            ))

            # FIX: OCR often turns "3." into "3-" / "5," / "_2-"; accept only when it is exactly the next number
            _ma = re.match(r'^[_~\s]*(\d{1,2})\s*(?:--?|,)\s*(?=[A-Za-z(])(.*)', line_str)
            if _ma and int(_ma.group(1)) == last_num + 1 and not re.match(r'^\d{1,2}[.)]', line_str):
                line_str = f"{_ma.group(1)}. {_ma.group(2)}"
            # FIX: OCR reads "11." as "41." / "43." -- accept the corrected number only if it is the next in sequence
            _mb = re.match(r'^(\d)(\d)([.)])\s*(.*)', line_str)
            if _mb and int(_mb.group(1) + _mb.group(2)) > 20 and last_num >= 8:
                _cand = int('1' + _mb.group(2))
                if _cand in (last_num + 1, last_num + 2):
                    line_str = f"{_cand}{_mb.group(3)} {_mb.group(4)}"
            # Question start: Q1. or 1. or 1)
            m_q = re.match(r'^(?:Q\.?\s*)?(\d{1,2})[\.\)]\s*(.*)', line_str, re.IGNORECASE)

            if m_q and (_is_non_question(m_q.group(2)) or _is_section_header(m_q.group(2))):
                # A numbered line that's actually an instruction/section-header
                # ("6. Part—A: Answer any five questions...") — not recognized by
                # is_boundary (that pattern expects an UNNUMBERED "PART"/"SECTION"
                # start, not a digit-prefixed one). Close out whatever question was
                # accumulating and discard this line entirely. Previously this fell
                # through to the plain "append as continuation" branch below, which
                # corrupted the PREVIOUS question's text with this rejected line's
                # content instead of just dropping it — found by testing against the
                # exact real failing case, not assumed.
                if cur_q_num and cur_lines:
                    q_blocks.append((cur_q_num, '\n'.join(cur_lines), current_section_marks, cur_start_page))
                cur_q_num = None
                cur_lines = []
                cur_start_page = None
            elif m_q and not re.match(r'[A-Za-z(\[{]', m_q.group(2).strip() or '#'):
                # FIX: "1. 2 3)" / "2) 10 11 12" inside a matrix/figure is OCR noise, not a question
                if cur_q_num:
                    cur_lines.append(line_str)
            elif m_q:
                # Checked against m_q.group(2) (the text AFTER the leading number is
                # stripped), not line_str — line_str always starts with the matched
                # digit prefix here, so checking it directly against
                # _NON_QUESTION_PHRASES/_SECTION_HEADER_RE (which expect "part"/
                # "answer any" etc. at position 0) could never match.
                if cur_q_num and cur_lines:
                    q_blocks.append((cur_q_num, '\n'.join(cur_lines), current_section_marks, cur_start_page))
                cur_q_num = m_q.group(1)
                last_num = int(cur_q_num)
                cur_lines = [m_q.group(2)]
                cur_start_page = page_idx
            elif is_boundary:
                if cur_q_num and cur_lines:
                    q_blocks.append((cur_q_num, '\n'.join(cur_lines), current_section_marks, cur_start_page))
                    cur_q_num = None
                    cur_lines = []
                    cur_start_page = None
            else:
                if cur_q_num:
                    cur_lines.append(line_str)
                    # A question can legitimately continue onto the next
                    # page (multi-line/multi-part proof questions). Do NOT
                    # reset cur_q_num/cur_lines/q_blocks per page — only a
                    # real new question number or a genuine boundary line
                    # (checked above, regardless of page) ends a capture.
                    # cur_start_page stays the page the question OPENED on,
                    # not wherever it happens to be finalized, so diagram
                    # association still points at the right page.

    if cur_q_num and cur_lines:
        q_blocks.append((cur_q_num, '\n'.join(cur_lines), current_section_marks, cur_start_page))

    # Process captured question blocks
    for q_num, raw_q, sec_marks, p_idx in q_blocks:
        clean_text = re.sub(r'\s+', ' ', raw_q).strip()

        clean_text, _watermark_marks = _strip_page_watermark(clean_text)

        # Check for sub-parts like a) ... b) ...
        # Sub-part markers must be a standalone lowercase letter (optionally in parens) that
        # starts the string or follows whitespace, and is followed by whitespace. The old
        # pattern also matched word endings ("DFA.", "Java.", "(B U C)"), which truncated
        # questions to "...function of DF" / "arrays in Jav".
        subparts = _SUBPART_RE.findall(clean_text)
        _first_sub = _SUBPART_RE.search(clean_text)
        stem = clean_text[:_first_sub.start()].strip()[:200] if _first_sub else ""
        _per_sub = None
        _mf = _MARKS_FORMULA_RE.search(stem)
        if _mf:
            _per_sub = float(_mf.group(1))
            stem = _MARKS_FORMULA_RE.sub('', stem).strip(' :-')
        if subparts and len(subparts) > 1 and subparts[0][0] == 'a':
            for sub_letter, sub_text in subparts:
                sub_clean = re.sub(r'\s+', ' ', sub_text).strip()
                m_marks = re.search(r'(?:[\(\[{]?(\d{1,2})\s*marks?[\)\]}]?|\s+(\d{1,2})\s*)$', sub_clean, re.IGNORECASE)
                # Rounded rather than floored (7 marks / 2 subparts -> 4, not 3 -- closer to the
                # real 3.5/3.5 split), and never below 2: a genuine long-form sub-question is
                # essentially never worth just 1 mark on these papers, so a value that low is
                # far more likely a parsing artifact than a real one.
                marks = max(2, int((_per_sub if _per_sub else sec_marks / len(subparts)) + 0.5))
                if m_marks:
                    val = m_marks.group(1) or m_marks.group(2)
                    if val and 2 <= int(val) <= 20:
                        marks = int(val)
                        sub_clean = sub_clean[:m_marks.start()].strip()
                sub_clean = re.sub(r'[_–\-\s]+$', '', sub_clean).strip()
                sub_clean = _strip_figure_noise(sub_clean)
                if len(sub_clean) >= 12 and not _is_non_question(sub_clean) and not _COVER_PAGE_MARKERS.search(sub_clean):
                    # Keep the parent question's stem ("Write short notes on:", "Create a class Book...")
                    # so each sub-part is answerable on its own.
                    full_text = f"{stem} {sub_clean}".strip() if stem else sub_clean
                    questions.append({'number': f"{q_num}({sub_letter})", 'text': full_text, 'marks': marks, 'page': p_idx})
        else:
            m_end_marks = re.search(r'(?:[\(\[{]?(\d{1,2})\s*marks?[\)\]}]?|\s+(\d{1,2})\s*)$', clean_text, re.IGNORECASE)
            marks = max(2, sec_marks)
            if m_end_marks:
                val = m_end_marks.group(1) or m_end_marks.group(2)
                if val and 2 <= int(val) <= 20:
                    marks = int(val)
                    clean_text = clean_text[:m_end_marks.start()].strip()
            elif _watermark_marks is not None and 2 <= _watermark_marks <= 20:
                marks = _watermark_marks
            clean_text = re.sub(r'[_–\-\s]+$', '', clean_text).strip()
            clean_text = _strip_figure_noise(clean_text)
            if len(clean_text) >= 12 and not _is_non_question(clean_text) and not _COVER_PAGE_MARKERS.search(clean_text):
                questions.append({'number': q_num, 'text': clean_text, 'marks': marks, 'page': p_idx})

    return questions


# A correctly-ordered single-column page essentially never has 2+ consecutive
# non-blank lines that are JUST a bare question number with nothing else on the
# line. That exact signature shows up when a two-column exam layout gets OCR'd
# in the wrong order: the page's automatic layout analysis misjudges the columns
# and reads straight across both of them, pulling out every column-1 question
# NUMBER first (with nothing else on that line, since its real text sits in
# column 2) followed by a separate block of unattributed content. Confirmed
# against a real failing page before writing this, not assumed.
_BARE_QUESTION_NUMBER_RE = re.compile(r'^(?:Q\.?\s*)?\d{1,2}[\.\)]\s*$')


def _has_scrambled_column_signature(text):
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    consecutive = 0
    for line in lines:
        if _BARE_QUESTION_NUMBER_RE.match(line):
            consecutive += 1
            if consecutive >= 2:
                return True
        else:
            consecutive = 0
    return False


def ocr_page_two_column(pdf_path, page_number, dpi=OCR_DPI):
    """Fallback for a page whose normal OCR came back with the scrambled-column
    signature above: splits the rendered page image at the horizontal midpoint
    and OCRs each half separately, left column first then right, each read
    top-to-bottom — the correct reading order for a standard two-column exam
    layout. Only called as a retry, never as the default path, since most pages
    are single-column and don't need this."""
    if not _OCR_AVAILABLE:
        return ""
    try:
        doc = fitz.open(str(pdf_path))
        page = doc[page_number]
        pix = page.get_pixmap(dpi=dpi)
        image = Image.open(io.BytesIO(pix.tobytes("png")))
        width, height = image.size
        left_half = image.crop((0, 0, width // 2, height))
        right_half = image.crop((width // 2, 0, width, height))
        left_text = pytesseract.image_to_string(left_half).strip()
        right_text = pytesseract.image_to_string(right_half).strip()
        doc.close()
        return (left_text + "\n" + right_text).strip()
    except Exception as e:
        print(f"      ⚠️ Two-column OCR retry failed on page {page_number} of {pdf_path.name}: {e}")
        return ""


def ocr_page(pdf_path, page_number, dpi=OCR_DPI, psm=None):
    """Render a single PDF page to an image and OCR it. Used as a fallback
    for scanned question papers where PyPDFLoader finds no real text layer."""
    if not _OCR_AVAILABLE:
        return ""
    try:
        doc = fitz.open(str(pdf_path))
        page = doc[page_number]
        pix = page.get_pixmap(dpi=dpi)
        image = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(image, config=f"--psm {psm}") if psm else pytesseract.image_to_string(image)
        doc.close()
        return text.strip()
    except Exception as e:
        print(f"      ⚠️ OCR failed on page {page_number} of {pdf_path.name}: {e}")
        return ""


def _safe_dirname(name):
    """Sanitize a subject name for use as a filesystem folder name — same
    logic as RAGPipeline._safe_dirname, duplicated locally to avoid a
    circular import (rag_pipeline.py already imports from this module)."""
    cleaned = re.sub(r'[^A-Za-z0-9_\-]', '_', name)[:80]
    return cleaned or "UNKNOWN"


def load_documents():
    """Load notes and question papers from separate folders"""
    documents = []
    
    # ============================================================
    # 1. Load NOTES
    # ============================================================
    if NOTES_DIR.exists():
        print(f"\n📚 Loading Notes from: {NOTES_DIR}")
        for pdf in NOTES_DIR.glob("**/*.pdf"):
            docs = load_pdf(pdf, "notes")
            documents.extend(docs)
    else:
        print(f"⚠️ Notes folder not found. Creating: {NOTES_DIR}")
        NOTES_DIR.mkdir(parents=True, exist_ok=True)
        print(f"📁 Please add your notes to: {NOTES_DIR}")
    
    # ============================================================
    # 2. Load QUESTION PAPERS
    # ============================================================
    if PAPERS_DIR.exists():
        print(f"\n📄 Loading Question Papers from: {PAPERS_DIR}")
        for pdf in PAPERS_DIR.glob("**/*.pdf"):
            docs = load_pdf(pdf, "question_paper")
            documents.extend(docs)
    else:
        print(f"⚠️ Question papers folder not found. Creating: {PAPERS_DIR}")
        PAPERS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"📁 Please add question papers to: {PAPERS_DIR}")
    
    # ============================================================
    # 3. SUMMARY
    # ============================================================
    total_notes = sum(1 for d in documents if d.metadata.get("type") == "notes")
    total_papers = sum(1 for d in documents if d.metadata.get("type") == "question_paper")
    
    print(f"\n📊 Summary:")
    print(f"   📚 Notes loaded: {total_notes} pages")
    print(f"   📄 Question papers: {total_papers} pages")
    print(f"   📦 Total: {len(documents)} pages")
    
    return documents


_SEP = r'[\s_-]*'  # separators seen across your folders: space, underscore, dash


def _is_semester_only_folder(name):
    """True for folder names that only encode a semester, with no subject
    info — e.g. 'SEM3', 'Sem-3', '1 Semester', '2nd Semester', '1st_Semester',
    '2nd_Sem'. Your data uses several different conventions for this across
    notes vs question papers, so all separator styles are checked."""
    if re.match(rf'^sem(?:ester)?{_SEP}\d+$', name, re.IGNORECASE):
        return True
    if re.match(rf'^\d+{_SEP}(?:st|nd|rd|th)?{_SEP}sem(?:ester)?$', name, re.IGNORECASE):
        return True
    return False


def extract_subject(pdf_path):
    """Find a (SUBJECT) style abbreviation anywhere in the path, e.g.
    .../SEM3/(DBMS)/unit2.pdf -> 'DBMS', or a filename like
    '(OOP Lab) Object Oriented Programming Lab.pdf' -> 'OOP LAB'. Falls
    back to a bare folder name if no parenthesized abbreviation is found —
    but never falls back to a semester-only folder name, since that isn't
    a subject."""
    for part in pdf_path.parts:
        match = re.search(r'\(([A-Za-z0-9&\-\s]+)\)', part)
        if match:
            return re.sub(r'\s+', ' ', match.group(1).strip()).upper()
    parent_name = pdf_path.parent.name
    if parent_name and not _is_semester_only_folder(parent_name):
        return parent_name.upper()
    return "UNKNOWN"


def build_subject_full_names(notes_dir, papers_dir):
    """Scans notes and question-paper folder/file names for the
    '(ABBR) Full Descriptive Name' pattern and returns {ABBR: full_name}.

    Used to expand bare-abbreviation queries (e.g. 'What is ML?') into
    something like 'What is ML (Machine Learning)?' before retrieval — a
    2-3 letter query is a very weak embedding-search signal on its own,
    and this gives it something real to match against, without any LLM
    call or guessing involved.
    """
    registry = {}
    for base_dir in (notes_dir, papers_dir):
        if not base_dir.exists():
            continue
        for path in base_dir.glob("**/*"):
            name = path.stem if path.is_file() else path.name
            match = re.search(r'\(([A-Za-z0-9&\-\s]+)\)\s*(.*)', name)
            if not match:
                continue
            abbr = re.sub(r'\s+', ' ', match.group(1).strip()).upper()
            full_name = match.group(2).strip()
            if full_name and abbr not in registry:
                registry[abbr] = full_name
    return registry


def extract_semester(pdf_path):
    """Search every part of the path for a semester marker, across all the
    separator styles seen in your data: 'SEM3'/'Sem-3', '1 Semester'/
    '2nd Semester', '1st_Semester'/'2nd_Sem'."""
    for part in pdf_path.parts:
        match = re.search(rf'sem(?:ester)?{_SEP}(\d+)', part, re.IGNORECASE)
        if match:
            return match.group(1)
        match = re.search(rf'(\d+){_SEP}(?:st|nd|rd|th)?{_SEP}sem(?:ester)?', part, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def extract_year_from_page_text(text):
    """Looks for the exam sitting's year, trying two strategies in order:

    1. Right after 'Examination' (tolerating common OCR misreadings of that
       word, e.g. 'Examinatton', 'Egeatination', 'Degree nation'), allowing
       for the month(s) in between ('Aug./Sept.' or 'June/July').
    2. A bare month-name (or month/month) immediately followed by a year,
       e.g. 'Oct./Nov. 2020', 'May/June 2025' — covers pages where the word
       'Examination' itself didn't OCR cleanly enough for strategy 1.

    Deliberately does NOT fall back to the '(20XX — 20XX and Onwards)'
    scheme-boilerplate phrase some papers also print — tested against a
    page with no real sitting header at all, and that phrase alone wrongly
    produces a year every time, which is exactly the false positive this
    function was built to avoid in the first place.
    """
    normalized = re.sub(r'\s+', ' ', text)

    m1 = re.search(
        r'(?:exam[a-z]*|egeatination|degree\s+nation)\s*,?\s*[-\u2013\u2014]?\s*'
        r'(?:[A-Za-z./–\-]+\s*){0,3}?(\b20[12]\d\b)',
        normalized, re.IGNORECASE,
    )
    if m1 and 2018 <= int(m1.group(1)) <= 2030:
        return m1.group(1)

    m2 = re.search(
        r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|'
        r'Sept?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
        r'(?:\s*[\/–\-]\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z.]*)?,?\s*(\b20[12]\d\b)',
        normalized, re.IGNORECASE,
    )
    if m2 and 2018 <= int(m2.group(1)) <= 2030:
        return m2.group(1)

    return None


def _apply_year_overrides(pdf_name, docs):
    """Manual page->year fixes for pages whose printed header OCR can't recover."""
    f = BASE_DIR / "year_overrides.json"
    if not f.exists():
        return
    try:
        with open(f, "r", encoding="utf-8") as fh:
            overrides = json.load(fh).get(pdf_name, {})
    except Exception:
        return
    for page_str, year in overrides.items():
        i = int(page_str)
        if 0 <= i < len(docs):
            docs[i].metadata["year"] = str(year)


def load_pdf(pdf_path, doc_type):
    """Load a single PDF with metadata. Falls back to OCR per-page for
    scanned PDFs where PyPDFLoader can't find a real text layer."""
    try:
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()

        subject = extract_subject(pdf_path)
        semester = extract_semester(pdf_path)
        filename_year_match = re.search(r'\b(19|20)\d{2}\b', pdf_path.name)
        filename_year = filename_year_match.group(0) if filename_year_match else None

        ocr_pages = 0
        blank_pages = 0
        column_fix_pages = 0
        current_page_year = None   # carried forward across THIS pdf's pages
        any_page_year_found = False

        for i, doc in enumerate(docs):
            if len(doc.page_content.strip()) < MIN_TEXT_LENGTH:
                blank_pages += 1
                ocr_text = ocr_page(pdf_path, i, psm=4 if doc_type == "question_paper" else None)
                if len(ocr_text) > len(doc.page_content.strip()):
                    doc.page_content = ocr_text
                    ocr_pages += 1

            # Question papers only (this never affects notes chunking/embedding):
            # a page whose text shows the scrambled-column signature gets ONE retry
            # with column-aware OCR. Only kept if the retry demonstrably fixes the
            # signature and isn't suspiciously short (a bad crop) — never silently
            # trusted over the original, since this can't be exhaustively tested
            # against every possible page layout.
            if doc_type == "question_paper" and _has_scrambled_column_signature(doc.page_content):
                # These papers are single-column: the 'scramble' is Tesseract's default page
                # layout mode pulling the question numbers out of the margin as a separate
                # block. Re-reading the whole page in a single-column mode (psm 4, then 6)
                # keeps each number with its text. The old half-page split is only the last
                # resort, because it cuts every line in half.
                retry_text = ""
                for _psm in (4, 6):
                    _cand = ocr_page(pdf_path, i, psm=_psm)
                    if (_cand
                            and not _has_scrambled_column_signature(_cand)
                            and len(_cand) > len(doc.page_content) * 0.5):
                        retry_text = _cand
                        break
                if not retry_text:
                    _cand = ocr_page_two_column(pdf_path, i)
                    if (_cand
                            and not _has_scrambled_column_signature(_cand)
                            and len(_cand) > len(doc.page_content) * 0.5):
                        retry_text = _cand
                if retry_text:
                    doc.page_content = retry_text
                    column_fix_pages += 1
                else:
                    print(f"      ⚠️ Page {i} of {pdf_path.name} looks like a scrambled two-column OCR read — "
                          f"automatic column-split retry didn't resolve it. May need manual review.")

            doc.metadata["source"] = pdf_path.name
            doc.metadata["type"] = doc_type
            doc.metadata["subject"] = subject
            doc.metadata["semester"] = semester

            if doc_type == "question_paper":
                # Question papers can merge several exam sittings (several
                # years) into one PDF, so the year has to come from each
                # page's own text, not the filename — carry the most
                # recently seen sitting year forward across continuation
                # pages (Part-B etc.) that don't repeat the header.
                page_year = extract_year_from_page_text(doc.page_content)
                if page_year:
                    current_page_year = page_year
                    any_page_year_found = True
                elif re.search(r"Max\.?\s*Marks\s*:?\s*\d+", doc.page_content[:1500], re.IGNORECASE):
                    # First page of a NEW sitting whose year line did not OCR: do NOT inherit
                    # the previous sitting's year (that silently merges two sittings).
                    current_page_year = None
                doc.metadata["year"] = current_page_year
            else:
                doc.metadata["year"] = filename_year

        if doc_type == "question_paper" and not any_page_year_found and filename_year:
            # No sitting-header year found anywhere in this file (e.g. an
            # older single-year paper whose scan quality didn't OCR
            # cleanly) — fall back to a year in the filename if present.
            for doc in docs:
                doc.metadata["year"] = filename_year

        if doc_type == "question_paper":
            _apply_year_overrides(pdf_path.name, docs)

        label = "📚 Notes" if doc_type == "notes" else "📄 Question Paper"
        ocr_note = ""
        if blank_pages and not _OCR_AVAILABLE:
            ocr_note = f" ⚠️ {blank_pages} blank page(s) found but PyMuPDF not installed — run: pip install pymupdf"
        elif ocr_pages:
            ocr_note = f", OCR'd {ocr_pages}/{blank_pages} blank page(s)"
        if column_fix_pages:
            ocr_note += f", fixed {column_fix_pages} scrambled-column page(s)"
        print(f"   ✅ {label}: {pdf_path.name} [{subject}, Sem {semester}]{ocr_note} ({len(docs)} pages)")
        return docs

    except Exception as e:
        print(f"   ❌ Error loading {pdf_path.name}: {e}")
        return []