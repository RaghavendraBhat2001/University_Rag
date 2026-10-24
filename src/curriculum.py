# src/curriculum.py
"""
Answers basic programme questions ("how many subjects in 1st sem", "what are the subjects in
3rd sem", "which lab in 2nd sem") WITHOUT the language model.

The answer is read from the folder structure that is already there:
    data/<university>/notes/SEM<n>/(ABBR) Full Subject Name/...
    data/<university>/question_papers/SEM<n>/(ABBR) Full Subject Name.pdf
so it is exact for what is indexed, instant, and can never be invented.
"""
import re
from pathlib import Path
from typing import Dict, List, Optional

_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6}
_ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6}

_LAB_RE = re.compile(r"\blabs?\b|\blaborator(?:y|ies)\b|\bpracticals?\b", re.IGNORECASE)
_EXAM_WORDS = re.compile(
    r"\b(appear\w*|asked|frequen\w*|times|exam\w*|questions?|marks|years?|repeat\w*|"
    r"important|importance|trend\w*|syllabus|unit|module|topic)\b", re.IGNORECASE)
_CODE_WORDS = re.compile(r"\b(write|program|code|algorithm|implement\w*|explain|define|what is)\b", re.IGNORECASE)
_ASKING = re.compile(
    r"\b(how many|which|what are|what is the|what all|list|name|tell me|show|number of|total|count)\b",
    re.IGNORECASE)
_SUBJECTS = re.compile(r"\b(subjects?|courses?)\b", re.IGNORECASE)


# ------------------------------------------------------------------ scanning the folders
def scan_curriculum(notes_dir, papers_dir) -> Dict[int, List[dict]]:
    """{semester: [{'abbr', 'name', 'is_lab'}, ...]} from the notes / question-paper folders."""
    from src.document_loader import extract_subject, extract_semester, build_subject_full_names

    notes_dir, papers_dir = Path(notes_dir), Path(papers_dir)
    names = build_subject_full_names(notes_dir, papers_dir)
    found: Dict[int, Dict[str, dict]] = {}
    for base in (notes_dir, papers_dir):
        if not base.exists():
            continue
        for pdf in base.glob("**/*.pdf"):
            sem = extract_semester(pdf)
            abbr = extract_subject(pdf)
            if not sem or not abbr or abbr == "UNKNOWN":
                continue
            name = names.get(abbr, "")
            is_lab = bool(_LAB_RE.search(abbr) or _LAB_RE.search(name))
            found.setdefault(int(sem), {}).setdefault(abbr, {"abbr": abbr, "name": name, "is_lab": is_lab})
    return {s: sorted(d.values(), key=lambda x: (x["is_lab"], x["abbr"])) for s, d in sorted(found.items())}


# ------------------------------------------------------------------ understanding the question
def find_semester(question: str) -> Optional[int]:
    q = question.lower()
    m = re.search(r"\bsem(?:ester)?\s*[-:]?\s*(\d)\b", q)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(\d)[a-z]{0,2}\s*[- ]?\s*sem(?:ester)?\b", q)   # 1st sem, 2nd semester, "1s sem"
    if m:
        return int(m.group(1))
    m = re.search(r"\b(first|second|third|fourth|fifth|sixth)\s+sem(?:ester)?\b", q)
    if m:
        return _ORDINALS[m.group(1)]
    m = re.search(r"\b(vi|iv|v|iii|ii|i)\s*[- ]?\s*sem(?:ester)?\b", q)
    if m:
        return _ROMAN[m.group(1)]
    return None


def parse_curriculum_query(question: str) -> Optional[dict]:
    """Returns {'sem': int|None, 'labs': bool, 'count': bool} or None when this is not such a question."""
    q = question.strip().lower()
    if not q or _EXAM_WORDS.search(q) or _CODE_WORDS.search(q):
        return None
    wants_labs = bool(_LAB_RE.search(q))
    wants_subjects = bool(_SUBJECTS.search(q))
    if not (wants_labs or wants_subjects):
        return None
    sem = find_semester(q)
    whole_programme = bool(re.search(r"\bmca\b|\ball\s+(?:subjects|labs|semesters)\b|\bin total\b|\boverall\b", q))
    if sem is None and not whole_programme:
        return None
    is_asking = bool(_ASKING.search(q))
    # Must either have an asking phrase ("how many", "which", "list") or explicitly specify a semester/whole programme
    if not is_asking and sem is None and not (whole_programme and (wants_labs or wants_subjects)):
        return None
    return {"sem": sem, "labs": wants_labs, "count": bool(re.search(r"\bhow many\b|\bnumber of\b|\btotal\b|\bcount\b", q))}


# ------------------------------------------------------------------ the answer
def _line(item: dict) -> str:
    return f"- **{item['abbr']}** — {item['name']}" if item["name"] else f"- **{item['abbr']}**"


def build_answer(parsed: dict, curriculum: Dict[int, List[dict]], dept: str = "MCA") -> str:
    if not curriculum:
        return ("I couldn't find any semester-wise subject folders yet. "
                "Subjects are read from the notes / question-paper folders (e.g. SEM1/(DS) Data Structures).")
    sem, only_labs, count = parsed["sem"], parsed["labs"], parsed["count"]
    sems = [sem] if sem else sorted(curriculum)
    foot = ("\n\n*Based on the subjects present in your notes and question-paper folders — "
            "not an official scheme document.*")

    if sem is not None and sem not in curriculum:
        have = ", ".join(str(s) for s in sorted(curriculum))
        return f"I couldn't find any subjects for {dept} semester {sem}. Semesters available: {have}." + foot

    blocks = []
    total = 0
    for s in sems:
        items = curriculum.get(s, [])
        items = [i for i in items if i["is_lab"]] if only_labs else items
        total += len(items)
        label = "lab subject" if only_labs else "subject"
        plural = "" if len(items) == 1 else "s"
        if not items:
            blocks.append(f"### {dept} Semester {s}\nNo {label}s found in the indexed folders.")
            continue
        n_lab = sum(1 for i in items if i["is_lab"])
        extra = "" if only_labs or not n_lab else f" ({len(items) - n_lab} theory, {n_lab} lab)"
        head = (f"### {dept} Semester {s}\n**{len(items)} {label}{plural}**{extra}" if count
                else f"### {dept} Semester {s} — {len(items)} {label}{plural}{extra}")
        blocks.append(head + "\n" + "\n".join(_line(i) for i in items))

    title = "🧪 LAB SUBJECTS" if only_labs else "📚 SUBJECTS"
    out = f"## {title}\n\n" + "\n\n".join(blocks)
    if len(sems) > 1:
        out += f"\n\n**Total across {len(sems)} semesters: {total}**"
    return out + foot


def curriculum_answer(question: str, notes_dir, papers_dir, dept: str = "MCA") -> Optional[str]:
    """None when the question is not a programme-structure question."""
    parsed = parse_curriculum_query(question)
    if parsed is None:
        return None
    return build_answer(parsed, scan_curriculum(notes_dir, papers_dir), dept)