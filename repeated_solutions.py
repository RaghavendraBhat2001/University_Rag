# repeated_solutions.py -- put this next to app.py (project root).
#
# Ready-made solutions for the "Most Repeated Questions" feature (Exam Intelligence tab).
#
#   python repeated_solutions.py status          coverage table: repeated questions vs. solutions on file (fast)
#   python repeated_solutions.py generate        overnight run: answers every repeated question that has no solution yet
#   python repeated_solutions.py generate DS CN  same, only for these subjects
#   python repeated_solutions.py export          writes one .txt per subject (+ one combined) to data/repeated_solutions/
#
# Where solutions live (the app reads both, hand-written wins):
#   data/repeated_solutions/solutions_index.json      hand-written, created by `python build_solutions.py`
#   data/repeated_solutions/solutions_generated.json  created by `generate` (resumable: Ctrl+C, run again, it continues)
#
# `generate` asks each question exactly the way the app does (same marks, same subject, plain_answer_only),
# so every answer is grounded in the indexed notes and also lands in _answer_cache.json.
# Keep Ollama running, the laptop plugged in, and do NOT use the app while it runs.

import json
import os
import re
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
SOL_DIR = BASE / "data" / "repeated_solutions"
HAND_FILE = SOL_DIR / "solutions_index.json"
GEN_FILE = SOL_DIR / "solutions_generated.json"
DEFAULT_MARKS = 5      # DEFAULT_CHAT_MARKS in app.py
TOP_K = 12             # same top_k the app uses for "Find Most Repeated Questions"
MIN_ANSWER_CHARS = 150


# ----------------------------------------------------------------------------------------------
# 1. Solution store (imported by app.py; no heavy imports here)
# ----------------------------------------------------------------------------------------------
def _norm(q):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", "", (q or "").lower())).strip()


_store = {"sig": None, "data": {}}


def _load_all():
    """{SUBJECT: {normalised question: (original question, info)}}. Re-read only when a file changes."""
    sig = tuple(p.stat().st_mtime_ns if p.exists() else 0 for p in (HAND_FILE, GEN_FILE))
    if _store["sig"] == sig:
        return _store["data"]
    merged = {}
    for path in (GEN_FILE, HAND_FILE):                    # hand-written is read last, so it wins
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for subj, qs in raw.items():
            if not isinstance(qs, dict):
                continue
            bucket = merged.setdefault(subj.strip().upper(), {})
            for q, info in qs.items():
                if isinstance(info, dict) and info.get("answer"):
                    bucket[_norm(q)] = (q, info)
    _store["sig"], _store["data"] = sig, merged
    return merged


def lookup_solution(subject, question):
    """Ready-made answer text for this question, or None. Never raises."""
    try:
        data = _load_all()
        subj = (subject or "").strip().upper()
        buckets = [data[subj]] if subj in data else list(data.values())   # unknown/'All' -> search everything
        key = _norm(question)
        for b in buckets:
            if key in b:
                return _repair_escapes(b[key][1]["answer"])
        words = set(key.split())
        best, best_score = None, 0.0
        for b in buckets:
            for k, (_q, info) in b.items():
                w2 = set(k.split())
                score = len(words & w2) / max(len(words | w2), 1)
                if score > best_score:
                    best, best_score = info, score
        if best is not None and best_score >= 0.8:
            return _repair_escapes(best["answer"])
    except Exception:
        pass
    return None


# ----------------------------------------------------------------------------------------------
# 2. Turning an answer into clean text for the .txt / .pdf downloads
# ----------------------------------------------------------------------------------------------
# build_solutions.py keeps its answers in ordinary (non-raw) Python strings, so "\times" became TAB+"imes"
# and "\forall" became FORMFEED+"orall". Put the backslash back (only when a known LaTeX word follows).
_ESC_FIX = [
    ("\t", r"(imes|ext\{|heta|ilde|au\b|an\b|o\b|op\b|riangle)", "\\t"),
    ("\x0c", r"(orall|rac\{|lat\b|loor|lfloor)", "\\f"),
    ("\r", r"(ightarrow|ightharpoon|ho\b|angle|ceil)", "\\r"),
    ("\x08", r"(mod|eta\b|ar\{|oldsymbol|igcup|igcap|inom)", "\\b"),
]


def _repair_escapes(text):
    for ch, tails, rep in _ESC_FIX:
        if ch in text:
            text = re.sub(re.escape(ch) + tails, lambda m, r=rep: r + m.group(1), text)
    return text


_MATH_SIMPLE = [
    (r"\\pmod\s*\{?([^}\s$]*)\}?", r"mod \1"),
    (r"\\implies|\\Rightarrow|\\Longrightarrow", "=>"), (r"\\iff|\\Leftrightarrow", "<=>"),
    (r"\\leftarrow|\\gets", "<-"), (r"\\rightarrow|\\to(?![a-zA-Z])", "->"),
    (r"\\leq?(?![a-zA-Z])", "<="), (r"\\geq?(?![a-zA-Z])", ">="), (r"\\neq?(?![a-zA-Z])", "!="),
    (r"\\pm(?![a-zA-Z])", "+/-"), (r"\\times|\\cdot", "x"), (r"\\mid", "|"),
    (r"\\ldots|\\cdots|\\dots", "..."), (r"\\forall", "for all"), (r"\\exists", "there exists"),
    (r"\\notin", " not in "), (r"\\in(?![a-zA-Z])", " in "), (r"\\cup", " U "), (r"\\cap", " intersect "),
    (r"\\emptyset|\\varnothing", "{}"), (r"\\infty", "infinity"), (r"\\sum", "Sum"), (r"\\prod", "Product"),
    (r"\\lceil", "ceil("), (r"\\rceil", ")"), (r"\\lfloor", "floor("), (r"\\rfloor", ")"),
    (r"\\langle", "<"), (r"\\rangle", ">"), (r"\\bmod|\\mod", " mod "),
    (r"\\varepsilon", "epsilon"), (r"\\left|\\right", ""), (r"\\[,;:! ]", " "),
    (r"\\\{", "{"), (r"\\\}", "}"), (r"\\%", "%"), (r"\\_", "_"),
]


def _braced(_n=1):       # one level of {...}
    return r"\{([^{}]*)\}"


def _math_to_text(m):
    s = m.group(1)
    s = re.sub(r"\\\\(?=[A-Za-z{}%_,;:! ])", r"\\", s)               # '\\{'  ->  '\{'
    s = re.sub(r"\\(?:text|textbf|textit|mathrm|mathbf|mathit|mathbb|operatorname)" + _braced(1), r"\1", s)
    s = re.sub(r"\\frac" + _braced(1) + _braced(2), r"(\1)/(\2)", s)
    s = re.sub(r"\\binom" + _braced(1) + _braced(2), r"C(\1,\2)", s)
    s = re.sub(r"\\sqrt" + _braced(1), r"sqrt(\1)", s)
    for pat, rep in _MATH_SIMPLE:
        s = re.sub(pat, rep, s)
    s = re.sub(r"([\^_])\{([^{}]*)\}", lambda k: k.group(1) + (k.group(2) if len(k.group(2)) == 1 else "(" + k.group(2) + ")"), s)
    s = re.sub(r"\\([A-Za-z]+)", r"\1", s)                               # any leftover \word -> word
    return re.sub(r"[ ]{2,}", " ", s).strip()


def _strip_math(line):
    line = re.sub(r"\$\$(.+?)\$\$", _math_to_text, line)
    return re.sub(r"\$([^$\n]+?)\$", _math_to_text, line)


_PDF_MAP = {
    "\u20b9": "Rs. ", "\u2192": "->", "\u2190": "<-", "\u21d2": "=>", "\u21d4": "<=>", "\u2264": "<=",
    "\u2265": ">=", "\u2260": "!=", "\u2248": "~", "\u221e": "infinity", "\u2211": "Sum", "\u2208": " in ",
    "\u2205": "{}", "\u2229": " intersect ", "\u222a": " U ", "\u221a": "sqrt", "\u03b8": "theta",
    "\u03c0": "pi", "\u03b1": "alpha", "\u03b2": "beta", "\u03b3": "gamma", "\u03bb": "lambda", "\u03bc": "mu",
    "\u03c3": "sigma", "\u03a3": "Sigma", "\u0394": "Delta", "\u03a9": "Omega", "\u0398": "Theta",
    "\u2713": "[x]", "\u2714": "[x]", "\u2705": "[OK]", "\u274c": "[X]", "\u2013": "-", "\u2014": "-",
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u2022": "-",
    "\u00a0": " ", "\u2212": "-",
}


_PDF_MAP.update({
    "\u03b5": "epsilon", "\u03f5": "epsilon", "\u03b4": "delta", "\u03c9": "w", "\u03a6": "{}", "\u03c6": "phi",
    "\u0393": "Gamma", "\u03c4": "tau", "\u03c1": "rho", "\u03b7": "eta", "\u03c8": "psi", "\u03ba": "kappa",
    "\u2209": " not in ", "\u2286": " subset of ", "\u2282": " proper subset of ", "\u2200": "for all ",
    "\u2203": "there exists ", "\u2227": " AND ", "\u2228": " OR ", "\u2295": " XOR ", "\u2261": "==",
    "\u2194": "<->", "\u27f9": "=>", "\u27f6": "->", "\u2217": "*", "\u22c5": ".", "\u2218": "o",
    "\u2032": "'", "\u2308": "ceil(", "\u2309": ")", "\u230a": "floor(", "\u230b": ")",
    "\u2115": "N", "\u2124": "Z", "\u211d": "R", "\u220f": "Product",
    "\u207f": "^n", "\u2071": "^i", "\u02b2": "^j", "\u2070": "^0", "\u2074": "^4", "\u2075": "^5",
    "\u2076": "^6", "\u2077": "^7", "\u2078": "^8", "\u2079": "^9", "\u207a": "^+", "\u207b": "^-",
    "\u2080": "0", "\u2081": "1", "\u2082": "2", "\u2083": "3", "\u2084": "4", "\u2085": "5", "\u2086": "6",
    "\u2087": "7", "\u2088": "8", "\u2089": "9", "\u2099": "n", "\u1d62": "i",
    "\u208a": "+", "\u208b": "-", "\u208c": "=", "\u208d": "(", "\u208e": ")",
})


def clean_for_export(text, pdf=False):
    """Markdown / LaTeX -> readable plain text. pdf=True also drops characters the PDF font cannot draw."""
    if not text:
        return ""
    t = _repair_escapes(str(text).replace("\r\n", "\n"))
    out, in_code = [], False
    for line in t.split("\n"):
        if re.match(r"^\s*```", line):                 # code fence: drop the fence line, keep the code
            in_code = not in_code
            continue
        if in_code:
            out.append(line.rstrip())
            continue
        s = _strip_math(line)
        if s.strip() == "$$":
            continue
        s = re.sub(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$", r"\1", s)                    # headings
        if "|" in s and re.match(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$", s):
            continue                                                                   # table separator row
        if re.match(r"^\s*\|.*\|\s*$", s):
            s = " | ".join(c.strip() for c in s.strip().strip("|").split("|"))      # table row
        if re.match(r"^\s*[-*_]{3,}\s*$", s):
            s = "-" * 40
        s = re.sub(r"^(\s*)\*\s+", r"\1- ", s)                                       # '* item' -> '- item'
        s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
        s = re.sub(r"__(.+?)__", r"\1", s)
        s = re.sub(r"(?<![\w*])\*(?=\S)(.+?)(?<=\S)\*(?![\w*])", r"\1", s)
        s = re.sub(r"`([^`\n]+)`", r"\1", s)
        out.append(s.rstrip())
    t = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()
    if pdf:
        for k, v in _PDF_MAP.items():
            t = t.replace(k, v)
        t = t.encode("latin-1", "ignore").decode("latin-1")
    return t


# ----------------------------------------------------------------------------------------------
# 3. Command line: status / generate / export   (everything below needs the project environment)
# ----------------------------------------------------------------------------------------------
def _import_project():
    os.environ.setdefault("ADMIN_PASSWORD", "repeated-solutions-run")   # keeps config.py quiet
    sys.path.insert(0, str(BASE))
    import config
    try:
        from src.question_index import load_question_bank
        from src.exam_intelligence import ExamIntelligence
    except ImportError:
        from question_index import load_question_bank
        from exam_intelligence import ExamIntelligence
    return config, load_question_bank, ExamIntelligence


def _subjects_and_clusters():
    config, load_question_bank, ExamIntelligence = _import_project()
    bank = load_question_bank(Path(config.FAISS_DIR) / "_question_bank.json")
    if not bank:
        sys.exit("Question bank is empty or missing in faiss_store/. Run the app / rebuild_questions.py first.")
    intel = ExamIntelligence()
    subjects = sorted({r.subject.strip().upper() for r in bank if r.subject})
    result = {}
    for s in subjects:
        years = {r.year for r in bank if r.subject.strip().upper() == s and r.year}
        result[s] = (intel.get_most_repeated_topics_from_bank(bank, subject=s, top_k=TOP_K), len(years))
    return config, result


def _items(subject, clusters):
    items = []
    for i, r in enumerate(clusters, 1):
        q = r["representative_question"]
        items.append({
            "index": i, "question": q, "marks": r.get("marks") or DEFAULT_MARKS,
            "years": r.get("years", []), "sittings": len(r.get("years", [])),
            "subject": r.get("subject") or subject, "answer": lookup_solution(r.get("subject") or subject, q),
        })
    return items


def cmd_status():
    _cfg, data = _subjects_and_clusters()
    print(f"{'Subject':8} {'Repeated Qs':>12} {'Solved':>8} {'Missing':>8}   (sittings in bank)")
    tot = solved = 0
    for s, (clusters, n_years) in data.items():
        items = _items(s, clusters)
        have = sum(1 for it in items if it["answer"])
        tot += len(items)
        solved += have
        note = "" if items else "  <- no question repeated across 2+ sittings"
        print(f"{s:8} {len(items):12d} {have:8d} {len(items) - have:8d}   ({n_years}){note}")
    print(f"{'TOTAL':8} {tot:12d} {solved:8d} {tot - solved:8d}")
    if tot > solved:
        print("\nRun:  python repeated_solutions.py generate     (overnight, resumable)")


def _looks_failed(ans):
    a = (ans or "").strip()
    return len(a) < MIN_ANSWER_CHARS or "don't have enough information" in a.lower() or a.startswith("[Error")


def _save_generated(data):
    SOL_DIR.mkdir(parents=True, exist_ok=True)
    tmp = GEN_FILE.with_name(GEN_FILE.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, GEN_FILE)            # a power cut can never leave half a file


def cmd_generate(only):
    config, data = _subjects_and_clusters()
    todo = []
    for s, (clusters, _n) in data.items():
        if only and s not in only:
            continue
        for it in _items(s, clusters):
            if not it["answer"]:
                todo.append((s, it))
    if not todo:
        print("Nothing to generate: every repeated question already has a solution.")
        return
    print(f"{len(todo)} repeated questions have no solution yet. Loading the pipeline (Ollama must be running)...")
    try:
        from src.rag_pipeline import RAGPipeline
    except ImportError:
        from rag_pipeline import RAGPipeline
    rag = RAGPipeline()
    known = {x.strip().upper(): x for x in rag.get_available_subjects()}
    store = {}
    if GEN_FILE.exists():
        try:
            store = json.loads(GEN_FILE.read_text(encoding="utf-8"))
        except Exception:
            sys.exit(f"{GEN_FILE} is not valid JSON. Rename it and run again (nothing was changed).")
    done = failed = 0
    t_all = time.time()
    for n, (s, it) in enumerate(todo, 1):
        rag_subject = known.get(s)
        if rag_subject is None:
            print(f"   note: '{s}' is not an index name; letting the pipeline auto-detect the subject")
        t0 = time.time()
        try:
            res = rag.ask(it["question"], marks=it["marks"], subject=rag_subject, plain_answer_only=True)
            ans = (res.get("answer") or "").strip()
        except Exception as e:
            ans, res = f"[Error: {e}]", {}
        if _looks_failed(ans):
            failed += 1
            status = "SKIPPED (empty / refusal / error) - not saved"
        else:
            store.setdefault(s, {})[it["question"]] = {
                "marks": it["marks"], "years": it["years"], "answer": ans, "source": "generated",
                "generated_at": time.strftime("%Y-%m-%d %H:%M"),
            }
            _save_generated(store)
            done += 1
            status = "ok"
        per = (time.time() - t_all) / n
        eta = per * (len(todo) - n) / 60
        print(f"[{n}/{len(todo)}] {time.time() - t0:5.0f}s {status}  {s} | {it['question'][:60]}  (~{eta:.0f} min left)", flush=True)
    print(f"\nFinished: {done} saved, {failed} skipped. Run `python repeated_solutions.py status`, then `export`.")


def build_text(subject, items, n_sittings=None):
    try:
        import config
        header_org = f"{config.UNIVERSITY_NAME} - {config.DEPARTMENT} Department"
    except Exception:
        header_org = ""
    lines = ["=" * 70, "uniRAG - MOST REPEATED EXAM QUESTIONS WITH MODEL ANSWERS",
             f"Subject: {subject}", header_org, f"Generated: {time.strftime('%d %B %Y')}", "=" * 70, ""]
    if not items:
        lines += [f"No question was repeated across two or more exam sittings in the indexed papers"
                  + (f" ({n_sittings} sitting(s) available)." if n_sittings is not None else "."), ""]
    for it in items:
        years = ", ".join(it["years"]) or "N/A"
        lines += [f"QUESTION #{it['index']}  ({it['marks']} Marks)  -  asked in {it['sittings']} exam sittings [{years}]",
                  f"Question: {it['question']}", "-" * 70, "Model Answer:",
                  clean_for_export(it["answer"]) or "[Answer not generated yet]", "", "=" * 70, ""]
    return "\n".join(l for l in lines if l is not None)


def cmd_export():
    _cfg, data = _subjects_and_clusters()
    SOL_DIR.mkdir(parents=True, exist_ok=True)
    combined, missing = [], 0
    for s, (clusters, n_years) in data.items():
        items = _items(s, clusters)
        missing += sum(1 for it in items if not it["answer"])
        text = build_text(s, items, n_years)
        (SOL_DIR / f"{s}_Repeated_Questions_Answers.txt").write_text(text, encoding="utf-8")
        combined.append(f"\n\n{'#' * 20} SUBJECT: {s} {'#' * 20}\n\n" + text)
        print(f"wrote {s}_Repeated_Questions_Answers.txt  ({len(items)} questions)")
    (SOL_DIR / "ALL_SUBJECTS_Repeated_Questions_Answers.txt").write_text("".join(combined), encoding="utf-8")
    print(f"\nFiles are in {SOL_DIR}")
    if missing:
        print(f"WARNING: {missing} questions still have no solution (marked '[Answer not generated yet]'). Run `generate` first.")


if __name__ == "__main__":
    cmd = sys.argv[1].lower() if len(sys.argv) > 1 else "status"
    if cmd == "status":
        cmd_status()
    elif cmd == "generate":
        cmd_generate({a.strip().upper() for a in sys.argv[2:]})
    elif cmd == "export":
        cmd_export()
    else:
        sys.exit("usage: python repeated_solutions.py status | generate [SUBJECT ...] | export")